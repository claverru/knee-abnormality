import os
import shutil
import subprocess
import time
from pathlib import Path

from knee_abnormality.autoresearch.isolation import environment
from knee_abnormality.autoresearch.storage import lock, read_json, sha256, write_json

INTERPRETATION = (
    "Per-label intervals are exploratory, unadjusted paired report-group bootstrap "
    "95% intervals for candidate minus reference. They do not provide simultaneous "
    "coverage across labels or account for training-seed, checkpoint-selection or "
    "repeated validation-selection uncertainty. Adoption requires macro gain >= 0.001 "
    "and a strictly positive central 80% macro interval; label diagnostics do not "
    "authorize per-label losses."
)


def diagnostics_path(comparison):
    return Path(comparison).with_name("per_label.json")


def read_diagnostics(comparison):
    comparison = Path(comparison)
    path = diagnostics_path(comparison)
    if not path.is_file():
        return None
    report = read_json(path)
    if report["comparison"]["sha256"] != sha256(comparison):
        raise ValueError(f"Label diagnostics refer to a changed comparison: {path}")
    return report


def experiment_diagnostics(exp):
    return read_diagnostics(Path(exp["directory"]) / "comparison/comparison.json")


def checked_path(record, root):
    path = (root / record["path"]).resolve()
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Recorded comparison input changed: {path}")
    return path


def write_diagnostics(comparison, root):
    import numpy as np
    import polars as pl

    from knee_abnormality.data import labels
    from knee_abnormality.evaluation import metrics

    comparison = comparison.resolve()
    with lock(comparison.with_name("per_label.lock")):
        if existing := read_diagnostics(comparison):
            return existing
        started = time.monotonic()
        report = read_json(comparison)
        targets = report["targets"]
        metric_source = next(
            item
            for item in report["analysis_sources"]
            if Path(item["path"]).name == "metrics.py"
        )
        if sha256(Path(metrics.__file__)) != metric_source["sha256"]:
            raise ValueError(
                "Diagnostic metrics source differs from the frozen evaluator"
            )
        inputs = {
            Path(item["path"]).name: checked_path(item, root)
            for item in report["inputs"]
        }
        frame = (
            pl.read_csv(inputs["studies.csv"])
            .filter(pl.col("fold") == report["fold"])
            .join(pl.read_csv(inputs["targets.csv"]), on=labels.STUDY, how="inner")
            .sort(labels.STUDY)
        )
        if (
            frame.height != report["studies"]
            or frame[labels.STUDY].n_unique() != frame.height
            or frame["report_group"].null_count()
        ):
            raise ValueError("Comparison cohort or report groups cannot be reproduced")
        hard = (frame.select(targets).to_numpy() >= labels.TARGET_THRESHOLD).astype(int)
        predictions = {}
        prediction_inputs = {}
        point_scores = {}
        for name in ("reference", "candidate"):
            record = next(
                item
                for item in report["runs"][name]["files"]
                if Path(item["path"]).name == "validation_predictions.csv"
            )
            path = checked_path(record, root)
            prediction = pl.read_csv(path).sort(labels.STUDY)
            if prediction[labels.STUDY].to_list() != frame[labels.STUDY].to_list():
                raise ValueError(f"Prediction membership differs from the fold: {path}")
            predictions[name] = prediction.select(targets).to_numpy().astype(float)
            if not np.isfinite(predictions[name]).all():
                raise ValueError(f"Non-finite prediction values: {path}")
            point_scores[name] = metrics.aucs(hard, predictions[name], targets)[
                "per_target_auc"
            ]
            prediction_inputs[name] = {"path": str(path), "sha256": record["sha256"]}
        bootstrap = report["paired_report_group_bootstrap"]
        groups, inverse = np.unique(
            frame["report_group"].to_numpy(), return_inverse=True
        )
        if len(groups) != bootstrap["groups"]:
            raise ValueError("Recorded report-group count differs")
        counts = np.random.default_rng(bootstrap["seed"]).multinomial(
            len(groups), np.full(len(groups), 1 / len(groups)), size=bootstrap["draws"]
        )[:, inverse]
        positive = counts @ hard
        valid = ((positive > 0) & (positive < counts.sum(axis=1, keepdims=True))).all(
            axis=1
        )
        weights = counts[valid]
        valid_draws = int(valid.sum())
        del counts
        rows = []
        deltas = []
        for index, target in enumerate(targets):
            row = report["per_target"][index]
            if row["target"] != target or row["positives"] != int(hard[:, index].sum()):
                raise ValueError(f"Recorded target or positive count differs: {target}")
            for name in predictions:
                score = point_scores[name][target]
                if not np.isclose(score, row[name], rtol=0, atol=1e-12):
                    raise ValueError(f"Recorded {name} AUC differs for {target}")
            if not np.isclose(
                row["delta"], row["candidate"] - row["reference"], rtol=0, atol=1e-12
            ):
                raise ValueError(f"Recorded AUC delta differs for {target}")
            delta = metrics.bootstrap_auc(
                hard[:, [index]], predictions["candidate"][:, [index]], weights
            ) - metrics.bootstrap_auc(
                hard[:, [index]], predictions["reference"][:, [index]], weights
            )
            deltas.append(delta)
            rows.append(
                {
                    **row,
                    "negatives": frame.height - row["positives"],
                    "bootstrap": interval(delta, valid_draws),
                }
            )
        macro_deltas = np.mean(deltas, axis=0)
        macro = {
            **interval(macro_deltas, valid_draws),
            "delta_percentile_80": np.percentile(macro_deltas, [10, 90]).tolist()
            if valid_draws
            else None,
        }
        recorded = bootstrap["intervals"]["all"]
        if macro["valid_draws"] != recorded["valid_draws"] or any(
            not np.allclose(macro[key], recorded[key], rtol=0, atol=1e-12)
            for key in ("delta_mean", "delta_percentile_95")
            if valid_draws
        ):
            raise ValueError(
                "Label resamples do not reproduce the recorded macro bootstrap"
            )
        sources = []
        for source in (Path(__file__), Path(metrics.__file__), Path(labels.__file__)):
            destination = comparison.parent / "per_label_source" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            sources.append({"path": str(destination), "sha256": sha256(destination)})
        result = {
            "comparison": {"path": str(comparison), "sha256": sha256(comparison)},
            "created_at": time.time(),
            "seconds": time.monotonic() - started,
            "fold": report["fold"],
            "studies": frame.height,
            "targets": targets,
            "target_threshold": labels.TARGET_THRESHOLD,
            "inputs": [
                {"path": str(path), "sha256": sha256(path)} for path in inputs.values()
            ],
            "predictions": prediction_inputs,
            "analysis_sources": sources,
            "bootstrap": {
                "groups": len(groups),
                "draws": bootstrap["draws"],
                "valid_draws": valid_draws,
                "seed": bootstrap["seed"],
                "unit": "report_group",
                "paired": True,
                "invalid_resample_rule": "discard draws missing either class for any target",
                "confidence_level": 0.95,
                "multiplicity_adjustment": "none",
            },
            "per_target": rows,
            "macro_bootstrap": macro,
            "macro_adoption_confidence_level": 0.8,
            "interpretation": INTERPRETATION,
        }
        write_json(diagnostics_path(comparison), result)
        return result


def interval(delta, valid):
    import numpy as np

    return {
        "valid_draws": valid,
        "delta_mean": float(delta.mean()) if valid else None,
        "delta_percentile_95": np.percentile(delta, [2.5, 97.5]).tolist()
        if valid
        else None,
    }


def compare(command_file, root):
    spec = read_json(command_file)
    source = Path(spec["source"])
    subprocess.run(spec["command"], cwd=source, env=environment(source), check=True)
    write_diagnostics(Path(spec["comparison"]), root)


def backfill(state, root):
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        POLARS_MAX_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    added, existing, unavailable, deferred = [], [], [], []
    for exp in state["experiments"].values():
        comparison = Path(exp["directory"]) / "comparison/comparison.json"
        if not comparison.is_file():
            unavailable.append(exp["id"])
            continue
        if not exp.get("eligible_decision"):
            deferred.append(exp["id"])
            continue
        present = diagnostics_path(comparison).is_file()
        write_diagnostics(comparison, root)
        (existing if present else added).append(exp["id"])
    return {
        "added": added,
        "existing": existing,
        "without_comparison": unavailable,
        "comparison_in_progress": deferred,
    }
