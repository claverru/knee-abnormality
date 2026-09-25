import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import polars as pl
import torch
from pydantic import Field
from pydantic_settings import SettingsError

from knee_abnormality.autoresearch.campaign.gates import without_fields
from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY, TARGET_THRESHOLD
from knee_abnormality.evaluation.metrics import (
    COMPARTMENT_PAIRS,
    aucs,
    bootstrap_auc,
    compartment_metrics,
    logit_margin,
    validation_metrics,
)
from knee_abnormality.inference.checkpoint import read_checkpoint
from knee_abnormality.provenance import copy_sources, sha256_file

ANALYSIS_SOURCES = (Path(__file__), Path("src/knee_abnormality/evaluation/metrics.py"))


class CompareCLI(CLISettings):
    reference: Path
    candidate: Path
    output_dir: Path
    allowed_config_differences: list[str] = Field(default_factory=lambda: ["name"])
    allowed_input_identity_differences: list[str] = Field(default_factory=list)
    reference_identity_exceptions: list[str] = Field(default_factory=list)
    fold: int | None = Field(default=None, ge=0)
    headers: Path = Path(
        "outputs/analysis/sharing-disagreement-20260914/study_headers.parquet"
    )
    bootstrap_draws: int = Field(default=2000, ge=100)
    seed: int = 20260914


def fingerprint(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def flatten(value: object, prefix: str = "") -> dict:
    if isinstance(value, dict):
        return {
            key: item
            for name, child in value.items()
            for key, item in flatten(child, f"{prefix}{name}.").items()
        }
    return {prefix.removesuffix("."): value}


def differences(reference: dict, candidate: dict) -> dict:
    return {
        key: {"reference": reference.get(key), "candidate": candidate.get(key)}
        for key in sorted(reference.keys() | candidate.keys())
        if reference.get(key) != candidate.get(key)
    }


def unexpected(changed: dict, allowed: list[str]) -> list[str]:
    return [
        key
        for key in changed
        if not any(key == item or key.startswith(f"{item}.") for item in allowed)
    ]


def binary_cross_entropy(truth: np.ndarray, values: np.ndarray) -> float | None:
    if not len(truth):
        return None
    p = np.clip(values, 1e-7, 1 - 1e-7)
    return float(-(truth * np.log(p) + (1 - truth) * np.log1p(-p)).mean())


def interval(delta: np.ndarray, valid: int) -> dict:
    return {
        "valid_draws": valid,
        "delta_mean": float(delta.mean()) if valid else None,
        "delta_percentile_95": np.percentile(delta, [2.5, 97.5]).tolist()
        if valid
        else None,
    }


def load_run(root: Path, fold: int, frame: pl.DataFrame, targets: list[str]) -> dict:
    fold_dir = root / f"fold_{fold}"
    selected = json.loads((fold_dir / "selected_checkpoint.json").read_text())
    checkpoint_path = fold_dir / Path(selected["path"]).name
    checkpoint = read_checkpoint(checkpoint_path)
    epoch = checkpoint["epoch"]
    state = {
        name.removeprefix("model."): value
        for name, value in checkpoint["state_dict"].items()
        if name.startswith("model.")
    }
    del checkpoint
    prediction_path = fold_dir / "validation_predictions.csv"
    prediction = pl.read_csv(prediction_path).sort(STUDY)
    if prediction[STUDY].to_list() != frame[STUDY].to_list():
        raise ValueError(f"Prediction membership differs from the fold: {root}")
    values = prediction.select(targets).to_numpy().astype(float)
    truth = frame.select(targets).to_numpy().astype(float)
    metrics = validation_metrics(truth, values, targets, frame["is_gold"].to_numpy())
    saved = json.loads(
        (fold_dir / "epochs" / f"metrics_epoch_{epoch:03d}.json").read_text()
    )
    recorded = selected.get("prediction_macro_auc", saved["macro_auc"])
    if abs(metrics["macro_auc"] - recorded) > 1e-12:
        raise ValueError(f"Selected predictions do not reproduce recorded AUC: {root}")
    epoch_paths = sorted((fold_dir / "epochs").glob("metrics_epoch_*.json"))
    epochs = []
    for path in epoch_paths:
        recorded = json.loads(path.read_text())
        epochs.append(
            {
                "epoch": int(path.stem.rsplit("_", 1)[1]) + 1,
                "macro_auc": recorded["macro_auc"],
                "soft_target_bce": recorded["soft_target_bce"],
                "gold_macro_auc": recorded["gold"]["macro_auc"],
                "report_derived_macro_auc": recorded["report_derived"]["macro_auc"],
            }
        )
    files = [
        root / "effective_config.json",
        root / "provenance.json",
        root / "source_snapshot.zip",
        checkpoint_path,
        prediction_path,
        *sorted(fold_dir.glob("*.json")),
        *epoch_paths,
    ]
    return {
        "predictions": values,
        "state": state,
        "input_identity": json.loads((fold_dir / "input_identity.json").read_text()),
        "record": {
            "selected_epoch": epoch + 1,
            "metrics": metrics,
            "parameters": sum(value.numel() for value in state.values()),
            "epochs": epochs,
            "timing": json.loads((fold_dir / "timing.json").read_text()),
            "files": [fingerprint(path) for path in files],
        },
    }


def compare(settings: CompareCLI) -> None:
    torch.set_num_threads(1)
    roots = {"reference": settings.reference, "candidate": settings.candidate}
    configs = {
        name: json.loads((root / "effective_config.json").read_text())
        for name, root in roots.items()
    }
    config_diff = differences(
        flatten(configs["reference"]), flatten(configs["candidate"])
    )
    if rejected := unexpected(config_diff, settings.allowed_config_differences):
        raise ValueError(f"Unexpected configuration differences: {rejected}")
    config = configs["candidate"]
    fold = settings.fold
    if fold is None:
        folds = config["validation"]["folds"]
        if len(folds) != 1:
            raise ValueError(f"Runs have several validation folds {folds}; pass --fold")
        fold = folds[0]
    if fold not in config["validation"]["folds"]:
        raise ValueError("Requested fold is absent from the compared configurations")
    split = Path(config["paths"]["data_root"]) / "folds" / config["validation"]["split"]
    targets_path, studies_path = split / "targets.csv", split / "studies.csv"
    labels = pl.read_csv(targets_path)
    targets = labels.columns[1:]
    frame = (
        pl.read_csv(studies_path)
        .filter(pl.col("fold") == fold)
        .join(labels, on=STUDY, validate="1:1")
        .join(
            pl.read_parquet(settings.headers).select(STUDY, "vendor"),
            on=STUDY,
            validate="1:1",
            how="left",
        )
        .sort(STUDY)
    )
    if frame["vendor"].null_count():
        raise ValueError("Missing acquisition metadata")
    loaded = {
        name: load_run(root, fold, frame, targets)
        for name, root in roots.items()
    }
    predictions = {name: run["predictions"] for name, run in loaded.items()}
    runs = {name: run["record"] for name, run in loaded.items()}
    identity_diff = differences(
        without_fields(
            loaded["reference"]["input_identity"],
            settings.reference_identity_exceptions,
        ),
        without_fields(
            loaded["candidate"]["input_identity"],
            settings.reference_identity_exceptions,
            required=False,
        ),
    )
    if rejected := unexpected(
        identity_diff, settings.allowed_input_identity_differences
    ):
        raise ValueError(f"Unexpected input identity differences: {rejected}")
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    states = {name: run["state"] for name, run in loaded.items()}
    parameter_differences = {
        name: [
            {
                "parameter": key,
                "shape": list(states[name][key].shape),
                "norm": float(states[name][key].float().norm()),
            }
            for key in sorted(states[name].keys() - states[other].keys())
        ]
        for name, other in (("reference", "candidate"), ("candidate", "reference"))
    }
    del loaded, states
    with (
        ZipFile(settings.reference / "source_snapshot.zip") as left,
        ZipFile(settings.candidate / "source_snapshot.zip") as right,
    ):
        left_names, right_names = set(left.namelist()), set(right.namelist())
        source_changes = [
            name
            for name in sorted(left_names | right_names)
            if name.startswith(("src/", "configs/"))
            and (
                name not in left_names
                or name not in right_names
                or left.read(name) != right.read(name)
            )
        ]
    truth = frame.select(targets).to_numpy().astype(float)
    hard = (truth >= TARGET_THRESHOLD).astype(int)
    indices = {
        family: [targets.index(target) for target in pair]
        for family, pair in COMPARTMENT_PAIRS.items()
    }
    families = {
        "all": list(range(len(targets))),
        **indices,
        "other_eight": [
            index
            for index in range(len(targets))
            if index not in [value for pair in indices.values() for value in pair]
        ],
    }
    compartments = {
        name: compartment_metrics(truth, values, targets)
        for name, values in predictions.items()
    }

    def stratum(mask: np.ndarray) -> dict:
        return {
            "studies": int(mask.sum()),
            "positive_counts": dict(
                zip(targets, hard[mask].sum(axis=0).tolist(), strict=True)
            ),
            **{
                name: {
                    **aucs(hard[mask], values[mask], targets, strict=False),
                    "soft_target_bce": binary_cross_entropy(truth[mask], values[mask]),
                }
                for name, values in predictions.items()
            },
        }

    vendors = frame["vendor"].to_numpy()
    strata = {
        f"vendor={vendor}": stratum(vendors == vendor)
        for vendor in sorted(frame["vendor"].unique().to_list())
    }
    groups, inverse = np.unique(frame["report_group"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(settings.seed)
    counts = rng.multinomial(
        len(groups),
        np.full(len(groups), 1 / len(groups)),
        size=settings.bootstrap_draws,
    )[:, inverse]
    positive = counts @ hard
    valid = ((positive > 0) & (positive < counts.sum(axis=1, keepdims=True))).all(
        axis=1
    )
    intervals = {}
    for family, columns in families.items():
        delta = bootstrap_auc(
            hard[:, columns], predictions["candidate"][:, columns], counts[valid]
        ) - bootstrap_auc(
            hard[:, columns], predictions["reference"][:, columns], counts[valid]
        )
        intervals[family] = interval(delta, int(valid.sum()))
    for family, (medial, lateral) in indices.items():
        exclusive = hard[:, medial] != hard[:, lateral]
        side = hard[exclusive, medial][:, None]
        weights = counts[:, exclusive]
        medial_weight = weights @ side[:, 0]
        usable = (medial_weight > 0) & (medial_weight < weights.sum(axis=1))
        delta = bootstrap_auc(
            side,
            logit_margin(predictions["candidate"][exclusive], medial, lateral)[:, None],
            weights[usable],
        ) - bootstrap_auc(
            side,
            logit_margin(predictions["reference"][exclusive], medial, lateral)[:, None],
            weights[usable],
        )
        intervals[f"{family}_margin"] = interval(delta, int(usable.sum()))
    per_target = [
        {
            "target": target,
            "positives": int(hard[:, index].sum()),
            **{name: runs[name]["metrics"]["per_target_auc"][target] for name in runs},
        }
        for index, target in enumerate(targets)
    ]
    for row in per_target:
        row["delta"] = row["candidate"] - row["reference"]
    report = {
        "settings": settings.model_dump(mode="json"),
        "fold": fold,
        "studies": frame.height,
        "targets": targets,
        "runs": runs,
        "per_target": per_target,
        "family_means": {
            family: {
                name: float(
                    np.mean(
                        [
                            runs[name]["metrics"]["per_target_auc"][targets[index]]
                            for index in columns
                        ]
                    )
                )
                for name in runs
            }
            for family, columns in families.items()
        },
        "compartments": compartments,
        "prediction_change": {
            "mean_abs_probability_difference": float(
                np.abs(predictions["candidate"] - predictions["reference"]).mean()
            ),
            "per_target_spearman": {
                target: float(
                    np.corrcoef(
                        np.argsort(np.argsort(predictions["candidate"][:, index])),
                        np.argsort(np.argsort(predictions["reference"][:, index])),
                    )[0, 1]
                )
                for index, target in enumerate(targets)
            },
        },
        "config_differences": config_diff,
        "input_identity_differences": sorted(identity_diff),
        "reference_identity_exceptions": settings.reference_identity_exceptions,
        "parameters_only_in_run": parameter_differences,
        "changed_source_archive_members": source_changes,
        "inputs": [
            fingerprint(path) for path in (targets_path, studies_path, settings.headers)
        ],
        "analysis_sources": [fingerprint(path) for path in ANALYSIS_SOURCES],
        "strata": strata,
        "paired_report_group_bootstrap": {
            "groups": len(groups),
            "draws": settings.bootstrap_draws,
            "seed": settings.seed,
            "unit": "report group; margin intervals keep each draw's weights on exclusive-positive studies",
            "intervals": intervals,
        },
        "limitations": [
            "Saved predictions and CPU-loaded checkpoint tensors only; no model execution or GPU access.",
            "One training seed and one development fold; intervals exclude training-seed and checkpoint-selection variability.",
            "Vendor is an acquisition proxy, not a site label; hidden public composition is unknown.",
            "Allowed configuration and identity differences change together; their separate contributions are not identifiable.",
        ],
    }
    (settings.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    copy_sources(ANALYSIS_SOURCES, settings.output_dir / "source")
    print(
        json.dumps(
            {
                "artifact": str(settings.output_dir / "comparison.json"),
                "selected_epoch": {
                    name: run["selected_epoch"] for name, run in runs.items()
                },
                "macro_auc": {
                    name: run["metrics"]["macro_auc"] for name, run in runs.items()
                },
                "family_means": report["family_means"],
                "compartments": {
                    name: {
                        f"{family}_{key}": values[key]
                        for family, values in compartment.items()
                        for key in ("margin_auc", "lateral_cross_talk")
                    }
                    for name, compartment in compartments.items()
                },
                "intervals": intervals,
            },
            indent=2,
        )
    )


def main() -> None:
    try:
        compare(CompareCLI())
    except (KeyError, OSError, TypeError, ValueError, SettingsError) as error:
        print(f"Fold comparison error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
