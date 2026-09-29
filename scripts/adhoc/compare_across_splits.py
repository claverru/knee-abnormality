import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD
from knee_abnormality.evaluation.metrics import aucs, bootstrap_auc
from knee_abnormality.provenance import copy_sources


class CompareCLI(CLISettings):
    candidate: Path
    candidate_split: Path = Path("data/folds/all_5fold_scanner_v1")
    references: dict[str, Path]
    reference_split: Path = Path("data/folds/all_5fold_v3")
    output_dir: Path
    fold: int = Field(default=0, ge=0)
    draws: int = Field(default=2000, ge=200)
    seed: int = 20260921


def held_out(split: Path, fold: int) -> pl.DataFrame:
    return (
        pl.read_csv(split / "studies.csv")
        .filter(pl.col("fold") == fold)
        .select(STUDY, "report_group", "is_gold")
    )


def compare(settings: CompareCLI) -> None:
    if settings.output_dir.exists():
        raise ValueError(f"Output directory exists: {settings.output_dir}")
    definitions = {
        "candidate_targets": pl.read_csv(settings.candidate_split / "targets.csv"),
        "reference_targets": pl.read_csv(settings.reference_split / "targets.csv"),
    }
    targets = definitions["candidate_targets"].columns[1:]
    candidate_fold = held_out(settings.candidate_split, settings.fold)
    reference_fold = held_out(settings.reference_split, settings.fold)
    shared = reference_fold.join(candidate_fold.select(STUDY), on=STUDY).sort(STUDY)
    runs = {"candidate": settings.candidate, **settings.references}
    predictions = {
        name: pl.read_csv(root / f"fold_{settings.fold}" / "validation_predictions.csv")
        for name, root in runs.items()
    }
    folds = {
        "candidate": candidate_fold,
        **{name: reference_fold for name in settings.references},
    }

    def evaluate(rows: pl.DataFrame, labels: pl.DataFrame, name: str):
        joined = (
            rows.select(STUDY, "report_group")
            .join(labels, on=STUDY, validate="1:1")
            .join(predictions[name], on=STUDY, suffix="_p", validate="1:1")
            .sort(STUDY)
        )
        truth = (joined.select(targets).to_numpy() >= TARGET_THRESHOLD).astype(np.int8)
        scores = joined.select([f"{t}_p" for t in targets]).to_numpy().astype(float)
        return truth, scores, joined["report_group"].to_numpy()

    rows = []
    for name in runs:
        for definition, labels in definitions.items():
            for scope, frame in (("own_fold", folds[name]), ("shared", shared)):
                truth, scores, _ = evaluate(frame, labels, name)
                result = aucs(truth, scores, targets)
                rows.append(
                    {
                        "run": name,
                        "targets": definition,
                        "studies": scope,
                        "n": len(truth),
                        "macro_auc": result["macro_auc"],
                        **result["per_target_auc"],
                    }
                )
    table = pl.DataFrame(rows)

    rng = np.random.default_rng(settings.seed)
    intervals = {}
    for definition, labels in definitions.items():
        truth, candidate, groups = evaluate(shared, labels, "candidate")
        unique, inverse = np.unique(groups, return_inverse=True)
        counts = rng.multinomial(
            len(unique), np.full(len(unique), 1 / len(unique)), size=settings.draws
        )[:, inverse]
        positive = counts @ truth
        counts = counts[
            ((positive > 0) & (positive < counts.sum(axis=1, keepdims=True))).all(
                axis=1
            )
        ]
        base = bootstrap_auc(truth, candidate, counts)
        for name in settings.references:
            _, reference, _ = evaluate(shared, labels, name)
            delta = base - bootstrap_auc(truth, reference, counts)
            intervals[f"{definition}:candidate-{name}"] = {
                "delta_mean": float(delta.mean()),
                "central_80": np.percentile(delta, [10, 90]).tolist(),
                "central_95": np.percentile(delta, [2.5, 97.5]).tolist(),
                "positive_fraction": float((delta > 0).mean()),
                "valid_draws": len(counts),
            }

    settings.output_dir.mkdir(parents=True)
    table.write_csv(settings.output_dir / "per_target.csv")
    summary = {
        "candidate": str(settings.candidate),
        "references": {k: str(v) for k, v in settings.references.items()},
        "splits": {
            "candidate": str(settings.candidate_split),
            "reference": str(settings.reference_split),
        },
        "shared_held_out_studies": shared.height,
        "shared_gold_studies": int(shared["is_gold"].sum()),
        "shared_positives": {
            definition: dict(
                zip(
                    targets,
                    evaluate(shared, labels, "candidate")[0].sum(axis=0).tolist(),
                    strict=True,
                )
            )
            for definition, labels in definitions.items()
        },
        "paired_report_group_bootstrap": intervals,
        "sources": copy_sources((Path(__file__),), settings.output_dir / "source"),
    }
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with pl.Config(tbl_rows=40, tbl_cols=17, tbl_width_chars=260, float_precision=3):
        print(table)
    print(json.dumps(intervals, indent=2))


def main() -> None:
    try:
        compare(CompareCLI())
    except (KeyError, OSError, TypeError, ValueError, SettingsError) as error:
        print(f"Cross-split comparison error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
