import csv
import json
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from sklearn.metrics import roc_auc_score

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.data.splits import STUDY
from knee_abnormality.modeling.model import (
    FRAME_ROTATION_RANGE_RAD,
    ROUTING_WIDTH_RANGE_MM,
    TARGET_ROUTES,
)

FAMILIES = {
    "OA": ("Medial OA", "Lateral OA", "PF OA"),
    "Meniscus": ("Medial Meniscus", "Lateral Meniscus"),
}
SIDE_HEAD_FLOOR = 0.99
SPREADS = 2.0
BOUND_MARGIN = 0.02


class Settings(CLISettings):
    model_config = describe(
        "Read a finished side-routed run against the campaign base, the best of the "
        "lineage and the same-fold sibling runs: per-target deltas with the sibling "
        "range, the routed targets' sum, compartment localisation, the side head, the "
        "learned geometry, the guardrail flags and whether the run joins the lineage"
    )
    run: Path = Field(description="Finished training run directory")
    base: Path = Field(description="Training run directory of the campaign base")
    lineage: list[Path] = Field(
        description="Training run directories whose best values are the marks a run "
        "must not fall below: the reference marks and every adopted run",
    )
    siblings: list[Path] = Field(
        description="Training run directories of same-fold runs of one recipe, for "
        "the per-target noise range",
    )
    output: Path = Field(description="JSON report path")


def predictions(run: Path, fold: int, frame: pl.DataFrame) -> pl.DataFrame:
    table = pl.read_csv(run / f"fold_{fold}" / "validation_predictions.csv").sort(STUDY)
    if table[STUDY].to_list() != frame[STUDY].to_list():
        raise ValueError(f"Prediction membership differs from the fold: {run}")
    return table


def scores(table: pl.DataFrame, truth: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        name: float(roc_auc_score(values, table[name].to_numpy()))
        for name, values in truth.items()
    }


def localisation(table: pl.DataFrame, truth: dict[str, np.ndarray]) -> dict[str, float]:
    result = {}
    for members in FAMILIES.values():
        anywhere = np.any([truth[name] for name in members], axis=0)
        for name in members:
            keep = anywhere
            if truth[name][keep].min() == truth[name][keep].max():
                continue
            result[name] = float(
                roc_auc_score(truth[name][keep], table[name].to_numpy()[keep])
            )
    return result


def last_logged(run: Path, fold: int) -> dict[str, float]:
    path = next((run / f"fold_{fold}" / "logs").glob("version_*/metrics.csv"))
    values = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                if value not in ("", None):
                    values[key] = float(value)
    return values


def main() -> None:
    settings = Settings()
    config = load_config(settings.run / "effective_config.json")
    fold = config.validation.folds[0]
    split = config.paths.data_root / "folds" / config.validation.split
    frame = (
        pl.read_csv(split / "studies.csv")
        .filter(pl.col("fold") == fold)
        .join(pl.read_csv(split / "targets.csv"), on=STUDY, validate="1:1")
        .sort(STUDY)
    )
    run = predictions(settings.run, fold, frame)
    targets = run.columns[2:]
    truth = {name: frame[name].to_numpy() >= 0.5 for name in targets}
    base = predictions(settings.base, fold, frame)
    run_scores, base_scores = scores(run, truth), scores(base, truth)
    sibling_tables = [predictions(path, fold, frame) for path in settings.siblings]
    sibling_scores = [scores(table, truth) for table in sibling_tables]
    sibling_local = [localisation(table, truth) for table in sibling_tables]
    per_target = {}
    for name in targets:
        deltas = [sibling[name] - base_scores[name] for sibling in sibling_scores]
        delta = run_scores[name] - base_scores[name]
        per_target[name] = {
            "auc": run_scores[name],
            "delta": delta,
            "routed": name in TARGET_ROUTES,
            "positives": int(truth[name].sum()),
            "sibling_min": min(deltas) if deltas else None,
            "sibling_max": max(deltas) if deltas else None,
            "below_all_siblings": bool(deltas) and delta < min(deltas),
            "base_below_all_siblings": bool(deltas) and min(deltas) > 0,
            "above_all_siblings": bool(deltas) and delta > max(deltas),
        }
    run_local, base_local = localisation(run, truth), localisation(base, truth)
    logged = last_logged(settings.run, fold)
    low, high = ROUTING_WIDTH_RANGE_MM
    span = high - low
    bound = float(np.degrees(FRAME_ROTATION_RANGE_RAD))
    widths = {k: v for k, v in logged.items() if k.startswith("routing_width_")}
    angles = {k: v for k, v in logged.items() if k.startswith("frame_rotation_")}
    routed_sum = sum(v["delta"] for v in per_target.values() if v["routed"])
    lineage_tables = [predictions(path, fold, frame) for path in settings.lineage]
    lineage_scores = [scores(table, truth) for table in lineage_tables]
    lineage_local = [localisation(table, truth) for table in lineage_tables]
    routed_mark = max(
        sum(member[name] for name in TARGET_ROUTES) for member in lineage_scores
    )
    routed_absolute = sum(run_scores[name] for name in TARGET_ROUTES)
    local_mark = {
        name: max(member[name] for member in lineage_local) for name in run_local
    }
    macro_delta = float(
        np.mean(list(run_scores.values())) - np.mean(list(base_scores.values()))
    )
    routed_spread = float(
        np.std(
            [sum(sibling[name] for name in TARGET_ROUTES) for sibling in sibling_scores]
        )
    )
    local_spread = {
        name: float(np.std([sibling[name] for sibling in sibling_local]))
        for name in run_local
    }
    flags = {
        "macro_below_base": macro_delta < 0,
        "routed_sum_below_best": routed_absolute - routed_mark
        < -SPREADS * routed_spread,
        "localisation_below_best": sorted(
            name
            for name in run_local
            if run_local[name] - local_mark[name] < -SPREADS * local_spread[name]
        ),
        "own_targets_below_all_siblings": sorted(
            name
            for name, value in per_target.items()
            if not value["routed"] and value["below_all_siblings"]
        ),
        "side_head_below_floor": logged.get("val_side_head_accuracy", 0.0)
        < SIDE_HEAD_FLOOR,
        "width_at_a_limit": sorted(
            name
            for name, value in widths.items()
            if min(value - low, high - value) < BOUND_MARGIN * span
        ),
        "angle_at_its_bound": sorted(
            name
            for name, value in angles.items()
            if bound - abs(value) < BOUND_MARGIN * bound
        ),
    }
    report = {
        "run": str(settings.run),
        "base": str(settings.base),
        "siblings": len(sibling_scores),
        "macro_auc": float(np.mean(list(run_scores.values()))),
        "macro_delta": macro_delta,
        "routed_sum": routed_sum,
        "routed_sum_against_best": routed_absolute - routed_mark,
        "routed_sum_sibling_sd": routed_spread,
        "own_sum": sum(v["delta"] for v in per_target.values() if not v["routed"]),
        "per_target": per_target,
        "localisation": {
            name: {
                "auc": run_local[name],
                "delta": run_local[name] - base_local[name],
                "against_best": run_local[name] - local_mark[name],
                "sibling_sd": local_spread[name],
            }
            for name in run_local
        },
        "side_head": {
            key: logged.get(key)
            for key in (
                "val_side_head_accuracy",
                "val_side_head_coordinate_agreement",
                "val_side_probability_coordinate_agreement",
            )
        },
        "learned_geometry": {**widths, **angles},
        "flags": flags,
    }
    new_own = [
        name
        for name in flags["own_targets_below_all_siblings"]
        if not per_target[name]["base_below_all_siblings"]
    ]
    report["new_own_targets_below_all_siblings"] = new_own
    report["joins_lineage"] = not (
        flags["macro_below_base"]
        or flags["routed_sum_below_best"]
        or flags["localisation_below_best"]
        or new_own
        or flags["side_head_below_floor"]
        or flags["width_at_a_limit"]
        or flags["angle_at_its_bound"]
    )
    settings.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
