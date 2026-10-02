import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset
from knee_abnormality.data.preprocessing.sampling import (
    build_pool,
    geometry_groups,
    limiting_reach,
    study_budgets,
    union_support,
)
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.data.splits import STUDY

MULTIPLIERS = (1, 2, 4)
STATE = {}


class Settings(CLISettings):
    model_config = describe(
        "Measure, for every training study and series type, how many distinct slice "
        "positions the sampler's candidate pool can supply against the per-type "
        "triplet budget at image multipliers 1, 2 and 4, with the pool's extent and "
        "slice gap. Headers only; no pixels are read."
    )

    config: Path = Field(description="effective_config.json of the measured recipe")
    output_dir: Path = Field(description="Output directory")
    workers: int = Field(default=8, ge=1, le=8, description="Worker processes")
    studies: int | None = Field(
        default=None, ge=1, description="Stop after this many studies"
    )


def initialise(config_path):
    config = load_config(config_path)
    index = build_series_index(config.paths.data_root, config.data, "train")
    studies = pl.DataFrame({STUDY: sorted(index)})
    STATE["config"] = config
    STATE["index"] = index
    STATE["dataset"] = KneeStudyDataset(studies, index, config.data, (), None)


def measure_study(study_uid):
    dataset, data = STATE["dataset"], STATE["config"].data
    study = dataset._index_study_geometry(STATE["index"][study_uid])
    failures = []
    slot_sources = [
        dataset._slot_sources(study, slot, failures) for slot in range(len(study.slots))
    ]
    present = tuple(len(sources) for _, sources in slot_sources)
    budgets = {
        multiplier: study_budgets(
            tuple(kind.max_images * multiplier // 3 for kind in data.series_types),
            present,
            data.study_triplet_total * multiplier,
        )
        for multiplier in MULTIPLIERS
    }
    radius = data.image_size * data.spacing_mm / 2
    rows = []
    for slot, (_, sources) in enumerate(slot_sources):
        kind = data.series_types[slot]
        row = {
            STUDY: study_uid,
            "plane": kind.plane,
            "fluid_sensitive": kind.fluid_sensitive,
            "sources": len(sources),
            "base_budget": kind.max_images // 3,
            **{f"budget_x{m}": budgets[m][slot] for m in MULTIPLIERS},
        }
        if sources:
            pool = build_pool(
                tuple(sources),
                kind.plane,
                dataset._study_references[study_uid],
                data.spacing_mm,
                radius,
                set(),
                set(),
            )
            groups = geometry_groups(pool, radius)
            row.update(candidates=len(pool.candidates), capacity=len(groups))
            row["physical"] = pool.physical
            if pool.physical and groups:
                depths = np.sort(np.asarray([group[0].depth for group in groups]))
                intervals = union_support(pool.candidates)
                row.update(
                    reach_mm=limiting_reach(pool),
                    support_lo_mm=float(intervals[:, 0].min()),
                    support_hi_mm=float(intervals[:, 1].max()),
                    support_mm=float((intervals[:, 1] - intervals[:, 0]).sum()),
                    slice_gap_mm=float(np.median(np.diff(depths)))
                    if len(depths) > 1
                    else None,
                )
        rows.append(row)
    dataset._slice_index.clear()
    return rows


def share(frame, expression):
    return float(frame.select(expression.mean()).item())


def summarise(frame):
    rows = []
    studies = frame[STUDY].n_unique()
    for (plane, fluid), part in frame.group_by(
        "plane", "fluid_sensitive", maintain_order=True
    ):
        present = part.filter(pl.col("sources") > 0)
        physical = present.filter(pl.col("physical"))
        quantiles = {
            f"capacity_p{q}": float(present["capacity"].quantile(q / 100))
            for q in (10, 25, 50, 75, 90)
        }
        rows.append(
            {
                "plane": plane,
                "fluid_sensitive": fluid,
                "studies_with_type": present.height,
                "share_of_studies": present.height / studies,
                "series_per_study": float(present["sources"].mean()),
                **quantiles,
                "slice_gap_mm_p50": float(physical["slice_gap_mm"].median()),
                "support_mm_p50": float(physical["support_mm"].median()),
                "reach_mm_p50": float(physical["reach_mm"].median()),
                **{
                    f"budget_x{m}_mean": float(present[f"budget_x{m}"].mean())
                    for m in MULTIPLIERS
                },
                **{
                    f"capped_x{m}": share(
                        present, pl.col("capacity") < pl.col(f"budget_x{m}")
                    )
                    for m in MULTIPLIERS
                },
                **{
                    f"used_share_x{m}": share(
                        present,
                        pl.min_horizontal("capacity", f"budget_x{m}")
                        / pl.col("capacity"),
                    )
                    for m in MULTIPLIERS
                },
                "spacing_between_picks_x1_mm_p50": float(
                    (physical["support_mm"] / physical["budget_x1"]).median()
                ),
            }
        )
    return pl.DataFrame(rows)


def run(settings: Settings) -> dict:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(settings.config)
    index = build_series_index(config.paths.data_root, config.data, "train")
    studies = sorted(index)
    if settings.studies is not None:
        studies = studies[: settings.studies]
    rows = []
    with get_context("spawn").Pool(
        settings.workers, initialise, (settings.config,)
    ) as pool:
        for number, values in enumerate(
            pool.imap_unordered(measure_study, studies, chunksize=8), 1
        ):
            rows.extend(values)
            if number % 500 == 0 or number == len(studies):
                print(
                    json.dumps({"studies": number, "total": len(studies)}),
                    file=sys.stderr,
                    flush=True,
                )
    frame = pl.DataFrame(rows, infer_schema_length=None).sort(
        STUDY, "plane", "fluid_sensitive"
    )
    frame.write_parquet(settings.output_dir / "slice_capacity.parquet")
    summary = summarise(frame)
    summary.write_csv(settings.output_dir / "slice_capacity_summary.csv")
    totals = frame.group_by(STUDY).agg(
        *(
            pl.col(f"budget_x{m}").filter(pl.col("sources") > 0).sum()
            for m in MULTIPLIERS
        ),
        *(
            pl.min_horizontal("capacity", f"budget_x{m}")
            .filter(pl.col("sources") > 0)
            .sum()
            .alias(f"selected_x{m}")
            for m in MULTIPLIERS
        ),
    )
    return {
        "config": str(settings.config),
        "studies": len(studies),
        "mean_budget": {
            f"x{m}": float(totals[f"budget_x{m}"].mean()) for m in MULTIPLIERS
        },
        "mean_selected": {
            f"x{m}": float(totals[f"selected_x{m}"].mean()) for m in MULTIPLIERS
        },
    }


def main() -> None:
    print(json.dumps(run(Settings()), indent=2))


if __name__ == "__main__":
    main()
