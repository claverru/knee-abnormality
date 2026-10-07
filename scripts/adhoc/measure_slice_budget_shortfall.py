import json
from multiprocessing import get_context
from pathlib import Path

import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset
from knee_abnormality.data.preprocessing.sampling import build_pool, select_triplets
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.data.splits import STUDY

DATASET = None
MULTIPLIERS = (1, 4, 6)
KEYS = ["multiplier", STUDY]


class Settings(CLISettings):
    model_config = describe(
        "Measure how many triplets the deterministic selection delivers against each "
        "study's per-view budget at image multipliers 1, 4 and 6, and how much of the "
        "shortfall the study's other views could refill, without decoding pixels"
    )
    config: Path = Field(description="Experiment configuration")
    data_root: Path = Field(description="Data root holding raw/")
    output_dir: Path = Field(description="Fresh output directory")
    workers: int = Field(default=6, gt=0, description="Worker processes")


def start(config_path: str, data_root: str) -> None:
    global DATASET
    config = load_config(Path(config_path), data_root=Path(data_root))
    index = build_series_index(config.paths.data_root, config.data, "train")
    DATASET = KneeStudyDataset(
        pl.DataFrame({STUDY: sorted(index)}), index, config.data, (), None
    )


def measure(uid: str) -> list[dict]:
    dataset = DATASET
    data = dataset.data
    study = dataset._index_study_geometry(dataset.series_index[uid])
    failures = []
    slot_sources = [
        dataset._slot_sources(study, slot, failures) for slot in range(len(study.slots))
    ]
    radius = data.reference_field_mm / 2
    pools = [
        build_pool(
            tuple(sources),
            data.series_types[slot].plane,
            dataset._study_references[uid],
            data.reference_spacing_mm,
            radius,
            set(),
            set(),
        )
        for slot, (_, sources) in enumerate(slot_sources)
    ]
    rows = []
    for multiplier in MULTIPLIERS:
        dataset.image_limit_multiplier = multiplier
        budgets = dataset._study_budgets(slot_sources)
        for slot, views in enumerate(dataset.slot_views):
            kind = data.series_types[slot]
            kind_key = (kind.plane, kind.fluid_sensitive, kind.fat_suppression)
            for order, (view_index, _) in enumerate(views):
                metadata = select_triplets(
                    pools[slot],
                    budgets[view_index],
                    data.sampling_sigma_fraction,
                    radius,
                    uid,
                    (*kind_key, order) if order else kind_key,
                    None,
                    0,
                ).metadata
                rows.append(
                    {
                        STUDY: uid,
                        "multiplier": multiplier,
                        "view": view_index,
                        "plane": kind.plane,
                        "fluid_sensitive": kind.fluid_sensitive,
                        "sources": len(slot_sources[slot][1]),
                        "pool_path": metadata["path"],
                        "candidates": len(pools[slot].candidates),
                        "base": data.views[view_index][1].slices * multiplier,
                        "requested": metadata["requested_triplets"],
                        "capacity": metadata["capacity"],
                        "delivered": metadata["triplets"],
                        "index_failures": len(failures),
                    }
                )
    return rows


def summarise(frame: pl.DataFrame) -> list[dict]:
    present = frame.filter(pl.col("sources") > 0).with_columns(
        (pl.col("requested") - pl.col("delivered")).alias("shortfall"),
        (pl.col("capacity") - pl.col("delivered")).alias("spare"),
    )
    plane = present.group_by([*KEYS, "plane"]).agg(
        pl.col("spare").sum().alias("plane_spare"),
    )
    same_plane = (
        present.join(plane, on=[*KEYS, "plane"])
        .with_columns((pl.col("plane_spare") - pl.col("spare")).alias("partner_spare"))
        .group_by(KEYS)
        .agg(
            pl.min_horizontal("shortfall", "partner_spare")
            .sum()
            .alias("same_plane_recoverable")
        )
    )
    studies = (
        present.group_by(KEYS)
        .agg(
            pl.len().alias("present_views"),
            pl.col("base").sum().alias("base"),
            pl.col("requested").sum().alias("requested"),
            pl.col("delivered").sum().alias("delivered"),
            pl.col("shortfall").sum().alias("shortfall"),
            pl.col("spare").filter(pl.col("shortfall") == 0).sum().alias("other_spare"),
            pl.col("spare").filter(pl.col("shortfall") > 0).sum().alias("own_spare"),
        )
        .join(same_plane, on=KEYS)
        .with_columns(
            pl.min_horizontal("shortfall", "other_spare").alias("recoverable"),
            pl.min_horizontal("shortfall", "same_plane_recoverable").alias(
                "same_plane_recoverable"
            ),
        )
    )
    result = []
    for multiplier in MULTIPLIERS:
        own = studies.filter(pl.col("multiplier") == multiplier)
        short = own.filter(pl.col("shortfall") > 0)
        shortfall = int(own["shortfall"].sum())
        views = (
            present.filter(pl.col("multiplier") == multiplier)
            .group_by("view", "plane", "fluid_sensitive")
            .agg(
                pl.len().alias("present_studies"),
                pl.col("requested").sum(),
                pl.col("delivered").sum(),
                pl.col("shortfall").sum(),
                (pl.col("shortfall") > 0).sum().alias("short_studies"),
                (pl.col("capacity") == 0).sum().alias("zero_capacity_studies"),
                pl.col("capacity").median().alias("median_capacity"),
                pl.col("spare").sum().alias("spare"),
            )
            .sort("view")
            .with_columns(
                (100 * pl.col("shortfall") / pl.col("requested")).alias(
                    "shortfall_pct_of_requested"
                ),
                (100 * pl.col("shortfall") / max(shortfall, 1)).alias(
                    "share_of_total_shortfall_pct"
                ),
            )
        )
        result.append(
            {
                "multiplier": multiplier,
                "studies": own.height,
                "requested": int(own["requested"].sum()),
                "delivered": int(own["delivered"].sum()),
                "mean_requested": float(own["requested"].mean()),
                "mean_delivered": float(own["delivered"].mean()),
                "studies_scaled_up": int((own["requested"] > own["base"]).sum()),
                "studies_with_shortfall": short.height,
                "studies_with_shortfall_pct": 100 * short.height / own.height,
                "shortfall": shortfall,
                "shortfall_pct_of_requested": 100
                * shortfall
                / int(own["requested"].sum()),
                "mean_shortfall_in_short_studies": float(short["shortfall"].mean())
                if short.height
                else 0.0,
                "shortfall_quantiles_in_short_studies": {
                    str(q): float(short["shortfall"].quantile(q))
                    for q in (0.5, 0.9, 0.99, 1.0)
                }
                if short.height
                else {},
                "other_view_spare_capacity": int(short["other_spare"].sum()),
                "other_view_spare_pct_of_shortfall": 100
                * int(short["other_spare"].sum())
                / max(shortfall, 1),
                "recoverable": int(short["recoverable"].sum()),
                "recoverable_pct_of_shortfall": 100
                * int(short["recoverable"].sum())
                / max(shortfall, 1),
                "same_plane_recoverable": int(short["same_plane_recoverable"].sum()),
                "same_plane_recoverable_pct_of_shortfall": 100
                * int(short["same_plane_recoverable"].sum())
                / max(shortfall, 1),
                "studies_fully_recoverable": int(
                    (short["recoverable"] == short["shortfall"]).sum()
                ),
                "delivered_below_min_budget_capacity": int(short["own_spare"].sum()),
                "views": views.to_dicts(),
            }
        )
    return result


def main() -> None:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    config = load_config(settings.config, data_root=settings.data_root)
    folds = pl.read_csv(
        config.paths.data_root / "folds" / config.validation.split / "studies.csv"
    ).select(STUDY, "fold")
    uids = sorted(folds[STUDY].to_list())
    with get_context("spawn").Pool(
        settings.workers, start, (str(settings.config), str(settings.data_root))
    ) as pool:
        rows = [row for rows in pool.map(measure, uids, chunksize=8) for row in rows]
    frame = pl.DataFrame(rows, infer_schema_length=None).join(
        folds, on=STUDY, validate="m:1"
    )
    frame.write_parquet(settings.output_dir / "study_views.parquet")
    summary = {
        "selection": "deterministic, first attempt, no decoding",
        "split": config.validation.split,
        "study_slice_total": config.data.study_slice_total,
        "all_studies": summarise(frame),
        "fold_0": summarise(frame.filter(pl.col("fold") == 0)),
    }
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
