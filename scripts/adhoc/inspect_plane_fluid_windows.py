import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset
from knee_abnormality.data.preprocessing.dicom import read_frame
from knee_abnormality.data.preprocessing.spatial import (
    detail_position,
    record_layout,
    record_transform,
)
from knee_abnormality.data.sources import SERIES, build_series_index
from knee_abnormality.data.splits import STUDY

CASE_SUFFIXES = ("21482192", "55725945", "93267826", "11059491")
COLORS = {
    84.0: "#ff5555",
    280 / 3: "#ffe45e",
    112.0: "#80ff80",
    120.0: "#55ddff",
    130.0: "#ffe45e",
    140.0: "#ff55dd",
}


class Settings(CLISettings):
    model_config = describe(
        "Draw exact native-space crop boundaries from the smart-crops archived loader "
        "on report-labelled cases and a seeded sample, including nine slices per series "
        "and per-plane/contrast middle-slice sheets"
    )

    config: Path = Field(description="Archived detail_84_93 effective configuration")
    output_dir: Path = Field(
        description="Directory for images and selected case manifest"
    )
    seed: int = Field(default=20261002, description="Random case selection seed")
    random_studies: int = Field(
        default=4, ge=0, description="Additional random training cases"
    )


def draw(axis, record, boundaries, title):
    frame = read_frame(record.path, record.descriptor)
    image = frame.pixels * frame.slope + frame.intercept
    axis.imshow(
        image,
        cmap="gray",
        vmin=np.percentile(image, 0.5),
        vmax=np.percentile(image, 99.5),
    )
    for width, offsets in boundaries:
        transform = record_transform(record, width / 224, 224, *offsets)
        first = transform @ np.array([-0.5, -0.5, 1.0])
        last = transform @ np.array([223.5, 223.5, 1.0])
        axis.add_patch(
            plt.Rectangle(
                first[:2],
                last[0] - first[0],
                last[1] - first[1],
                fill=False,
                edgecolor=COLORS[width],
                linewidth=0.9,
            )
        )
    axis.set_xlim(-0.5, image.shape[1] - 0.5)
    axis.set_ylim(image.shape[0] - 0.5, -0.5)
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def run(settings):
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(settings.config)
    reports = pl.read_csv(config.paths.data_root / "raw/train.csv")
    raw_series = pl.read_csv(config.paths.data_root / "raw/train_series.csv")
    chosen = reports.filter(
        pl.any_horizontal([pl.col(STUDY).str.ends_with(s) for s in CASE_SUFFIXES])
    )[STUDY].to_list()
    axial_nonfluid = (
        raw_series.filter(
            (pl.col("Anatomical_Plane") == "Axial") & (pl.col("Fluid_Sensitive") == 0)
        )[STUDY]
        .unique()
        .sort()
    )
    chosen += (
        axial_nonfluid.filter(~axial_nonfluid.is_in(chosen))
        .sample(min(settings.random_studies, len(axial_nonfluid)), seed=settings.seed)
        .to_list()
    )
    index = build_series_index(config.paths.data_root, config.data, "train")
    dataset = KneeStudyDataset(
        pl.DataFrame({STUDY: chosen}), index, config.data, (), None
    )
    manifest = []
    overviews = {}
    for study_uid in chosen:
        study = dataset._index_study_geometry(index[study_uid])
        excluded = dataset._excluded_series[study_uid]
        for slot, candidates in enumerate(study.slots):
            kind = config.data.series_types[slot]
            eligible = sorted(
                (c for c in candidates if c.uid not in excluded), key=lambda c: c.uid
            )
            if not eligible:
                continue
            candidate = eligible[0]
            indexed = dataset._slice_index[candidate.directory]
            records = [r for r in indexed.records if record_layout(r) is not None]
            if not records:
                continue
            geometry = dataset._series_geometry[study_uid].get(
                candidate.uid, (0.0, 0.0, 0.0, False)
            )
            geometry = (
                *geometry[:3],
                bool(
                    geometry[3]
                    and indexed.geometry.native_poses_valid
                    and indexed.geometry.reason is None
                ),
            )
            if kind.plane == "Axial":
                widths = (120.0, 130.0, 140.0)
                boundaries = [(w, (0, 0)) for w in widths]
            else:
                width = 84.0 if kind.fluid_sensitive == 0 else 280 / 3
                widths = (width, 112.0, 120.0)
                boundaries = [
                    (
                        w,
                        detail_position(
                            indexed.records,
                            kind.plane,
                            geometry,
                            config.data.spacing_mm,
                            w / 224,
                            224,
                        )[:2],
                    )
                    for w in widths[:-1]
                ]
                boundaries.append((120.0, (0, 0)))
            positions = sorted(
                {round(q * (len(records) - 1)) for q in np.linspace(0.1, 0.9, 9)}
            )
            short = study_uid[-8:]
            name = f"{short}_{kind.plane.lower()}_fs{kind.fluid_sensitive}"
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            for axis in axes.flat:
                axis.axis("off")
            for axis, position in zip(axes.flat, positions):
                draw(
                    axis,
                    records[position],
                    boundaries,
                    f"slice {position + 1}/{len(records)}",
                )
            legend = ", ".join(f"{w:.1f} mm {COLORS[w]}" for w in widths)
            fig.suptitle(
                f"{name}\n{legend}; narrow windows use production rule within 120 mm",
                fontsize=10,
            )
            fig.tight_layout()
            fig.savefig(settings.output_dir / f"{name}.png", dpi=100)
            plt.close(fig)
            overviews.setdefault((kind.plane, kind.fluid_sensitive), []).append(
                (records[len(records) // 2], boundaries, short)
            )
            manifest.append(
                {
                    STUDY: study_uid,
                    SERIES: candidate.uid,
                    "plane": kind.plane,
                    "fluid_sensitive": kind.fluid_sensitive,
                    "geometry_trusted": geometry[3],
                    "slices": len(records),
                    "review_positions": positions,
                    "boundaries": [
                        {"width_mm": w, "offset_y": o[0], "offset_x": o[1]}
                        for w, o in boundaries
                    ],
                    "sheet": f"{name}.png",
                }
            )
        dataset._slice_index.clear()
    for (plane, fluid), items in overviews.items():
        rows = (len(items) + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=(16, rows * 4), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, (record, boundaries, short) in zip(axes.flat, items):
            draw(axis, record, boundaries, short)
        fig.suptitle(
            f"{plane} fs{fluid}: middle slice; 120 cyan; detail red/yellow; 112 green; axial 130 yellow / 140 magenta"
        )
        fig.tight_layout()
        fig.savefig(
            settings.output_dir / f"overview_{plane.lower()}_fs{fluid}.png", dpi=100
        )
        plt.close(fig)
    reports.filter(pl.col(STUDY).is_in(chosen)).write_csv(
        settings.output_dir / "case_reports.csv"
    )
    payload = {
        "config": str(settings.config),
        "seed": settings.seed,
        "selected_studies": chosen,
        "selection": "four purposeful labelled/reported cases plus seeded sample of studies with axial nonfluid",
        "series": manifest,
    }
    (settings.output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "studies": len(chosen),
                "series": len(manifest),
                "directory": str(settings.output_dir),
            }
        )
    )


def main():
    run(Settings())


if __name__ == "__main__":
    main()
