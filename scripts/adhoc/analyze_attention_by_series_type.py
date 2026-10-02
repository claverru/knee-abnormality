import json
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe

STUDY = "StudyInstanceUID"
TYPES = (
    ("Axial", 0),
    ("Axial", 1),
    ("Coronal", 0),
    ("Coronal", 1),
    ("Sagittal", 0),
    ("Sagittal", 1),
)
GRID = 7
STRIDE = 32
WIDTHS = (60.0, 70.0, 84.0, 280 / 3, 100.0, 112.0, 120.0, 130.0, 140.0)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
BELOW_MM = 10.0
SAGITTAL_ANTERIOR_MM = 7.0
THRESHOLD = 0.5


class Settings(CLISettings):
    model_config = describe(
        "From a label-attention dump, tabulate per label the share of attention on "
        "each series type, the share of each type's attention inside square windows "
        "of several widths (rule-placed on coronal and sagittal slices, image-centred "
        "on axial ones) and the knee-centred extent of axial attention"
    )

    dump_dir: Path = Field(description="Directory written by dump_label_attention")
    targets: Path = Field(
        default=Path("data/folds/all_5fold_scanner_v2/targets.csv"),
        description="Fold targets used to mark positive studies",
    )
    output_dir: Path = Field(description="Output directory")


def weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> list[float]:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative = cumulative / cumulative[-1]
    return [
        float(values[order][min(np.searchsorted(cumulative, q), len(values) - 1)])
        for q in QUANTILES
    ]


def overlap(edges: np.ndarray, centre: np.ndarray, width: float, cell: float):
    low = np.maximum(edges[None], (centre - width / 2)[:, None])
    high = np.minimum(edges[None] + cell, (centre + width / 2)[:, None])
    return np.clip(high - low, 0, None) / cell


def block_centres(values: np.ndarray) -> np.ndarray:
    across = (values[:, :, -1:] - values[:, :, :1]) / (GRID - 1)
    down = (values[:, -1:, :] - values[:, :1, :]) / (GRID - 1)
    return values + 0.5 * across + 0.5 * down


def run(settings: Settings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((settings.dump_dir / "summary.json").read_text())
    config = json.loads((settings.dump_dir / "effective_config.json").read_text())
    targets = summary["targets"]
    spacing = config["data"]["spacing_mm"]
    size = config["data"]["image_size"]
    cell = STRIDE * spacing
    field = size * spacing
    anchors = (np.arange(GRID) * STRIDE - (size - 1) / 2) * spacing
    edges = (np.arange(GRID) * STRIDE - size / 2) * spacing
    tokens = pl.read_parquet(settings.dump_dir / "tokens.parquet")
    cells = GRID * GRID
    attention = (
        tokens.select([f"attention|{t}" for t in targets])
        .to_numpy()
        .astype(np.float64)
        .reshape(-1, GRID, GRID, len(targets))
    )
    medial = tokens["medial_mm"].to_numpy().reshape(-1, GRID, GRID).astype(np.float64)
    anterior = (
        tokens["anterior_mm"].to_numpy().reshape(-1, GRID, GRID).astype(np.float64)
    )
    trusted = tokens["trusted"].to_numpy().reshape(-1, cells).any(1)
    first = tokens.gather_every(cells)
    kind = first["image_type"].to_numpy()
    studies = first[STUDY].to_numpy()
    truth = (
        pl.DataFrame({STUDY: studies})
        .join(pl.read_csv(settings.targets), on=STUDY, how="left")
        .select(targets)
        .to_numpy()
        >= THRESHOLD
    )
    slices = len(kind)
    span = anchors[-1] - anchors[0]
    middle = GRID // 2
    centre_x = np.zeros(slices)
    for plane_types, axis, target in (
        ((2, 3), medial, 0.0),
        ((4, 5), anterior, SAGITTAL_ANTERIOR_MM),
    ):
        rows = np.isin(kind, plane_types) & trusted
        start, stop = axis[rows, middle, 0], axis[rows, middle, -1]
        slope = (stop - start) / span
        usable = np.abs(slope) > 0.5
        position = np.zeros(rows.sum())
        position[usable] = anchors[0] + (target - start[usable]) / slope[usable]
        centre_x[rows] = position
    centre_y = np.where(kind >= 2, BELOW_MM, 0.0)
    mass = attention.sum((1, 2))
    total = mass.sum(0)
    share_rows, window_rows, extent_rows = [], [], []
    slice_share = np.array([(kind == k).mean() for k in range(len(TYPES))])
    cohorts = {"all": np.ones_like(truth), "positive": truth}
    for cohort, member in cohorts.items():
        weights = mass * member
        for t, label in enumerate(targets):
            for k, (plane, fluid) in enumerate(TYPES):
                share = weights[kind == k, t].sum() / weights[:, t].sum()
                share_rows.append(
                    {
                        "cohort": cohort,
                        "label": label,
                        "studies": len(np.unique(studies[member[:, t]])),
                        "plane": plane,
                        "fluid_sensitive": fluid,
                        "share_of_slices": float(slice_share[k]),
                        "share_of_label_attention": float(share),
                        "per_slice_against_uniform": float(share / slice_share[k]),
                    }
                )
    for width in WIDTHS:
        limit = max(0.0, (field - width) / 2)
        cx = np.clip(centre_x, -limit, limit)
        cy = np.clip(centre_y, -limit, limit)
        inside = np.einsum(
            "srct,sr,sc->st",
            attention,
            overlap(edges, cy, width, cell),
            overlap(edges, cx, width, cell),
        )
        for cohort, member in cohorts.items():
            for t, label in enumerate(targets):
                label_total = (mass[:, t] * member[:, t]).sum()
                for k, (plane, fluid) in enumerate(TYPES):
                    rows = (kind == k) & member[:, t]
                    type_mass = mass[rows, t].sum()
                    kept = inside[rows, t].sum()
                    window_rows.append(
                        {
                            "cohort": cohort,
                            "label": label,
                            "plane": plane,
                            "fluid_sensitive": fluid,
                            "width_mm": width,
                            "placement": "image_centre" if k < 2 else "rule",
                            "share_of_type_attention_inside": float(kept / type_mass),
                            "label_attention_outside": float(
                                (type_mass - kept) / label_total
                            ),
                        }
                    )
    grid_medial = block_centres(medial).reshape(slices, cells)
    grid_anterior = block_centres(anterior).reshape(slices, cells)
    flat = attention.reshape(slices, cells, len(targets))
    for cohort, member in cohorts.items():
        for t, label in enumerate(targets):
            for k in (0, 1):
                rows = (kind == k) & trusted & member[:, t]
                weights = flat[rows, :, t].reshape(-1)
                for axis, values in (
                    ("medial_mm", grid_medial),
                    ("anterior_mm", grid_anterior),
                ):
                    quantiles = weighted_quantiles(values[rows].reshape(-1), weights)
                    extent_rows.append(
                        {
                            "cohort": cohort,
                            "label": label,
                            "plane": TYPES[k][0],
                            "fluid_sensitive": TYPES[k][1],
                            "axis": axis,
                            "slices": int(rows.sum()),
                            **{
                                f"p{round(q * 100)}": value
                                for q, value in zip(QUANTILES, quantiles, strict=True)
                            },
                        }
                    )
    for name, rows in (
        ("attention_share_by_type", share_rows),
        ("attention_window_share", window_rows),
        ("attention_axial_extent", extent_rows),
    ):
        pl.DataFrame(rows).write_csv(
            settings.output_dir / f"{name}.csv", float_precision=4
        )
    print(
        json.dumps(
            {
                "studies": len(np.unique(studies)),
                "slices": slices,
                "field_mm": field,
                "cell_mm": cell,
                "attention_total_per_label": dict(zip(targets, total.tolist())),
            },
            indent=2,
        )
    )


def main() -> None:
    run(Settings())


if __name__ == "__main__":
    main()
