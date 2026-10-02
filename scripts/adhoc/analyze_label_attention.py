import json
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD, load_split
from knee_abnormality.provenance import copy_sources

PLANES = ("Axial", "Coronal", "Sagittal")
TYPES = (
    "Axial T1/PD",
    "Axial fluid",
    "Coronal T1/PD",
    "Coronal fluid",
    "Sagittal T1/PD",
    "Sagittal fluid",
)
DETAIL = (
    "ACL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
)
GRID = 7
STRIDE = 32
QUANTILES = (0.05, 0.1, 0.5, 0.9, 0.95)


class Settings(CLISettings):
    model_config = describe(
        "Summarise a label-attention dump: sharpness, use of each series type, the "
        "share of attention inside fixed and per-study windows, knee-centred extents "
        "and the offset between the image centre and the fitted knee centre"
    )

    dump_dir: Path = Field(description="Directory written by dump_label_attention")


def windows(maps: np.ndarray, size: int) -> np.ndarray:
    steps = GRID - size + 1
    out = np.zeros((maps.shape[0], steps, steps, maps.shape[-1]), dtype=np.float64)
    for row in range(steps):
        for column in range(steps):
            out[:, row, column] = maps[:, row : row + size, column : column + size].sum(
                (1, 2)
            )
    return out


def entropy(values: np.ndarray, axis: int) -> np.ndarray:
    safe = np.where(values > 0, values, 1.0)
    return -(values * np.log(safe)).sum(axis)


def weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> list[float]:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative = cumulative / cumulative[-1]
    return [
        float(values[order][min(np.searchsorted(cumulative, q), len(values) - 1)])
        for q in QUANTILES
    ]


def run(settings: Settings) -> dict:
    tokens = pl.read_parquet(settings.dump_dir / "tokens.parquet")
    summary = json.loads((settings.dump_dir / "summary.json").read_text())
    targets = summary["targets"]
    config = load_config(settings.dump_dir / "effective_config.json")
    split = load_split(config.raw_dir, config.split_dir)
    spacing = config.data.spacing_mm
    centre = (config.data.image_size - 1) / 2
    cells = GRID * GRID
    if (
        len(tokens) % cells
        or (tokens["cell"].to_numpy().reshape(-1, cells) != np.arange(cells)).any()
    ):
        raise ValueError("Tokens are not complete 7x7 slices in order")
    attention = (
        tokens.select([f"attention|{t}" for t in targets])
        .to_numpy()
        .astype(np.float64)
        .reshape(-1, GRID, GRID, len(targets))
    )
    medial = tokens["medial_mm"].to_numpy().reshape(-1, GRID, GRID)
    anterior = tokens["anterior_mm"].to_numpy().reshape(-1, GRID, GRID)
    trusted = tokens["trusted"].to_numpy().reshape(-1, cells).all(1)
    first = tokens.gather_every(cells)
    plane = first["plane"].to_numpy()
    kind = first["image_type"].to_numpy()
    studies, study_index = np.unique(first[STUDY].to_numpy(), return_inverse=True)
    count = len(studies)
    truth_frame = (
        pl.DataFrame({STUDY: studies})
        .join(split.evaluation, on=STUDY, how="left")
        .select(targets)
    )
    if truth_frame.null_count().to_numpy().any():
        raise ValueError("Evaluation targets do not cover the dumped studies")
    truth = truth_frame.to_numpy() >= TARGET_THRESHOLD
    detail = [targets.index(name) for name in DETAIL]
    wide = [j for j in range(len(targets)) if j not in detail]

    flat = attention.reshape(len(attention), cells, len(targets))
    slice_mass = flat.sum(1)
    share = flat / np.where(slice_mass[:, None] > 0, slice_mass[:, None], 1.0)
    slice_entropy = entropy(share, 1) / np.log(cells)
    anchors = (np.arange(GRID) * STRIDE - centre) * spacing
    top_entropy = np.zeros((count, len(targets)))
    slices80 = np.zeros((count, len(targets)))
    planar = sorted(set(plane.tolist()) - {0})
    blocks = {size: windows(attention, size) for size in (3, 4)}
    plane_blocks = {
        size: np.zeros(
            (count, len(PLANES), GRID - size + 1, GRID - size + 1, len(targets))
        )
        for size in blocks
    }
    per_slice = np.zeros((count, len(targets)))
    centroid = np.full((count, len(PLANES), 2, len(targets)), np.nan)
    for study in range(count):
        rows = np.flatnonzero(study_index == study)
        mass = slice_mass[rows]
        top = rows[mass.argmax(0)]
        top_entropy[study] = slice_entropy[top, np.arange(len(targets))]
        ordered = np.sort(mass, 0)[::-1].cumsum(0)
        slices80[study] = (ordered < 0.8).sum(0) + 1
        per_slice[study] = blocks[3][rows].max((1, 2)).sum(0)
        for p in range(len(PLANES)):
            chosen = rows[plane[rows] == p]
            if not len(chosen):
                continue
            for size in blocks:
                plane_blocks[size][study, p] = blocks[size][chosen].sum(0)
            maps = attention[chosen].sum(0)
            total = maps.sum((0, 1))
            with np.errstate(invalid="ignore", divide="ignore"):
                centroid[study, p, 0] = (maps.sum(1) * anchors[:, None]).sum(0) / total
                centroid[study, p, 1] = (maps.sum(0) * anchors[:, None]).sum(0) / total

    result = {
        "studies": count,
        "slices_per_study_by_type": {
            name: float((kind == k).sum() / count) for k, name in enumerate(TYPES)
        },
        "labels": {},
    }
    type_share = (
        np.stack([slice_mass[kind == k].sum(0) for k in range(len(TYPES))], 1)
        / slice_mass.sum(0)[:, None]
    )
    slice_share = np.array([(kind == k).mean() for k in range(len(TYPES))])
    result["per_slice_use_against_uniform"] = {
        group: dict(zip(TYPES, (type_share[members].mean(0) / slice_share).tolist()))
        for group, members in (("detail", detail), ("wide", wide))
    }
    for t, target in enumerate(targets):
        positive = truth[:, t]
        entry = {
            "positives": int(positive.sum()),
            "top_slice_entropy_median": float(np.median(top_entropy[:, t])),
            "slices_for_80_median": float(np.median(slices80[:, t])),
            "share_by_type": dict(zip(TYPES, type_share[t].tolist())),
            "share_on_coronal_and_sagittal": float(type_share[t, 2:].sum()),
            "per_slice_60_all_planes": float(per_slice[:, t].mean()),
            "per_plane_60_all_planes": float(
                plane_blocks[3][..., t].max((2, 3)).sum(1).mean()
            ),
        }
        for size, name in ((3, "60"), (4, "80")):
            values = plane_blocks[size][:, planar][..., t]
            fixed = sum(values[:, i].mean(0).max() for i in range(len(planar)))
            entry[f"fixed_{name}_coronal_sagittal"] = float(fixed)
            entry[f"per_plane_{name}_coronal_sagittal"] = float(
                values.max((2, 3)).sum(1).mean()
            )
        weights = flat[..., t]
        coronal = (plane == 1) & trusted
        sagittal = (plane == 2) & trusted
        entry["coronal_medial_mm_quantiles"] = weighted_quantiles(
            medial[coronal].reshape(-1), weights[coronal].reshape(-1)
        )
        entry["sagittal_anterior_mm_quantiles"] = weighted_quantiles(
            anterior[sagittal].reshape(-1), weights[sagittal].reshape(-1)
        )
        entry["coronal_medial_mm_mean"] = float(
            (medial[coronal].reshape(-1) * weights[coronal].reshape(-1)).sum()
            / weights[coronal].sum()
        )
        shifts = {}
        for p, name in enumerate(PLANES):
            points = centroid[:, p, :, t]
            present = np.isfinite(points).all(1)
            shifts[name] = (
                (
                    points[present & positive].mean(0)
                    - points[present & ~positive].mean(0)
                )
                .round(2)
                .tolist()
            )
        entry["positive_minus_negative_centroid_mm"] = shifts
        result["labels"][target] = entry

    position = (centre / STRIDE) - GRID // 2
    middle = GRID // 2
    offsets = {}
    for p, name, axis in (
        (1, "coronal_medial", medial),
        (2, "sagittal_anterior", anterior),
    ):
        rows = (plane == p) & trusted
        along = axis[rows][:, middle]
        value = along[:, middle] + position * (along[:, middle + 1] - along[:, middle])
        offsets[name] = {
            "slices": int(rows.sum()),
            "median_mm": float(np.median(value)),
            "sd_mm": float(value.std()),
            "absolute_quantiles_50_90_95_mm": np.quantile(
                np.abs(value), [0.5, 0.9, 0.95]
            ).tolist(),
        }
    result["image_centre_from_knee_centre"] = offsets
    result["trusted_slice_share"] = {
        name: float(trusted[plane == p].mean()) for p, name in enumerate(PLANES)
    }
    source = Path(__file__).resolve()
    result["source_snapshot"] = copy_sources(
        [source], settings.dump_dir / "analysis_source"
    )
    (settings.dump_dir / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    result = run(Settings())
    print(
        "| Label | Top-slice entropy | Slices for 80% | Coronal+sagittal share "
        "| 60 mm fixed | 60 mm per plane | 80 mm fixed | 80 mm per plane |"
    )
    print("|" + "---|" * 8)
    for target, e in result["labels"].items():
        print(
            f"| {target} | {e['top_slice_entropy_median']:.2f} "
            f"| {e['slices_for_80_median']:.0f} "
            f"| {e['share_on_coronal_and_sagittal']:.2f} "
            f"| {e['fixed_60_coronal_sagittal']:.2f} "
            f"| {e['per_plane_60_coronal_sagittal']:.2f} "
            f"| {e['fixed_80_coronal_sagittal']:.2f} "
            f"| {e['per_plane_80_coronal_sagittal']:.2f} |"
        )
    print(json.dumps(result["per_slice_use_against_uniform"], indent=2))
    print(json.dumps(result["image_centre_from_knee_centre"], indent=2))


if __name__ == "__main__":
    main()
