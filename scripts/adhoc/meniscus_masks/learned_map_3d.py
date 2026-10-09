import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
import torch
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import move_batch
from knee_abnormality.training.anatomy import ANATOMY_CLASSES, MENISCUS_CLASSES

RUN = Path("outputs/training") / sys.argv[1]
OUT = Path("outputs/analysis") / sys.argv[2]
FRAME_KNEES = ("088749", "933066", "259018", "909133")
MASK_KNEES = 2
VOXEL_MM = 3.0
SMOOTH_MM = 5.0
DENSITY = 0.08
PAD = 4
IMAGE_MULTIPLIER = 4
MENISCUS_SHARE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
BONE_COLOURS = {
    "femur": "#d9534f",
    "tibia": "#337ab7",
    "kneecap": "#e0a800",
    "fibula": "#3c9d4b",
    "joint": "#8e44ad",
    "kneecap_joint": "#e67e22",
}
MENISCUS_COLOURS = {"medial_meniscus": "#00bcd4", "lateral_meniscus": "#f5e000"}
MASK_COLOURS = {"medial_meniscus": "#006978", "lateral_meniscus": "#8a7d00"}
VIEW_COLOURS = {"coronal": "#1b9e77", "sagittal": "#c2185b"}


def knee(model, config, dataset, item, masks):
    batch = with_images(
        move_batch(collate_studies([dataset[item]]), torch.device("cpu"))
    )
    with torch.no_grad():
        encoded = model.encode_batch(batch)
    study = batch["study_uid"][0]
    position = encoded["fitted_position"].float().numpy()
    valid = (encoded["visual_valid"] & encoded["routable"][:, None]).numpy()
    patient = encoded["patient_position"].float().numpy()
    design = np.concatenate((patient[valid], np.ones((int(valid.sum()), 1))), 1)
    solution, *_ = np.linalg.lstsq(design, position[valid], rcond=None)
    residual = float(np.abs(design @ solution - position[valid]).max())
    shares = np.concatenate(
        (
            encoded["anatomy_shares"].float().numpy(),
            encoded["meniscus_scores"].float().softmax(-1).numpy(),
        ),
        -1,
    )
    points = position[valid]
    low = np.floor(points.min(0) / VOXEL_MM).astype(int) - PAD
    cells = np.rint(points / VOXEL_MM).astype(int) - low
    shape = tuple(cells.max(0) + PAD + 1)
    weight = np.zeros(shape)
    np.add.at(weight, tuple(cells.T), 1.0)
    total = np.zeros((*shape, shares.shape[-1]))
    np.add.at(total, tuple(cells.T), shares[valid])
    sigma = SMOOTH_MM / VOXEL_MM
    weight = gaussian_filter(weight, sigma)
    total = gaussian_filter(total, (sigma, sigma, sigma, 0))
    dense = weight >= DENSITY
    mean = total[dense] / weight[dense, None]
    cells = np.argwhere(dense) + low
    bones, menisci = mean[:, : len(ANATOMY_CLASSES)], mean[:, len(ANATOMY_CLASSES) :]
    main = bones.argmax(-1)
    clouds = {
        name: cells[main == index] * VOXEL_MM
        for index, name in enumerate(ANATOMY_CLASSES)
        if index
    }
    predicted = {
        name: (
            cells[menisci[:, index] >= MENISCUS_SHARE] * VOXEL_MM,
            menisci[menisci[:, index] >= MENISCUS_SHARE, index],
        )
        for index, name in enumerate(MENISCUS_CLASSES)
        if index
    }
    truth = {}
    path = masks / "volumes" / f"{study}.npz"
    if path.exists():
        stored = np.load(path)
        for index, name in enumerate(MENISCUS_CLASSES):
            if index:
                voxels = np.argwhere(stored["labels"] == index)
                voxels = voxels[(voxels % 2 == 0).all(-1)]
                world = stored["origin"] + float(stored["spacing"]) * voxels
                truth[name] = world @ solution[:3] + solution[3]
    shift = encoded["study_frame_shift"][0].float().numpy()
    rotation = encoded["study_frame_rotation"][0].float().numpy()
    sizes = encoded["study_frame_sizes"][0].float().numpy()
    size = config.data.image_size
    kinds = config.data.series_types
    crops = {}
    types = encoded["image_type"].numpy()
    for kind in sorted(set(types.tolist())):
        plane = kinds[kind].plane.lower()
        if plane not in VIEW_COLOURS:
            continue
        members = np.flatnonzero(types == kind)
        image = members[len(members) // 2]
        side = round(position.shape[1] ** 0.5)
        origin = position[image, 0]
        across = (position[image, 1] - origin) / 32.0
        down = (position[image, side] - origin) / 32.0
        corners = [(0, 0), (size - 1, 0), (size - 1, size - 1), (0, size - 1), (0, 0)]
        contrast = "fluid" if kinds[kind].fluid_sensitive else "T1/PD"
        crops[f"{plane} {contrast}"] = np.array(
            [origin + x * across + y * down for x, y in corners]
        )
    tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1))))
    return {
        "study": study,
        "clouds": clouds,
        "predicted": predicted,
        "truth": truth,
        "shift": shift,
        "rotation": rotation,
        "sizes": sizes,
        "crops": crops,
        "tokens": int(valid.sum()),
        "slices": len(valid),
        "tilt": tilt,
        "turn": float(encoded["crop_angle"].abs().max()),
        "residual": residual,
        "largest_meniscus_share": menisci[:, 1:].max(0).round(3).tolist(),
    }


def quad(centre, first, second, half):
    return np.array(
        [
            centre + half * (a * first + b * second)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))
        ]
    )


def traces(result, legend):
    items = []
    for name, points in result["clouds"].items():
        items.append(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={"color": BONE_COLOURS[name], "opacity": 0.55, "size": 3},
                name=f"model's map: {name}",
                legendgroup=name,
                showlegend=bool(legend),
            )
        )
    for name, (points, share) in result["predicted"].items():
        items.append(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={
                    "color": MENISCUS_COLOURS[name],
                    "opacity": 0.95,
                    "size": 4 + 10 * share,
                    "line": {"color": "black", "width": 1},
                },
                name=f"model's map: {name} (share at least {MENISCUS_SHARE})",
                legendgroup=name,
                showlegend=bool(legend),
            )
        )
    for name, points in result["truth"].items():
        items.append(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={"color": MASK_COLOURS[name], "opacity": 0.5, "size": 1.5},
                name=f"segmenter's mask: {name}",
                legendgroup=f"mask {name}",
                showlegend=legend == "mask",
            )
        )
    shift, rotation, sizes = result["shift"], result["rotation"], result["sizes"]
    plane = quad(shift, rotation[0], rotation[1], 45.0)
    items.append(
        go.Mesh3d(
            x=plane[:4, 0],
            y=plane[:4, 1],
            z=plane[:4, 2],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#111111",
            opacity=0.3,
            name="joint plane",
            legendgroup="plane",
            showlegend=bool(legend),
        )
    )
    for index, height in enumerate(sizes[:2]):
        cut = quad(shift + height * rotation[2], rotation[0], rotation[1], 45.0)
        items.append(
            go.Scatter3d(
                x=cut[:, 0],
                y=cut[:, 1],
                z=cut[:, 2],
                mode="lines",
                line={"color": "#666666", "dash": "dash", "width": 3},
                name="section cuts above and below the joint",
                legendgroup="cuts",
                showlegend=bool(legend) and index == 0,
            )
        )
    for index, label in enumerate(("medial", "front", "up")):
        end = shift + 50.0 * rotation[index]
        items.append(
            go.Scatter3d(
                x=[shift[0], end[0]],
                y=[shift[1], end[1]],
                z=[shift[2], end[2]],
                mode="lines+text",
                text=["", label],
                textposition="top center",
                line={"color": "black", "width": 7},
                name="frame axes",
                legendgroup="axes",
                showlegend=bool(legend) and index == 0,
            )
        )
    items.append(
        go.Scatter3d(
            x=[shift[0]],
            y=[shift[1]],
            z=[shift[2]],
            mode="markers",
            marker={"color": "black", "size": 7},
            name="joint centre",
            legendgroup="centre",
            showlegend=bool(legend),
        )
    )
    for name, corners in result["crops"].items():
        plane, contrast = name.split(" ")
        items.append(
            go.Scatter3d(
                x=corners[:, 0],
                y=corners[:, 1],
                z=corners[:, 2],
                mode="lines",
                line={
                    "color": VIEW_COLOURS[plane],
                    "dash": "solid" if contrast == "fluid" else "dot",
                    "width": 4,
                },
                name=f"posed crop, middle slice: {name}",
                legendgroup=name,
                showlegend=bool(legend),
            )
        )
    return items


def preview(results, path):
    figure = plt.figure(figsize=(16, 7 * len(results)))
    for row, result in enumerate(results):
        for column, (elevation, azimuth, view) in enumerate(
            ((8, -90, "front"), (8, 0, "medial side"), (89, -90, "above"))
        ):
            panel = figure.add_subplot(
                len(results), 3, 3 * row + column + 1, projection="3d"
            )
            for name, points in result["clouds"].items():
                panel.scatter(*points.T, s=3, color=BONE_COLOURS[name], alpha=0.45)
            for name, points in result["truth"].items():
                panel.scatter(*points.T, s=1, color=MASK_COLOURS[name], alpha=0.4)
            for name, (points, share) in result["predicted"].items():
                if len(points):
                    panel.scatter(
                        *points.T,
                        s=8 + 40 * share,
                        color=MENISCUS_COLOURS[name],
                        edgecolors="black",
                        linewidths=0.5,
                        alpha=1.0,
                    )
            panel.view_init(elevation, azimuth)
            panel.set_box_aspect((1, 1, 1))
            panel.set_xlim(-70, 70)
            panel.set_ylim(-70, 70)
            panel.set_zlim(-80, 60)
            panel.set_title(f"…{result['study'][-6:]} seen from {view}", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=60)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    config = load_config(RUN / "effective_config.json")
    targets, studies, split = prediction_studies(config, "validation", 1)
    masks = config.paths.meniscus_targets
    accepted = {
        row["study"]
        for row in json.loads((masks / "knees.json").read_text())
        if row["accepted"]
    }
    uids = studies[STUDY].to_list()
    chosen = sorted(uid for uid in uids if uid in accepted)[:MASK_KNEES] + [
        uid for suffix in FRAME_KNEES for uid in uids if uid.endswith(suffix)
    ]
    frame = pl.concat([studies.filter(pl.col(STUDY) == uid) for uid in chosen])
    index = build_series_index(config.paths.data_root, config.data, split, chosen)
    dataset = KneeStudyDataset(
        frame, index, config.data, targets, None, IMAGE_MULTIPLIER
    )
    checkpoint = max(
        (RUN / "fold_1").glob("best-epoch=*.ckpt"),
        key=lambda path: path.stat().st_mtime,
    )
    model = load_checkpoint_model(
        config, targets, checkpoint, torch.device("cpu"), expected_fold=1
    ).eval()
    results = [knee(model, config, dataset, item, masks) for item in range(len(chosen))]
    columns = 2
    rows = -(-len(results) // columns)
    figure = make_subplots(
        rows=rows,
        cols=columns,
        specs=[[{"type": "scene"}] * columns for _ in range(rows)],
        subplot_titles=[
            f"…{result['study'][-6:]}"
            f"{' (DESS knee, segmenter mask shown)' if result['truth'] else ''}: "
            f"{result['tokens']} tokens from {result['slices']} slices; plane tilted "
            f"{result['tilt']:.1f}° from the scanner's axis; largest crop turn "
            f"{result['turn']:.1f}°"
            for result in results
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.04,
    )
    for number, result in enumerate(results):
        legend = "mask" if number == 0 else ("frame" if number == MASK_KNEES else "")
        for trace in traces(result, legend):
            figure.add_trace(trace, row=number // columns + 1, col=number % columns + 1)
    figure.update_scenes(
        aspectmode="data",
        xaxis_title="medial (mm)",
        yaxis_title="front (mm)",
        zaxis_title="up (mm)",
        camera={"eye": {"x": 1.5, "y": -1.5, "z": 0.5}},
    )
    figure.update_layout(
        title=(
            "What the model believes about each knee: its bone map and its meniscus "
            f"map fused over all views, and the frame it fits ({RUN.name}, "
            f"{checkpoint.name[:13]}). The segmenter's meniscus masks are drawn only "
            "on the two DESS knees, for comparison"
        ),
        height=750 * rows,
        width=1750,
        legend={"itemsizing": "constant"},
    )
    figure.write_html(OUT / "learned-map-3d.html", include_plotlyjs=True)
    preview(results, OUT / "learned-map-3d-preview.png")
    for result in results:
        print(
            f"…{result['study'][-6:]}: tokens {result['tokens']}, frame fit residual "
            f"{result['residual']:.3f} mm, largest fused meniscus share (medial, "
            f"lateral) {result['largest_meniscus_share']}, voxels drawn "
            f"{[len(points) for points, _ in result['predicted'].values()]}, mask "
            f"points {[len(points) for points in result['truth'].values()]}"
        )
    print("saved", OUT / "learned-map-3d.html")


if __name__ == "__main__":
    main()
