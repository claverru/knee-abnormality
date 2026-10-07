import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import DataConfig, load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import move_batch


class Settings(CLISettings):
    model_config = describe(
        "Render, for validation studies of a side-routed run, each series type's "
        "middle slice as the model sees it with the fitted and learned midline in the "
        "knee's frame, the front boundary, the routing weights, one target's position "
        "prior and the crop's "
        "outline in the 120 mm frame"
    )
    run: Path = Field(description="Training run directory")
    output_dir: Path = Field(description="Fresh output directory for the images")
    scan: int = Field(default=60, gt=0, description="Validation studies to scan")
    studies: int = Field(default=8, gt=0, description="Studies to render")
    prior_target: str = Field(
        default="ACL", description="Target whose position prior is drawn"
    )


def wide_data(data: DataConfig) -> DataConfig:
    values = data.model_dump(mode="json")
    for kind in values["series_types"]:
        for view in kind["views"]:
            view["field_mm"] = values["reference_field_mm"]
            view["horizontal"] = {"anchor": "image", "offset_mm": 0.0}
            view["vertical"] = {"anchor": "image", "offset_mm": 0.0}
    return DataConfig.model_validate(values)


def prepared(dataset: KneeStudyDataset, row: int, device: torch.device) -> dict:
    return with_images(move_batch(collate_studies([dataset[row]]), device))


def slices(batch: dict) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    mask = batch["slice_mask"].flatten()
    images = batch["images"].flatten(0, 1)[mask][:, 1].float().cpu()
    maps = batch["patient_maps"].flatten(0, 1)[mask][:, 1].float().cpu()
    paths = [triplet[1] for series in batch["selected_paths"] for triplet in series]
    return images, maps, paths


def study_maps(model, batch: dict) -> dict:
    encoded = model.encode_batch(batch)
    valid = encoded["visual_valid"]
    routing = model.routing_weights(encoded)
    anterior = encoded["anterior_axis"]
    prior = model.position_priors(encoded["position"], encoded["routable"])
    left = encoded["left_probability"]
    axis = encoded["study_axis"][0]
    return {
        "knee_x": encoded["knee_x"].cpu(),
        "anterior": anterior.cpu(),
        "medial": routing[0].cpu(),
        "front": routing[2].cpu(),
        "prior": prior.cpu(),
        "shift": torch.stack((torch.zeros(()), torch.zeros(()))).cpu(),
        "rotation": float(torch.rad2deg(torch.atan2(axis[1], axis[0]))),
        "left": left.cpu(),
        "type": encoded["image_type"].cpu(),
        "valid": valid.cpu(),
    }


def side_state(batch: dict) -> str:
    if bool(batch["knee_side_certain"][0]):
        return "certain"
    if bool(batch["knee_side_coordinates"][0]):
        return "coordinates"
    return "none"


def draw(axis, image, values, grid, extent, title, cmap, vmax=1.0):
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    if values is not None:
        axis.imshow(
            values.reshape(grid, grid),
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            alpha=0.45,
            extent=extent,
            interpolation="nearest",
        )
    axis.set_title(title, fontsize=7)
    axis.set_xticks([])
    axis.set_yticks([])


def main() -> None:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    config = load_config(settings.run / "effective_config.json")
    fold = config.validation.folds[0]
    targets, studies, split = prediction_studies(config, "validation", fold)
    studies = studies.sort(STUDY).head(settings.scan)
    device = torch.device("cuda")
    selected = json.loads(
        (settings.run / f"fold_{fold}" / "selected_checkpoint.json").read_text()
    )
    model = load_checkpoint_model(
        config, targets, Path(selected["path"]), device, expected_fold=fold
    )
    index = build_series_index(
        config.paths.data_root, config.data, split, studies[STUDY].to_list()
    )
    dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
    wide = KneeStudyDataset(studies, index, wide_data(config.data), targets, None, 1)
    names = [
        f"{kind.plane} {'fluid' if kind.fluid_sensitive else 'T1/PD'}"
        for kind in config.data.series_types
    ]
    origin_y, origin_x, stride_y, stride_x = model.feature_mapping
    prior_target = targets.index(settings.prior_target)
    records = []
    with torch.inference_mode():
        for row in range(studies.height):
            batch = prepared(dataset, row, device)
            maps = study_maps(model, batch)
            records.append(
                {
                    "row": row,
                    STUDY: batch["study_uid"][0],
                    "state": side_state(batch),
                    "left_probability": float(maps["left"].mean()),
                    "shift_midline_mm": float(maps["shift"][0]),
                    "rotation_deg": maps["rotation"],
                    "shift_front_mm": float(maps["shift"][1]),
                }
            )
        table = pl.DataFrame(records)
        table.write_csv(settings.output_dir / "studies.csv")
        ranked = table.sort(pl.col("shift_midline_mm").abs(), descending=True)
        chosen = (
            ranked.head(2)["row"].to_list()
            + ranked.tail(2)["row"].to_list()
            + table.filter(pl.col("state") == "none").head(2)["row"].to_list()
            + table.filter(pl.col("state") == "coordinates").head(2)["row"].to_list()
        )
        chosen = list(dict.fromkeys(chosen))[: settings.studies]
        for row in chosen:
            batch = prepared(dataset, row, device)
            maps = study_maps(model, batch)
            images, crop_maps, paths = slices(batch)
            wide_images, wide_maps, wide_paths = slices(prepared(wide, row, device))
            wide_index = {path: number for number, path in enumerate(wide_paths)}
            present = sorted(set(maps["type"].tolist()))
            figure, axes = plt.subplots(
                len(present), 5, figsize=(11.0, 2.25 * len(present)), squeeze=False
            )
            grid = int(maps["knee_x"].shape[1] ** 0.5)
            size = images.shape[-1]
            xs = origin_x + stride_x * np.arange(grid)
            ys = origin_y + stride_y * np.arange(grid)
            extent = (
                xs[0] - stride_x / 2,
                xs[-1] + stride_x / 2,
                ys[-1] + stride_y / 2,
                ys[0] - stride_y / 2,
            )
            shift = maps["shift"]
            for line, kind in enumerate(present):
                members = (maps["type"] == kind).nonzero().flatten().tolist()
                item = members[len(members) // 2]
                image = images[item].numpy()
                knee_x = maps["knee_x"][item].reshape(grid, grid).numpy()
                anterior = maps["anterior"][item].reshape(grid, grid).numpy()
                draw(axes[line][0], image, None, grid, extent, names[kind], None)
                for values, level, colour, style in (
                    (knee_x, 0.0, "white", "dashed"),
                    (knee_x, float(shift[0]), "yellow", "solid"),
                    (anterior, float(shift[1]), "cyan", "solid"),
                ):
                    if values.min() < level < values.max():
                        axes[line][0].contour(
                            xs,
                            ys,
                            values,
                            [level],
                            colors=colour,
                            linestyles=style,
                            linewidths=1.0,
                        )
                axes[line][0].scatter(*np.meshgrid(xs, ys), s=2, c="lime")
                axes[line][0].set_xlim(0, size - 1)
                axes[line][0].set_ylim(size - 1, 0)
                draw(
                    axes[line][1],
                    image,
                    maps["medial"][item].numpy(),
                    grid,
                    extent,
                    "medial weight",
                    "coolwarm",
                )
                draw(
                    axes[line][2],
                    image,
                    maps["front"][item].numpy(),
                    grid,
                    extent,
                    "front weight",
                    "viridis",
                )
                values = maps["prior"][prior_target, item]
                axes[line][3].imshow(image, cmap="gray", vmin=0, vmax=255)
                axes[line][3].imshow(
                    values.reshape(grid, grid).numpy(),
                    cmap="coolwarm",
                    vmin=-1,
                    vmax=1,
                    alpha=0.45,
                    extent=extent,
                    interpolation="nearest",
                )
                axes[line][3].set_title(
                    f"position prior, {settings.prior_target}", fontsize=7
                )
                axes[line][3].set_xticks([])
                axes[line][3].set_yticks([])
                other = wide_index.get(paths[item])
                if other is None:
                    axes[line][4].axis("off")
                    continue
                draw(
                    axes[line][4],
                    wide_images[other].numpy(),
                    None,
                    grid,
                    extent,
                    "120 mm frame, crop outlined",
                    None,
                )
                corners = torch.tensor(
                    [
                        [0, 0, 1],
                        [size - 1, 0, 1],
                        [size - 1, size - 1, 1],
                        [0, size - 1, 1],
                        [0, 0, 1],
                    ],
                    dtype=torch.float32,
                )
                points = corners @ crop_maps[item].T
                basis = wide_maps[other]
                outline = (points - basis[:, 2]) @ torch.linalg.pinv(basis[:, :2]).T
                axes[line][4].plot(outline[:, 0], outline[:, 1], color="orange", lw=1)
                axes[line][4].set_xlim(0, size - 1)
                axes[line][4].set_ylim(size - 1, 0)
            record = table.row(by_predicate=pl.col("row") == row, named=True)
            figure.suptitle(
                f"{record[STUDY][-12:]}  side {record['state']}  "
                f"p(left) {record['left_probability']:.3f}  "
                f"rotation {record['rotation_deg']:+.1f} deg  "
                f"midline shift {record['shift_midline_mm']:+.1f} mm (yellow; fitted "
                f"white dashed)  front boundary {record['shift_front_mm']:+.1f} mm (cyan)",
                fontsize=9,
            )
            figure.tight_layout()
            figure.savefig(settings.output_dir / f"study-{row:03d}.png", dpi=100)
            plt.close(figure)
    print(table.select(pl.exclude(STUDY)).describe())
    print(chosen)


if __name__ == "__main__":
    main()
