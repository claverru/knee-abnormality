import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ANATOMY_CLASSES = (
    "other",
    "femur",
    "tibia",
    "kneecap",
    "fibula",
    "joint",
    "kneecap_joint",
)
SOFT_TISSUE_CLASSES = (
    "neither",
    "medial_meniscus",
    "lateral_meniscus",
    "medial_tibial_cartilage",
    "lateral_tibial_cartilage",
    "femoral_cartilage",
    "patellar_cartilage",
)
OUTSIDE = 255
ANATOMY_POINTS = 4
MENISCUS_POINTS = 4
MINIMUM_COVERAGE = 0.5


class AnatomyTargets(nn.Module):
    def __init__(self, root: Path, classes: int, points: int) -> None:
        super().__init__()
        self.classes = classes
        paths = sorted((root / "volumes").glob("*.npz"))
        if not paths:
            raise ValueError(f"No anatomy volumes under {root}")
        self.rows = {path.stem: row for row, path in enumerate(paths)}
        volumes, shapes, origins, spacings = [], [], [], []
        for path in paths:
            stored = np.load(path)
            volumes.append(torch.from_numpy(stored["labels"]).flatten())
            shapes.append(stored["labels"].shape)
            origins.append(stored["origin"])
            spacings.append(float(stored["spacing"]))
        sizes = torch.tensor([len(volume) for volume in volumes])
        self.register_buffer("labels", torch.cat(volumes), persistent=False)
        self.register_buffer("starts", torch.cumsum(sizes, 0) - sizes, persistent=False)
        self.register_buffer("shapes", torch.tensor(shapes), persistent=False)
        self.register_buffer(
            "origins", torch.tensor(np.stack(origins)).float(), persistent=False
        )
        self.register_buffer("spacings", torch.tensor(spacings), persistent=False)
        knees = {
            knee["study"]: knee
            for knee in json.loads((root / "knees.json").read_text())
            if "plateau_normal_x" in knee
        }
        planes = torch.zeros((len(paths), 9))
        for uid, row in self.rows.items():
            if uid in knees:
                knee = knees[uid]
                planes[row, :6] = torch.tensor(
                    [
                        knee[f"plateau_{part}_{axis}"]
                        for part in ("centre", "normal")
                        for axis in "xyz"
                    ]
                )
                if "femur_axis_x" in knee:
                    shaft = torch.tensor([knee[f"femur_axis_{axis}"] for axis in "xyz"])
                    planes[row, 6:] = shaft if shaft[2] > 0 else -shaft
        self.register_buffer("planes", planes, persistent=False)
        offsets = (torch.arange(points) + 0.5) / points - 0.5
        self.register_buffer(
            "grid",
            torch.stack(torch.meshgrid(offsets, offsets, indexing="ij"), -1).flatten(
                0, 1
            ),
            persistent=False,
        )

    def token_rows(
        self,
        study_uids: list[str],
        token_study: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(
            [self.rows.get(uid, -1) for uid in study_uids], device=token_study.device
        )[token_study]

    def token_shares(
        self,
        row: torch.Tensor,
        patient_position: torch.Tensor,
        token_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        known_study = row >= 0
        row = row.clamp_min(0)
        steps = token_steps.float()
        points = (
            patient_position.float()[:, :, None]
            + self.grid[None, None, :, :1] * steps[:, None, None, 0]
            + self.grid[None, None, :, 1:] * steps[:, None, None, 1]
        )
        shape = self.shapes[row][:, None, None]
        index = (
            (
                (points - self.origins[row][:, None, None])
                / self.spacings[row][:, None, None, None]
            )
            .nan_to_num(-1.0)
            .round()
            .long()
        )
        inside = (
            known_study[:, None, None] & (index >= 0).all(-1) & (index < shape).all(-1)
        )
        index = index.clamp_min(0).minimum(shape - 1)
        flat = (
            self.starts[row][:, None, None]
            + (index[..., 0] * shape[..., 1] + index[..., 1]) * shape[..., 2]
            + index[..., 2]
        )
        values = torch.where(inside, self.labels[flat].long(), OUTSIDE)
        known = values != OUTSIDE
        counts = torch.nn.functional.one_hot(
            values.clamp_max(self.classes), self.classes + 1
        )[..., :-1].sum(-2)
        total = known.sum(-1)
        shares = counts / total.clamp_min(1)[..., None]
        covered = total >= MINIMUM_COVERAGE * values.shape[-1]
        return shares, covered

    def token_heights(
        self,
        study_uids: list[str],
        token_study: torch.Tensor,
        patient_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = torch.tensor(
            [self.rows.get(uid, -1) for uid in study_uids],
            device=patient_position.device,
        )[token_study]
        plane = self.planes[row.clamp_min(0)]
        known = (row >= 0) & (plane[:, 3:6].abs().sum(-1) > 0)
        height = (
            (patient_position.float() - plane[:, None, :3]) * plane[:, None, 3:6]
        ).sum(-1)
        return height.nan_to_num(0.0), known

    def study_centres(self, study_uids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        row = torch.tensor(
            [self.rows.get(uid, -1) for uid in study_uids], device=self.planes.device
        )
        plane = self.planes[row.clamp_min(0)]
        known = (row >= 0) & (plane[:, 3:6].abs().sum(-1) > 0)
        return plane[:, :3], known


def anatomy_loss(
    logits: torch.Tensor,
    shares: torch.Tensor,
    mask: torch.Tensor,
    names: tuple[str, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = logits.float()
    if not mask.any():
        return logits.sum() * 0.0, {}
    chosen = logits[mask]
    target = shares[mask]
    loss = -(target * chosen.log_softmax(-1)).sum(-1).mean()
    shares = chosen.softmax(-1)
    present = target.sum(0) > 0
    overlap = (
        2 * (shares * target).sum(0) / (shares.sum(0) + target.sum(0)).clamp_min(1e-6)
    )
    loss = loss + (1 - overlap[present]).mean()
    with torch.no_grad():
        probability = chosen.softmax(-1)
        metrics = {
            "agreement": (probability.argmax(-1) == target.argmax(-1)).float().mean(),
            "share_error": (probability - target).abs().sum(-1).mean() / 2,
        }
        for item, name in enumerate(names[1:], start=1):
            both = probability[:, item].sum() + target[:, item].sum()
            metrics[f"overlap_{name}"] = (
                2
                * torch.minimum(probability[:, item], target[:, item]).sum()
                / both.clamp_min(1e-6)
            )
    return loss, metrics
