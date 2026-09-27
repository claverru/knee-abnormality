import hashlib
import json
import math
from pathlib import Path

import polars as pl
import torch
from torch import nn

from knee_abnormality.modeling.model import MIL_ATTENTION_WIDTH, PROJECTION_WIDTH

SCANNER_TABLE_SHA256 = (
    "8714fa8b025afe0ff47d0811ca735f0ec75c84808b462486db19553d0c3058f4"
)
SCANNER_GROUPS = (
    "GE/1.5T",
    "GE/3T",
    "GE/unknown",
    "Philips/1.5T",
    "Philips/3T",
    "Siemens/1.5T",
    "Siemens/3T",
    "minor",
)


def load_scanner_groups(path: Path) -> dict[str, int]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != SCANNER_TABLE_SHA256:
        raise ValueError("Scanner header table SHA-256 differs from the fold protocol")
    frame = pl.read_parquet(path).select(
        "StudyInstanceUID", "vendor", "MagneticFieldStrength"
    )
    groups = {name: index for index, name in enumerate(SCANNER_GROUPS)}
    targets = {}
    for uid, vendor, strength in frame.iter_rows():
        if vendor in {"Siemens", "Philips", "GE"}:
            field = (
                "unknown"
                if strength is None
                else ("3T" if float(strength) >= 2 else "1.5T")
            )
            group = f"{vendor}/{field}"
        else:
            group = "minor"
        if group not in groups or uid in targets:
            raise ValueError(
                "Scanner header table has an unexpected or repeated study group"
            )
        targets[uid] = groups[group]
    if set(targets.values()) != set(groups.values()):
        raise ValueError("Scanner header table does not cover all eight groups")
    return targets


def scanner_group_prior(
    split_dir: Path, fold: int, targets: dict[str, int]
) -> tuple[torch.Tensor, list[int]]:
    protocol = json.loads((split_dir / "protocol.json").read_text())
    if protocol["scanner_groups"]["sha256"] != SCANNER_TABLE_SHA256:
        raise ValueError("Scanner table SHA-256 differs from the fold protocol")
    studies_path = split_dir / "studies.csv"
    if (
        hashlib.sha256(studies_path.read_bytes()).hexdigest()
        != protocol["manifest_sha256"]
    ):
        raise ValueError("Scanner prior membership differs from the fold protocol")
    membership = pl.read_csv(studies_path).select("StudyInstanceUID", "fold")
    if membership["StudyInstanceUID"].n_unique() != membership.height:
        raise ValueError("Scanner prior membership contains repeated studies")
    if fold not in range(protocol["n_splits"]):
        raise ValueError("Scanner prior fold is outside the split protocol")
    if (
        membership["fold"].null_count()
        or not membership["fold"].is_in(range(protocol["n_splits"])).all()
    ):
        raise ValueError("Scanner prior membership has an invalid fold")
    training_uids = membership.filter(pl.col("fold") != fold)["StudyInstanceUID"]
    if any(uid not in targets for uid in training_uids):
        raise ValueError("Training study lacks a scanner-group target")
    counts = torch.bincount(
        torch.tensor([targets[uid] for uid in training_uids], dtype=torch.long),
        minlength=len(SCANNER_GROUPS),
    )
    if not counts.all():
        raise ValueError("Scanner prior has an empty training group")
    return counts.float() / counts.sum(), counts.tolist()


def confusion_ramp(step: int, total_steps: int) -> float:
    progress = min(1.0, step / max(1, total_steps - 1))
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


class ScannerConfusionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(PROJECTION_WIDTH, MIL_ATTENTION_WIDTH),
            nn.Tanh(),
            nn.Linear(MIL_ATTENTION_WIDTH, 1),
        )
        self.classifier = nn.Linear(PROJECTION_WIDTH, len(SCANNER_GROUPS))

    def forward(
        self,
        tokens: torch.Tensor,
        token_study: torch.Tensor,
        token_valid: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            tokens = tokens.float()
            membership = (
                torch.arange(batch_size, device=tokens.device)[:, None]
                == token_study[None]
            ) & token_valid[None]
            if not membership.any(-1).all():
                raise ValueError("A study has no visually valid spatial tokens")
            attention = self.attention(tokens).squeeze(-1)
            attention = attention.unsqueeze(0).masked_fill(~membership, -torch.inf)
            pooled = attention.softmax(-1) @ tokens
            return self.classifier(pooled)


def scanner_group_labels(
    study_uids: list[str], targets: dict[str, int], device: torch.device
) -> torch.Tensor:
    if any(uid not in targets for uid in study_uids):
        raise ValueError("Training study lacks a scanner-group target")
    group = torch.tensor(
        [targets[uid] for uid in study_uids], device=device, dtype=torch.long
    )
    return group
