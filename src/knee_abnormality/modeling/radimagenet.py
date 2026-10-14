import argparse
import hashlib
from pathlib import Path
from typing import NamedTuple

import torch

REFINED_RADIMAGENET_URL = (
    "https://huggingface.co/convergedmachine/RadImagenet/resolve/"
    "7fa62dccee2b14e49188d11c4b0e66d88c29b41e/{name}"
)


class RadImageNetWeights(NamedTuple):
    file: str
    sha256: str
    source: str


RADIMAGENET_WEIGHTS = {
    "resnet50.a1_in1k": RadImageNetWeights(
        "refined_radimagenet_resnet50.pth",
        "b6daa90de92589c404e9da30f4c0ce49182763d4a952f1222caeec647237bacf",
        REFINED_RADIMAGENET_URL.format(name="resnet50.pth"),
    ),
}


def radimagenet_path(backbone: str) -> Path:
    return (
        Path(torch.hub.get_dir()) / "checkpoints" / RADIMAGENET_WEIGHTS[backbone].file
    )


def resnet_avg_down_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    renamed = {}
    for key, value in state.items():
        parts = key.split(".")
        if (
            parts[0] in ("layer1", "layer2", "layer3", "layer4")
            and parts[1:3] == ["0", "downsample"]
            and parts[3] in ("0", "1")
        ):
            parts[3] = str(int(parts[3]) + 1)
        renamed[".".join(parts)] = value
    return renamed


def radimagenet_state_dict(backbone: str) -> dict[str, torch.Tensor]:
    weights = RADIMAGENET_WEIGHTS[backbone]
    path = radimagenet_path(backbone)
    if not path.is_file():
        raise FileNotFoundError(
            f"RadImageNet weights missing: fetch {weights.source} and save it as {path}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != weights.sha256:
        raise ValueError(f"RadImageNet weights hash mismatch for {path}: {digest}")
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["model"].items()
        if not key.removeprefix("_orig_mod.").startswith("fc.")
    }
