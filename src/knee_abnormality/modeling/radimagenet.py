import argparse
import hashlib
from pathlib import Path
from typing import Literal, NamedTuple

import torch

REFINED_RADIMAGENET_URL = (
    "https://huggingface.co/convergedmachine/RadImagenet/resolve/"
    "7fa62dccee2b14e49188d11c4b0e66d88c29b41e/{name}"
)
SEQUENTIAL_PREFIXES = {
    "0": "conv1",
    "1": "bn1",
    "4": "layer1",
    "5": "layer2",
    "6": "layer3",
    "7": "layer4",
}


class RadImageNetWeights(NamedTuple):
    file: str
    sha256: str
    source: str
    layout: Literal["sequential", "refined"]


RADIMAGENET_WEIGHTS = {
    "resnet50.tv_in1k": RadImageNetWeights(
        "radimagenet_resnet50.pt",
        "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734",
        "kaggle datasets download marwanmath/resnet-50-radimagenet-marwan --unzip",
        "sequential",
    ),
    "resnet50.a1_in1k": RadImageNetWeights(
        "refined_radimagenet_resnet50.pth",
        "b6daa90de92589c404e9da30f4c0ce49182763d4a952f1222caeec647237bacf",
        REFINED_RADIMAGENET_URL.format(name="resnet50.pth"),
        "refined",
    ),
    "resnet18.a1_in1k": RadImageNetWeights(
        "refined_radimagenet_resnet18.pth",
        "c8c9259ac24068d1986bce1d1828f32b396642fc5ac86d17ab86baa0a13d344f",
        REFINED_RADIMAGENET_URL.format(name="resnet18.pth"),
        "refined",
    ),
    "resnet10t.c3_in1k": RadImageNetWeights(
        "refined_radimagenet_resnet10t.pth",
        "f5219dff128ca752d47e8acd07817564c10bacbbc4241b3a31845e1a67a079ab",
        REFINED_RADIMAGENET_URL.format(name="resnet10t.pth"),
        "refined",
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
    if weights.layout == "sequential":
        state = torch.load(path, map_location="cpu", weights_only=True)
        renamed = {}
        for key, value in state.items():
            _, index, rest = key.split(".", 2)
            renamed[f"{SEQUENTIAL_PREFIXES[index]}.{rest}"] = value
        return renamed
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["model"].items()
        if not key.removeprefix("_orig_mod.").startswith("fc.")
    }
