import importlib
import importlib.util
import sys
import types

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

STAGED_BACKBONES = {
    "mamba_vision_T": {
        "module": "mambavision.models.mamba_vision",
        "url": "https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar",
        "widths": (160, 320, 640),
    },
}


class StagedBackbone(nn.Module):
    def __init__(
        self,
        stem: nn.Module,
        stages: list[nn.Module],
        norm: nn.Module,
        widths: tuple[int, int, int],
        pretrained_cfg: dict,
    ) -> None:
        super().__init__()
        self.stem = stem
        self.stages = nn.ModuleList(stages)
        self.norm = norm
        self.stage_widths = widths
        self.num_features = widths[-1]
        self.pretrained_cfg = pretrained_cfg

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        for stage in self.stages:
            features = stage(features)
        return self.norm(features)

    def group_matcher(self, coarse: bool = False) -> dict:
        return {"stem": r"^stem", "blocks": r"^stages\.(\d+)"}

    def geometry_layers(self, stages: int) -> list[nn.Module]:
        modules = [self.stem, *(stage.downsample for stage in self.stages[:stages])]
        return [
            layer
            for module in modules
            for layer in module.modules()
            if isinstance(layer, nn.Conv2d)
        ]


def _stage(downsample: nn.Module | None, blocks: nn.Module) -> nn.Sequential:
    stage = nn.Sequential()
    stage.add_module("downsample", downsample or nn.Identity())
    stage.add_module("blocks", blocks)
    return stage


def _detached_downsamples(layers: nn.ModuleList) -> list[nn.Module | None]:
    downsamples = [None]
    for layer in layers[:-1]:
        downsamples.append(layer.downsample)
        layer.downsample = None
    return downsamples


def create_staged_backbone(
    name: str, pretrained: bool, stochastic_depth: float
) -> StagedBackbone:
    settings = STAGED_BACKBONES[name]
    if "mamba_ssm" not in sys.modules:
        package = types.ModuleType("mamba_ssm")
        package.__path__ = list(
            importlib.util.find_spec("mamba_ssm").submodule_search_locations
        )
        sys.modules["mamba_ssm"] = package
    original = getattr(importlib.import_module(settings["module"]), name)(
        pretrained=False
    )
    if pretrained:
        state = torch.hub.load_state_dict_from_url(
            settings["url"], map_location="cpu", check_hash=False
        )
        original.load_state_dict(state["state_dict"], strict=True)
    layers = original.levels
    backbone = StagedBackbone(
        original.patch_embed,
        [
            _stage(downsample, layer)
            for downsample, layer in zip(_detached_downsamples(layers), layers)
        ],
        original.norm,
        settings["widths"],
        {"url": settings["url"], "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    )
    rates = torch.linspace(
        0, stochastic_depth, sum(1 for _ in _drop_paths(backbone))
    ).tolist()
    for drop_path, rate in zip(_drop_paths(backbone), rates):
        drop_path.drop_prob = rate
    return backbone


def _drop_paths(backbone: nn.Module) -> list[nn.Module]:
    return [
        module for module in backbone.modules() if type(module).__name__ == "DropPath"
    ]
