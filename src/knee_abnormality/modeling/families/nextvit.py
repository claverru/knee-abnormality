import torch
from timm.models.nextvit import NextViT
from torch import nn


class NextViTBackbone(nn.Module):
    def __init__(self, original: NextViT) -> None:
        super().__init__()
        self.stem = original.stem
        self.stages = original.stages
        self.norm = original.norm
        self.num_features = original.num_features
        self.feature_info = original.feature_info
        self.pretrained_cfg = original.pretrained_cfg

    def group_matcher(self, coarse: bool = False) -> dict:
        return {
            "stem": r"^stem",
            "blocks": [
                (r"^stages\.(\d+)", None),
                (r"^norm", (3,)),
            ],
        }

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.norm(self.stages(self.stem(images)))
