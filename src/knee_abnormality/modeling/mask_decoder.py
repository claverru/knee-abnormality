import torch
import torch.nn.functional as F
from torch import nn

ANATOMY_CLASS_COUNT = 7
SOFT_TISSUE_CLASS_COUNT = 7
(
    MEDIAL_MENISCUS_CLASS,
    LATERAL_MENISCUS_CLASS,
    MEDIAL_TIBIAL_CARTILAGE_CLASS,
    LATERAL_TIBIAL_CARTILAGE_CLASS,
    FEMORAL_CARTILAGE_CLASS,
    PATELLAR_CARTILAGE_CLASS,
) = range(1, SOFT_TISSUE_CLASS_COUNT)
(
    FEMUR_CLASS,
    TIBIA_CLASS,
    KNEECAP_CLASS,
    FIBULA_CLASS,
    JOINT_CLASS,
    KNEECAP_JOINT_CLASS,
) = range(1, ANATOMY_CLASS_COUNT)


class MaskDecoder(nn.Module):
    def __init__(self, widths: tuple[int, int, int]) -> None:
        super().__init__()
        self.second = nn.Conv2d(widths[0], 64, 1)
        self.third = nn.Conv2d(widths[1], 64, 1)
        self.fourth = nn.Conv2d(widths[2], 64, 1)
        self.third_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.second_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.bone = nn.Conv2d(64, ANATOMY_CLASS_COUNT, 1)
        self.soft_tissue = nn.Conv2d(64, SOFT_TISSUE_CLASS_COUNT, 1)

    def forward(
        self, second: torch.Tensor, third: torch.Tensor, fourth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        merged = self.third_block(
            self.third(third)
            + F.interpolate(
                self.fourth(fourth),
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
        )
        merged = self.second_block(
            self.second(second)
            + F.interpolate(
                merged, scale_factor=2, mode="bilinear", align_corners=False
            )
        )
        return (
            self.bone(merged).flatten(2).transpose(1, 2),
            self.soft_tissue(merged).flatten(2).transpose(1, 2),
        )
