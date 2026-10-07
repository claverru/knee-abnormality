import torch
from torch import nn


class PlaneContrastReadout(nn.Module):
    def __init__(self, shared: nn.Linear, type_count: int) -> None:
        super().__init__()
        if type_count != 6:
            raise ValueError("Plane/contrast readout requires six series types")
        initial = shared.weight.detach().clone()
        self.weight = nn.Parameter(initial[None].repeat(type_count, 1, 1))
        self.register_buffer("initial_weight", initial)

    def forward(self, values: torch.Tensor, image_type: torch.Tensor) -> torch.Tensor:
        grouped = values.float().reshape(len(image_type), -1, values.shape[-1])
        weights = self.weight.float()[image_type].transpose(1, 2)
        return torch.bmm(grouped, weights).flatten(0, 1)

    def divergence(self) -> torch.Tensor:
        difference = self.weight.detach().float() - self.initial_weight.float()[None]
        return difference.square().mean(-1).sqrt()
