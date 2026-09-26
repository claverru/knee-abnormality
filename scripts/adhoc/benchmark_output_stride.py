import time

import timm
import torch
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe


class OutputStrideCLI(CLISettings):
    model_config = describe(
        "Time one training forward and backward pass of the backbone at each output stride and record peak CUDA memory"
    )
    backbone: str = Field(default="rexnet_200.nav_in1k", description="timm backbone")
    images: int = Field(
        default=208, description="Triplets per step (8 studies x 26 triplets)"
    )
    image_size: int = Field(default=224, description="Input side in pixels")
    strides: list[int] = Field(
        default=[32, 16], description="Output strides to compare"
    )
    warmup: int = Field(default=3, description="Untimed steps per stride")
    steps: int = Field(default=10, description="Timed steps per stride")


def main() -> None:
    cli = OutputStrideCLI()
    torch.set_float32_matmul_precision("high")
    for stride in cli.strides:
        model = (
            timm.create_model(
                cli.backbone,
                pretrained=False,
                in_chans=3,
                num_classes=0,
                global_pool="",
                output_stride=stride,
            )
            .cuda()
            .train()
        )
        inputs = torch.randn(
            cli.images, 3, cli.image_size, cli.image_size, device="cuda"
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(cli.warmup + cli.steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = model.forward_features(inputs)
            features.float().square().mean().backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if step >= cli.warmup:
                times.append(time.perf_counter() - start)
        print(
            f"stride {stride}: features {tuple(features.shape)}, "
            f"{sum(times) / len(times):.4f} s/step, "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        )
        del model, inputs, features
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
