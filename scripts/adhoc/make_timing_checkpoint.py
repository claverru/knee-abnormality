import json
import sys
from pathlib import Path

import pytorch_lightning as L
import torch
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import ExperimentConfig
from knee_abnormality.identity import preprocessing_identity
from knee_abnormality.modeling.model import KneeAbnormalityModel
from knee_abnormality.provenance import sha256_file


class TimingCheckpointCLI(CLISettings):
    source: Path = Path("outputs/submission/timing-profile-random-weights.ckpt")
    output: Path = Path("outputs/submission/timing-profile-random-weights.ckpt")
    seed: int = 20260918


def main() -> None:
    try:
        args = TimingCheckpointCLI()
        source = torch.load(args.source, map_location="cpu", weights_only=False)
        parameters = source["hyper_parameters"]
        config = ExperimentConfig.model_validate(parameters["config"])
        targets = parameters["targets"]
        torch.manual_seed(args.seed)
        model = KneeAbnormalityModel(
            config.model, config.data, targets, load_pretrained=False
        )
        identity = preprocessing_identity(config.model.backbone)
        checkpoint = {
            "state_dict": {
                f"model.{name}": value for name, value in model.state_dict().items()
            },
            "hyper_parameters": {**parameters, "preprocessing_identity": identity},
            "pytorch-lightning_version": L.__version__,
            "epoch": 0,
            "global_step": 0,
            "timing_profile_only": True,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": sha256_file(args.output),
                    "seed": args.seed,
                    "backbone": config.model.backbone,
                    "targets": len(targets),
                    "fold": parameters["fold"],
                    "sources_sha256": identity["sources_sha256"],
                    "parameters": sum(
                        value.numel() for value in model.state_dict().values()
                    ),
                },
                indent=2,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError, SettingsError) as error:
        print(f"Timing checkpoint error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
