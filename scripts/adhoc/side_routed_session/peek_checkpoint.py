import json
import sys
from pathlib import Path

import torch

run = Path(sys.argv[1])
state = torch.load(next((run / "fold_1").glob("best-*.ckpt")), map_location="cpu", weights_only=False)["state_dict"]
print(len(state), sorted({key.split(".")[0] for key in state}))
print({key: tuple(value.shape) for key, value in state.items() if key.startswith("report_attribute_heads")})
print(json.loads((run / "effective_config.json").read_text())["paths"]["report_attribute_manifest"])
