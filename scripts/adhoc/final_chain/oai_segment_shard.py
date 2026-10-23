import json
import sys
import time
from pathlib import Path

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

CACHE = Path.home() / ".cache/knee-segmenters"
MODEL = CACHE / "nnunet_knee/models/Dataset500_KneeMRI" / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"


def main() -> None:
    root = CACHE / sys.argv[1]
    shard, shards, workers = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    out = root / "nnunet_knee"
    out.mkdir(exist_ok=True)
    images = sorted((root / "images").glob("*_0000.nii.gz"))[shard::shards]
    images = [path for path in images if not (out / path.name.replace("_0000", "")).exists()]
    print("shard", shard, "volumes to segment", len(images), flush=True)
    start = time.perf_counter()
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, device=torch.device("cuda"), verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(MODEL), use_folds=(1,), checkpoint_name="checkpoint_final.pth")
    predictor.predict_from_files(
        [[str(path)] for path in images],
        [str(out / path.name.replace("_0000.nii.gz", "")) for path in images],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=workers,
        num_processes_segmentation_export=workers,
    )
    print(json.dumps({"shard": shard, "volumes": len(images), "seconds": round(time.perf_counter() - start, 1), "gpu_peak_mib": torch.cuda.max_memory_allocated() // 2**20}), flush=True)


if __name__ == "__main__":
    main()
