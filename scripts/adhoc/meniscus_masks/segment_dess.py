import json
import sys
import time
from pathlib import Path

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

CACHE = Path.home() / ".cache/knee-segmenters"
MODEL = (
    CACHE
    / "nnunet_knee/models/Dataset500_KneeMRI"
    / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
)


def main():
    root = CACHE / sys.argv[1]
    out = root / "nnunet_knee"
    out.mkdir(exist_ok=True)
    images = [
        path
        for path in sorted((root / "images").glob("*_0000.nii.gz"))
        if not (out / path.name.replace("_0000", "")).exists()
    ]
    print("volumes to segment", len(images), flush=True)
    if not images:
        return
    start = time.perf_counter()
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=torch.device("cuda"),
        verbose=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL), use_folds=(1,), checkpoint_name="checkpoint_final.pth"
    )
    predictor.predict_from_files(
        [[str(path)] for path in images],
        [str(out / path.name.replace("_0000.nii.gz", "")) for path in images],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )
    seconds = time.perf_counter() - start
    timing = {
        "volumes": len(images),
        "seconds": round(seconds, 1),
        "gpu_peak_mib": torch.cuda.max_memory_allocated() // 2**20,
    }
    (out / "timing.json").write_text(json.dumps(timing))
    print(timing)


if __name__ == "__main__":
    main()
