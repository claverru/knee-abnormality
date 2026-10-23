import fcntl
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import polars as pl
import torch

from knee_abnormality.config import ExperimentConfig, load_config
from knee_abnormality.data.dataset import KneeStudyDataset, build_data_loader
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.inference.checkpoint import checkpoint_model, read_checkpoint
from knee_abnormality.inference.engine import move_batch, resolve_precision
from knee_abnormality.inference.reflection import reflect_batch, sagittal_types
from knee_abnormality.provenance import sha256_file

DATASETS = {"fastmri": Path("/workspace/data/fastmri"), "oai": Path("/workspace/data/oai")}
MULTIPLIER = 4
TEMPERATURE = 0.85
STUDY = "StudyInstanceUID"


def main() -> None:
    checkpoint_path, output = Path(sys.argv[1]), Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    output.mkdir(parents=True, exist_ok=True)
    lock = open("/workspace/predictions/silver15/.gpu.lock2", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    print("lock acquired", time.strftime("%H:%M:%S"), flush=True)
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.3, 0)
    torch.set_float32_matmul_precision("high")
    precision, autocast = resolve_precision("auto", device)
    checkpoint = read_checkpoint(checkpoint_path)
    checkpoint["state_dict"] = {name: value for name, value in checkpoint["state_dict"].items() if "mask_decoder." not in name}
    saved = ExperimentConfig.model_validate(checkpoint["hyper_parameters"]["config"])
    targets = tuple(checkpoint["hyper_parameters"]["targets"])
    model = checkpoint_model(checkpoint, saved, targets, device, TEMPERATURE)
    sagittal = sagittal_types(model.augmentation_settings.series_types)
    middle_only = model.repeated_middle_slice
    record = {"checkpoint": str(checkpoint_path), "sha256": sha256_file(checkpoint_path), "fold": checkpoint["hyper_parameters"]["fold"], "backbone": saved.model.backbone, "image_multiplier": MULTIPLIER, "attention_temperature": TEMPERATURE, "reflection_tta": True, "precision": precision}
    base = Path("configs/resnet50.yaml").resolve()
    for name, root in DATASETS.items():
        target = output / f"{name}.parquet"
        if target.exists():
            continue
        started = time.perf_counter()
        config = load_config(base, data_root=root, label_manifest=None).model_copy(update={"data": saved.data})
        _, studies, split = prediction_studies(config, "test", 0)
        index = build_series_index(config.paths.data_root, config.data, split, studies[STUDY].to_list())
        loader = build_data_loader(KneeStudyDataset(studies, index, config.data, targets, None, MULTIPLIER), config.trainer, 8, num_workers=workers, pin_memory=False, persistent_workers=False, prepare_on_loader=False)
        loader.in_order = False
        uids, rows = [], []
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=autocast, enabled=autocast is not None):
            for batch in loader:
                uids += batch["study_uid"]
                batch = with_images(move_batch(batch, device), middle_only)
                mirrored = reflect_batch(batch, sagittal, middle_only)
                eligible = batch["reflection_eligible"][:, None]
                angle = model.first_look_pose(batch)
                plain = torch.sigmoid(model(batch, angle).float())
                mirror = torch.sigmoid(model(mirrored, angle).float())
                rows.append(torch.where(eligible, (plain + mirror) / 2, plain).cpu())
        values = torch.cat(rows).numpy()
        frame = pl.DataFrame({STUDY: uids, **{target: values[:, column] for column, target in enumerate(targets)}}).sort(STUDY)
        if frame.height != studies.height or frame[STUDY].n_unique() != studies.height:
            raise ValueError(f"{name}: predictions do not cover every study once")
        frame.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        record[name] = {"studies": frame.height, "seconds": round(time.perf_counter() - started, 1)}
        print(name, record[name], flush=True)
    (output / "record.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
