import json
import os
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    inv_ornt_aff,
    io_orientation,
    ornt_transform,
)

ROOT = Path("/home/claudio/.cache/knee-segmenters")
BENCH = ROOT / sys.argv[1]
NAME = sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
MODEL = (
    ROOT
    / "mrannotator_weights/Dataset002_ShoulderKnee/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
)
MEMORY_LIMIT_KB = 14 * 1024 * 1024
READ_AHEAD = 6
STEP = float(os.environ.get("STEP", "0.75"))
SKIP_AIR = os.environ.get("SKIP_AIR", "0") == "1"
SQUASH = float(os.environ.get("SQUASH", "1"))
REMAP = torch.tensor([0, 7, 9, 10, 8, 0, 0, 0, 0, 0, 0], dtype=torch.uint8)


def read(path):
    image = nib.load(str(path))
    change = ornt_transform(io_orientation(image.affine), axcodes2ornt(("L", "A", "S")))
    array = apply_orientation(np.asarray(image.dataobj, dtype=np.float32), change)
    affine = image.affine @ inv_ornt_aff(change, image.shape)
    return path, np.ascontiguousarray(array), affine


def write(job):
    labels, affine, target = job
    nib.save(nib.Nifti1Image(labels, affine), str(target))
    return True


def starts(length, patch):
    if length <= patch:
        return [0]
    count = int(np.ceil((length - patch) / (patch * STEP))) + 1
    return [int(round(v)) for v in np.linspace(0, length - patch, count)]


def predict(network, data, patch, weight):
    pad = [max(p - s, 0) for p, s in zip(patch, data.shape)]
    before = [p // 2 for p in pad]
    data = F.pad(
        data,
        (
            before[2],
            pad[2] - before[2],
            before[1],
            pad[1] - before[1],
            before[0],
            pad[0] - before[0],
        ),
    )
    corners = [
        (a, b, c)
        for a in starts(data.shape[0], patch[0])
        for b in starts(data.shape[1], patch[1])
        for c in starts(data.shape[2], patch[2])
    ]
    total = None
    count = torch.zeros(data.shape, device=data.device)
    floor = data.min()
    tissue = data > floor + 0.1 * (torch.quantile(data.flatten()[::97], 0.99) - floor)
    used = 0
    for a, b, c in corners:
        if (
            SKIP_AIR
            and tissue[a : a + patch[0], b : b + patch[1], c : c + patch[2]]
            .float()
            .mean()
            < 0.05
        ):
            continue
        used += 1
        with torch.autocast("cuda"):
            scores = network(
                data[None, None, a : a + patch[0], b : b + patch[1], c : c + patch[2]]
            )[0].float()
        if total is None:
            total = torch.zeros(
                (scores.shape[0], *data.shape), device=data.device, dtype=torch.float16
            )
        total[:, a : a + patch[0], b : b + patch[1], c : c + patch[2]] += (
            scores * weight
        ).half()
        count[a : a + patch[0], b : b + patch[1], c : c + patch[2]] += weight
    total = total[
        :,
        before[0] : data.shape[0] - (pad[0] - before[0]),
        before[1] : data.shape[1] - (pad[1] - before[1]),
        before[2] : data.shape[2] - (pad[2] - before[2]),
    ]
    count = count[
        before[0] : data.shape[0] - (pad[0] - before[0]),
        before[1] : data.shape[1] - (pad[1] - before[1]),
        before[2] : data.shape[2] - (pad[2] - before[2]),
    ]
    return total, count, used


def nearest(length, wanted, device):
    return (
        ((torch.arange(wanted, device=device) + 0.5) * length / wanted)
        .floor()
        .long()
        .clamp(0, length - 1)
    )


def main():
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    out = BENCH / NAME
    out.mkdir(exist_ok=True)
    paths = [
        p
        for p in sorted((BENCH / "images").glob("*_0000.nii.gz"))
        if not (out / p.name.replace("_0000", "")).exists()
    ][:LIMIT]
    predictor = nnUNetPredictor(
        tile_step_size=0.75,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda"),
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL), use_folds=(0,), checkpoint_name="checkpoint_final.pth"
    )
    network = predictor.network.cuda().eval()
    patch = tuple(predictor.configuration_manager.patch_size)
    weight = compute_gaussian(
        patch, sigma_scale=1.0 / 8, value_scaling_factor=10, device=torch.device("cuda")
    ).float()
    target = np.array(predictor.configuration_manager.spacing)
    remap = REMAP.cuda()
    patches = 0
    start = time.perf_counter()
    with (
        ThreadPoolExecutor(2) as readers,
        ThreadPoolExecutor(2) as writers,
        torch.inference_mode(),
    ):
        pending = []
        for path, array, affine in (
            item
            for begin in range(0, len(paths), READ_AHEAD)
            for item in list(readers.map(read, paths[begin : begin + READ_AHEAD]))
        ):
            spacing = np.linalg.norm(affine[:3, :3], axis=0)[[0, 2, 1]]
            volume = torch.from_numpy(array).cuda().permute(0, 2, 1)
            filled = (volume != 0).nonzero()
            low, high = filled.min(0).values, filled.max(0).values + 1
            cropped = volume[low[0] : high[0], low[1] : high[1], low[2] : high[2]]
            cropped = (cropped - cropped.mean()) / cropped.std().clamp_min(1e-8)
            shape = [
                int(v) for v in np.round(np.array(cropped.shape) * spacing / target)
            ]
            coarse = int(np.argmax(spacing))
            shape[coarse] = max(int(round(shape[coarse] / SQUASH)), 1)
            order = [coarse] + [axis for axis in range(3) if axis != coarse]
            back = [order.index(axis) for axis in range(3)]
            moved = cropped.permute(*order)
            wanted = [shape[axis] for axis in order]
            planes = F.interpolate(
                moved[:, None], size=wanted[1:], mode="bicubic", align_corners=False
            )[:, 0]
            data = (
                planes[nearest(planes.shape[0], wanted[0], planes.device)]
                .permute(*back)
                .contiguous()
            )
            total, count, used = predict(network, data, patch, weight)
            patches += used
            total = total.permute(0, *[1 + axis for axis in order])
            count = count.permute(*order)
            index = nearest(wanted[0], moved.shape[0], total.device)
            scores = total[:, index].float() / count[index].clamp_min(1e-6)
            restored = F.interpolate(
                scores.transpose(0, 1),
                size=tuple(moved.shape[1:]),
                mode="bilinear",
                align_corners=False,
            ).argmax(1)
            labels = torch.zeros(volume.shape, dtype=torch.uint8, device=volume.device)
            labels[low[0] : high[0], low[1] : high[1], low[2] : high[2]] = remap[
                restored.permute(*back)
            ]
            pending.append(
                writers.submit(
                    write,
                    (
                        labels.permute(0, 2, 1).contiguous().cpu().numpy(),
                        affine,
                        out / path.name.replace("_0000", ""),
                    ),
                )
            )
            pending = [job for job in pending if not job.done() or not job.result()]
            del total, count, scores, restored, data, planes
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MEMORY_LIMIT_KB:
                raise SystemExit("memory limit reached, stopping")
        for job in pending:
            job.result()
    seconds = time.perf_counter() - start
    count = max(len(paths), 1)
    report = {
        "volumes": len(paths),
        "seconds": round(seconds, 1),
        "per_volume": round(seconds / count, 2),
        "patches_per_volume": round(patches / count, 1),
        "peak_memory_gb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1
        ),
        "gpu_peak_mib": torch.cuda.max_memory_allocated() // 2**20,
    }
    (out / "timing.json").write_text(json.dumps(report, indent=1))
    print(report)


if __name__ == "__main__":
    main()
