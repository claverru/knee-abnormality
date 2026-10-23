import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

SOURCE = Path("/home/claudio/.cache/knee-segmenters") / sys.argv[1]
TARGETS = Path(__file__).resolve().parents[3] / "data/labels/prepared" / sys.argv[2]
JOINT_ABOVE_PLATEAU_MM = 6.4


def largest(mask):
    parts, count = ndimage.label(mask)
    if count <= 1:
        return mask
    return parts == (1 + int(np.argmax(ndimage.sum(mask, parts, range(1, count + 1)))))


def measure(knee):
    if "plateau_normal_x" not in knee:
        return {"study": knee["study"]}
    image = nib.as_closest_canonical(
        nib.load(str(SOURCE / "mrannotator" / f"{knee['study']}.nii.gz"))
    )
    raw = np.asarray(image.dataobj)
    zooms = np.array(image.header.get_zooms()[:3], dtype=float)
    affine = image.affine
    femur, tibia, kneecap = largest(raw == 7), largest(raw == 8), largest(raw == 9)
    centre = np.array([knee[f"plateau_centre_{a}"] for a in "xyz"]) * np.array(
        [-1.0, -1.0, 1.0]
    )
    normal = np.array([knee[f"plateau_normal_{a}"] for a in "xyz"]) * np.array(
        [-1.0, -1.0, 1.0]
    )
    out = {"study": knee["study"]}
    if kneecap.any():
        points = np.argwhere(kneecap) @ affine[:3, :3].T + affine[:3, 3] - centre
        out["kneecap_front_mm"] = float(np.percentile(points[:, 1], 98))
        out["kneecap_back_mm"] = float(np.percentile(points[:, 1], 2))
        out["kneecap_top_mm"] = float(np.percentile(points @ normal, 98))
        out["kneecap_bottom_mm"] = float(np.percentile(points @ normal, 2))
    femur_front = np.argwhere(femur) @ affine[:3, :3].T + affine[:3, 3] - centre
    out["femur_front_mm"] = float(np.percentile(femur_front[:, 1], 99))
    height = np.arange(raw.shape[2])[None, None, :]
    tibia_top = np.where(
        tibia.any(2), raw.shape[2] - 1 - np.argmax(tibia[:, :, ::-1], 2), -1
    )
    above = femur & (height > tibia_top[:, :, None])
    femur_bottom = np.where(above.any(2), np.argmax(above, 2), -1)
    gap = (femur_bottom - tibia_top - 1) * zooms[2]
    both = (tibia_top >= 0) & (femur_bottom >= 0) & (gap >= 0) & (gap < 15)
    columns = np.argwhere(both)
    across = (
        np.c_[columns, tibia_top[both]] @ affine[:3, :3].T + affine[:3, 3] - centre
    )[:, 0]
    values = gap[both]
    for name, chosen in (("patient_right", across > 8), ("patient_left", across < -8)):
        if chosen.sum() >= 30:
            out[f"gap_{name}_mm"] = float(np.percentile(values[chosen], 10))
            out[f"gap_{name}_median_mm"] = float(np.median(values[chosen]))
    return out


def main():
    knees = json.loads((TARGETS / "knees.json").read_text())
    with ProcessPoolExecutor(3) as pool:
        rows = list(pool.map(measure, knees, chunksize=8))
    (TARGETS / "measurements.json").write_text(json.dumps(rows, indent=1))
    for key in (
        "kneecap_front_mm",
        "kneecap_back_mm",
        "kneecap_top_mm",
        "kneecap_bottom_mm",
        "femur_front_mm",
        "gap_patient_right_mm",
        "gap_patient_left_mm",
        "gap_patient_right_median_mm",
    ):
        v = np.array([r[key] for r in rows if key in r])
        print(
            f"{key:30} n {len(v):3d} mean {v.mean():+6.1f} sd {v.std():5.1f} 5th {np.percentile(v, 5):+6.1f} 95th {np.percentile(v, 95):+6.1f}"
        )


if __name__ == "__main__":
    main()
