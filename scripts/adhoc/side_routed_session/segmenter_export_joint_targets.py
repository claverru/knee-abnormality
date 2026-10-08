import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

SOURCE = Path("/home/claudio/.cache/knee-segmenters") / sys.argv[1]
BASE = Path("/home/claudio/src/knee-abnormality-side-routed-heads/data/labels/prepared")
PREVIOUS = BASE / sys.argv[2]
OUT = BASE / sys.argv[3]
WORKERS = 4
SPACING = 2.5
CLOSING_MM = 4.0
WORKING_MM = 1.0
ANGLE = 135.0
FLIP = np.array([-1.0, -1.0, 1.0])
BONES = (("femur", 7, 1), ("tibia", 8, 2), ("kneecap", 9, 3), ("fibula", 10, 4))
JOINT, KNEECAP_JOINT = 5, 6


def largest(mask):
    parts, count = ndimage.label(mask)
    if count <= 1:
        return mask
    return parts == (1 + int(np.argmax(ndimage.sum(mask, parts, range(1, count + 1)))))


def closed(mask, zooms):
    if not mask.any():
        return mask
    grown = ndimage.distance_transform_edt(~mask, sampling=zooms) <= CLOSING_MM
    mask = (ndimage.distance_transform_edt(grown, sampling=zooms) > CLOSING_MM) | mask
    for axis in range(3):
        moved = np.moveaxis(mask, axis, 0)
        mask = np.moveaxis(np.stack([ndimage.binary_fill_holes(plane) for plane in moved]), 0, axis)
    return mask


def nearest(bone, zooms):
    distance, index = ndimage.distance_transform_edt(~bone, sampling=zooms, return_indices=True)
    return distance, (index - np.indices(bone.shape)) * np.array(zooms)[:, None, None, None]


def export(study):
    target = OUT / "volumes" / f"{study}.npz"
    image = nib.as_closest_canonical(nib.load(str(SOURCE / "mrannotator" / f"{study}.nii.gz")))
    raw = np.asarray(image.dataobj)
    zooms = np.array(image.header.get_zooms()[:3], dtype=float)
    stride = np.array([max(1, int(round(WORKING_MM / z))) if z < 0.7 * WORKING_MM else 1 for z in zooms])
    raw = raw[:: stride[0], :: stride[1], :: stride[2]]
    affine = image.affine.copy()
    affine[:3, :3] = affine[:3, :3] * stride
    zooms = tuple(zooms * stride)
    bones = {name: closed(largest(raw == value), zooms) for name, value, _ in BONES}
    solid = np.zeros(raw.shape, bool)
    for mask in bones.values():
        solid |= mask
    fields = {name: nearest(bones[name], zooms) for name in ("femur", "tibia", "kneecap") if bones[name].any()}

    def between(first, second):
        if first not in fields or second not in fields:
            return np.zeros(raw.shape, bool)
        cosine = (fields[first][1] * fields[second][1]).sum(0) / np.maximum(fields[first][0] * fields[second][0], 1e-6)
        return closed(~solid & (cosine <= np.cos(np.radians(ANGLE))), zooms) & ~solid

    joint = between("femur", "tibia")
    behind = between("kneecap", "femur") & ~joint
    labels = np.zeros(raw.shape, np.uint8)
    labels[behind] = KNEECAP_JOINT
    labels[joint] = JOINT
    for name, _, value in BONES[::-1]:
        labels[bones[name]] = value
    corners = np.array([[i, j, k] for i in (0, raw.shape[0] - 1) for j in (0, raw.shape[1] - 1) for k in (0, raw.shape[2] - 1)])
    world = (corners @ affine[:3, :3].T + affine[:3, 3]) * FLIP
    origin = np.floor(world.min(0) / SPACING) * SPACING
    shape = np.ceil((world.max(0) - origin) / SPACING).astype(int) + 1
    grid = np.stack(np.meshgrid(*[origin[a] + SPACING * np.arange(shape[a]) for a in range(3)], indexing="ij"), -1)
    inverse = np.linalg.inv(affine)
    voxel = np.rint((grid * FLIP) @ inverse[:3, :3].T + inverse[:3, 3]).astype(int)
    inside = ((voxel >= 0) & (voxel < np.array(raw.shape))).all(-1)
    volume = np.full(tuple(shape), 255, np.uint8)
    volume[inside] = labels[tuple(voxel[inside].T)]
    np.savez_compressed(target, labels=volume, origin=origin.astype(np.float32), spacing=np.float32(SPACING))
    size = float(np.prod(zooms)) / 1000.0
    return {"study": study, "joint_ml": float(joint.sum() * size), "kneecap_joint_ml": float(behind.sum() * size), **{f"{name}_ml": float(mask.sum() * size) for name, mask in bones.items()}}


def main():
    (OUT / "volumes").mkdir(parents=True, exist_ok=True)
    shutil.copy(PREVIOUS / "knees.json", OUT / "knees.json")
    studies = sorted(path.stem for path in (PREVIOUS / "volumes").glob("*.npz"))
    with ProcessPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(export, studies, chunksize=4))
    (OUT / "classes.json").write_text(
        json.dumps(
            {
                "classes": ["other", "femur", "tibia", "kneecap", "fibula", "joint", "kneecap_joint"],
                "closing_mm": CLOSING_MM, "angle_degrees": ANGLE, "spacing_mm": SPACING, "working_mm": WORKING_MM, "knees": rows,
            },
            indent=1,
        )
    )
    print("knees", len(rows))
    for key in ("joint_ml", "kneecap_joint_ml", "femur_ml", "tibia_ml", "kneecap_ml", "fibula_ml"):
        v = np.array([r[key] for r in rows])
        print(f"  {key:17} median {np.median(v):6.1f} 5th percentile {np.percentile(v, 5):6.1f} 95th {np.percentile(v, 95):6.1f} smallest {v.min():6.1f} largest {v.max():6.1f} | empty in {(v < 0.5).sum()}")


if __name__ == "__main__":
    main()
