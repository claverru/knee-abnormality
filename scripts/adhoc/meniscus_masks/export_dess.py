import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
CACHE = Path.home() / ".cache/knee-segmenters"
SPACING = 1.0
OUTSIDE = 255
CLASSES = (
    "neither",
    "medial_meniscus",
    "lateral_meniscus",
    "medial_tibial_cartilage",
    "lateral_tibial_cartilage",
    "femoral_cartilage",
    "patellar_cartilage",
)
MENISCI = {"medial": 5, "lateral": 6}
CARTILAGE = {
    "medial_tibial_cartilage": 3,
    "lateral_tibial_cartilage": 4,
    "femoral_cartilage": 2,
    "patellar_cartilage": 1,
}
FEMUR, TIBIA = 7, 8
FLIP = np.array([-1.0, -1.0, 1.0])
VOLUME_ML = (0.8, 7.0)
MAIN_PIECE = 0.8
TIBIA_MM = 8.0
FEMUR_MM = 10.0
OFF_CENTRE_MM = 20.0


def largest(mask):
    pieces, count = ndimage.label(mask)
    if count < 2:
        return mask
    sizes = np.bincount(pieces.ravel())[1:]
    return pieces == 1 + sizes.argmax()


def export(job):
    source, out, case = Path(job[0]), Path(job[1]), job[2]
    study = case["study"]
    record = {"study": study, "fold": case["fold"], "series": case["series"]}
    path = source / "nnunet_knee" / f"{study}.nii.gz"
    if not path.exists():
        return {**record, "accepted": False, "problems": ["no mask"]}
    image = nib.as_closest_canonical(nib.load(str(path)))
    raw = np.asarray(image.dataobj).astype(np.uint8)
    affine = image.affine
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    voxel_ml = float(np.prod(zooms)) / 1000.0
    problems = []
    distances = {}
    for name, bone in (("tibia", TIBIA), ("femur", FEMUR)):
        mask = raw == bone
        if not mask.any():
            problems.append(f"no {name}")
            continue
        distances[name] = ndimage.distance_transform_edt(~mask, sampling=zooms)
    kept = np.zeros_like(raw)
    centres = {}
    for value, (side, old) in enumerate(MENISCI.items(), start=1):
        mask = raw == old
        total = int(mask.sum())
        record[f"{side}_ml"] = total * voxel_ml
        if not total:
            problems.append(f"no {side} meniscus")
            continue
        piece = largest(mask)
        share = float(piece.sum() / total)
        record[f"{side}_main_piece"] = share
        kept[piece] = value
        centre = np.argwhere(piece).mean(0) @ affine[:3, :3].T + affine[:3, 3]
        centres[side] = centre * FLIP
        if not VOLUME_ML[0] <= total * voxel_ml <= VOLUME_ML[1]:
            problems.append(f"{side} volume")
        if share < MAIN_PIECE:
            problems.append(f"{side} in pieces")
        for name, limit in (("tibia", TIBIA_MM), ("femur", FEMUR_MM)):
            if name in distances:
                gap = float(np.median(distances[name][piece]))
                record[f"{side}_to_{name}_mm"] = gap
                if gap > limit:
                    problems.append(f"{side} far from the {name}")
    for name, old in CARTILAGE.items():
        mask = raw == old
        record[f"{name}_ml"] = int(mask.sum()) * voxel_ml
        kept[mask] = CLASSES.index(name)
    if len(centres) == 2:
        medial, lateral = centres["medial"][0], centres["lateral"][0]
        middle = (medial + lateral) / 2
        record["knee_left_mm"] = float(middle)
        record["menisci_apart_mm"] = float(abs(medial - lateral))
        if abs(middle) < OFF_CENTRE_MM:
            problems.append("knee at the centre, side not checked")
        elif abs(medial) >= abs(lateral):
            problems.append("sides swapped")
    corners = np.array(
        [
            [i, j, k]
            for i in (0, raw.shape[0] - 1)
            for j in (0, raw.shape[1] - 1)
            for k in (0, raw.shape[2] - 1)
        ]
    )
    world = (corners @ affine[:3, :3].T + affine[:3, 3]) * FLIP
    origin = np.floor(world.min(0) / SPACING) * SPACING
    shape = np.ceil((world.max(0) - origin) / SPACING).astype(int) + 1
    grid = np.stack(
        np.meshgrid(
            *[origin[axis] + SPACING * np.arange(shape[axis]) for axis in range(3)],
            indexing="ij",
        ),
        -1,
    )
    inverse = np.linalg.inv(affine)
    voxel = np.rint((grid * FLIP) @ inverse[:3, :3].T + inverse[:3, 3]).astype(int)
    inside = ((voxel >= 0) & (voxel < np.array(raw.shape))).all(-1)
    volume = np.full(tuple(shape), OUTSIDE, np.uint8)
    volume[inside] = kept[tuple(voxel[inside].T)]
    record.update(accepted=not problems, problems=problems)
    if not problems:
        np.savez_compressed(
            out / "volumes" / f"{study}.npz",
            labels=volume,
            origin=origin.astype(np.float32),
            spacing=np.float32(SPACING),
        )
    return record


def main():
    source = CACHE / sys.argv[1]
    out = ROOT / "data/labels/prepared" / sys.argv[2]
    workers = int(sys.argv[3])
    (out / "volumes").mkdir(parents=True, exist_ok=True)
    for stale in (out / "volumes").glob("*.npz"):
        stale.unlink()
    cases = [
        case
        for case in json.loads((source / "cases.json").read_text())
        if "size" in case
    ]
    jobs = [(str(source), str(out), case) for case in cases]
    with ProcessPoolExecutor(workers) as pool:
        knees = list(pool.map(export, jobs, chunksize=2))
    (out / "knees.json").write_text(json.dumps(knees, indent=1))
    (out / "classes.json").write_text(
        json.dumps(
            {
                "classes": CLASSES,
                "spacing_mm": SPACING,
                "segmenter": "aagatti/nnunet_knee 3d_fullres fold 1",
                "volume_ml": VOLUME_ML,
                "main_piece": MAIN_PIECE,
                "tibia_mm": TIBIA_MM,
                "femur_mm": FEMUR_MM,
                "off_centre_mm": OFF_CENTRE_MM,
            },
            indent=1,
        )
    )
    accepted = [knee for knee in knees if knee["accepted"]]
    print("knees", len(knees), "accepted", len(accepted))
    reasons = {}
    for knee in knees:
        for problem in knee["problems"]:
            reasons[problem] = reasons.get(problem, 0) + 1
    print("problems", reasons)
    folds = {}
    for knee in accepted:
        folds[knee["fold"]] = folds.get(knee["fold"], 0) + 1
    print("accepted per fold", dict(sorted(folds.items())))
    for key in (
        "medial_ml",
        "lateral_ml",
        "medial_main_piece",
        "lateral_main_piece",
        "medial_to_tibia_mm",
        "lateral_to_tibia_mm",
        "medial_to_femur_mm",
        "lateral_to_femur_mm",
        "knee_left_mm",
        "menisci_apart_mm",
        *(f"{name}_ml" for name in CARTILAGE),
    ):
        values = np.array([knee[key] for knee in knees if key in knee])
        if key == "knee_left_mm":
            values = np.abs(values)
        low, mid, high = np.percentile(values, [1, 50, 99])
        print(
            f"  {key:20} 1% {low:6.2f} median {mid:6.2f} 99% {high:6.2f}"
            f" min {values.min():6.2f} max {values.max():6.2f}"
        )


if __name__ == "__main__":
    main()
