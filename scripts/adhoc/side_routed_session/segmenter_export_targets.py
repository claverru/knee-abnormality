import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, "/home/claudio/.cache/knee-segmenters")
from bone_geometry import knee

SOURCE = Path("/home/claudio/.cache/knee-segmenters") / sys.argv[1]
OUT = Path(__file__).resolve().parents[3] / "data/labels/prepared" / sys.argv[2]
SPACING = 2.5
CLASSES = {7: 1, 8: 2, 9: 3, 10: 4}
FLIP = np.array([-1.0, -1.0, 1.0])

(OUT / "volumes").mkdir(parents=True, exist_ok=True)
cases = {c["study"]: c for c in json.loads((SOURCE / "cases.json").read_text()) if "series" in c}
knees = []
for study, case in sorted(cases.items()):
    path = SOURCE / "mrannotator" / f"{study}.nii.gz"
    if not path.exists():
        continue
    image = nib.as_closest_canonical(nib.load(str(path)))
    raw = np.asarray(image.dataobj)
    affine = image.affine
    corners = np.array([[i, j, k] for i in (0, raw.shape[0] - 1) for j in (0, raw.shape[1] - 1) for k in (0, raw.shape[2] - 1)])
    world = (corners @ affine[:3, :3].T + affine[:3, 3]) * FLIP
    origin = np.floor(world.min(0) / SPACING) * SPACING
    shape = np.ceil((world.max(0) - origin) / SPACING).astype(int) + 1
    grid = np.stack(np.meshgrid(*[origin[a] + SPACING * np.arange(shape[a]) for a in range(3)], indexing="ij"), -1)
    inverse = np.linalg.inv(affine)
    voxel = np.rint((grid * FLIP) @ inverse[:3, :3].T + inverse[:3, 3]).astype(int)
    inside = ((voxel >= 0) & (voxel < np.array(raw.shape))).all(-1)
    volume = np.full(tuple(shape), 255, np.uint8)
    values = raw[tuple(voxel[inside].T)]
    mapped = np.zeros_like(values)
    for old, new in CLASSES.items():
        mapped[values == old] = new
    volume[inside] = mapped
    np.savez_compressed(OUT / "volumes" / f"{study}.npz", labels=volume, origin=origin.astype(np.float32), spacing=np.float32(SPACING))
    result = knee(path)
    record = {
        "study": study, "fold": case["fold"], "series": case["series"], "fat_suppressed": case["fat_suppressed"], "problem": result.get("problem"),
        **{name: float((raw == old).sum() * abs(np.linalg.det(affine[:3, :3])) / 1000.0) for old, name in ((7, "femur_ml"), (8, "tibia_ml"), (9, "kneecap_ml"), (10, "fibula_ml"))},
    }
    if "plateau" in result:
        a, b, _ = result["plateau"]["plane"]
        normal = np.array([a, b, 1.0]) / np.linalg.norm([a, b, 1.0])
        centre = result["plateau"]["points"].mean(0) * FLIP
        record.update({f"plateau_centre_{axis}": float(centre[i]) for i, axis in enumerate("xyz")})
        record.update({f"plateau_normal_{axis}": float(normal[i]) for i, axis in enumerate("xyz")})
        if result["femur_axis"]:
            shaft = np.array([np.tan(np.radians(result["femur_axis"]["coronal"])), np.tan(np.radians(result["femur_axis"]["sagittal"])), 1.0])
            shaft = shaft / np.linalg.norm(shaft) * FLIP
            record.update({f"femur_axis_{axis}": float(shaft[i]) for i, axis in enumerate("xyz")})
            record["femur_lever_mm"] = result["femur_axis"]["lever"]
    knees.append(record)
(OUT / "knees.json").write_text(json.dumps(knees, indent=1))
usable = [k for k in knees if k["problem"] is None]
print("knees", len(knees), "usable", len(usable), "with a femur axis", sum("femur_axis_x" in k for k in knees), "| problems", {k["study"][-6:]: k["problem"] for k in knees if k["problem"]})
for key in ("femur_ml", "tibia_ml", "kneecap_ml", "fibula_ml"):
    v = np.array([k[key] for k in knees])
    print(f"  {key:11} median {np.median(v):6.1f} 1st percentile {np.percentile(v, 1):6.1f} smallest {v.min():6.1f}")
