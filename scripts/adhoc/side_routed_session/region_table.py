from pathlib import Path

import numpy as np

ROOT = Path("outputs/analysis/side-routing-attention-shape-20261006")
data = np.load(ROOT / "attention.npz")
density = np.load(ROOT / "sections-spots.npz")["density"]
names = [str(name) for name in data["names"]]
EDGE, CELL = 54.0, 6.0
centres = np.arange(-EDGE + CELL / 2, EDGE, CELL)
grid = np.stack(np.meshgrid(centres, centres, centres, indexing="ij"), -1).reshape(-1, 3)
CORE, RING, LOW, HIGH = 15.0, 30.0, -12.0, 18.0
SECTORS = ("AM", "A", "AL", "L", "PL", "P", "PM", "M")
MIRROR = {"AM": "AL", "AL": "AM", "L": "M", "M": "L", "PL": "PM", "PM": "PL", "A": "A", "P": "P"}
angle = np.degrees(np.arctan2(grid[:, 1], grid[:, 0]))
radius = np.hypot(grid[:, 0], grid[:, 1])
sector = np.floor(((angle - 22.5) % 360.0) / 45.0).astype(int)
band = np.searchsorted(np.array([LOW, HIGH]), grid[:, 2])
ring = np.where(radius < CORE, 0, np.where(radius < RING, 1, 2))


def shares(name, mirrored=False):
    values = density[names.index(name)]
    table = np.zeros((3, 8, 3))
    np.add.at(table, (ring, sector, band), values)
    if mirrored:
        order = [SECTORS.index(MIRROR[s]) for s in SECTORS]
        table = table[:, order]
    return table / table.sum()


def show(title, table):
    core = table[0].sum(0)
    print(f"\n{title}")
    print(f"  core by height (tibia, joint, upper): {(100 * core).round(0)}")
    for which, label in ((1, "inner"), (2, "outer")):
        for level, height in enumerate(("tibia", "joint", "upper")):
            row = table[which, :, level]
            if row.sum() > 0.02:
                print(f"  {label} ring, {height:5}: " + "  ".join(f"{s} {100 * v:4.1f}" for s, v in zip(SECTORS, row) if v >= 0.015))


GROUPS = {
    "ACL": [("ACL", False), ("report:acl_status", False)],
    "MCL": [("MCL", False), ("report:mcl_status", False)],
    "Meniscus (medial as is, lateral mirrored)": [("Medial Meniscus", False), ("Lateral Meniscus", True), ("report:medial_meniscus_status", False), ("report:lateral_meniscus_status", True), ("report:medial_meniscus_tear_location", False), ("report:lateral_meniscus_tear_location", True)],
    "Femorotibial OA (medial as is, lateral mirrored)": [("Medial OA", False), ("Lateral OA", True), ("report:medial_cartilage_depth", False), ("report:lateral_cartilage_depth", True)],
    "PF OA": [("PF OA", False), ("report:pf_cartilage_depth", False)],
    "Effusion": [("Effusion", False), ("report:effusion_size", False)],
    "Synovitis": [("Synovitis", False), ("report:synovitis_status", False)],
    "Baker's": [("Baker's", False), ("report:baker_size", False)],
    "Contusion": [("Contusion", False), ("report:contusion_status", False)],
    "Fracture": [("Fracture", False), ("report:fracture_status", False)],
}
for group, members in GROUPS.items():
    for name, mirrored in members:
        show(f"{group} <- {name}{' (mirrored)' if mirrored else ''}", shares(name, mirrored))
