from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path("outputs/analysis/s/side-routing-attention-shape-20261006")
data = np.load(ROOT / "attention.npz")
density = np.load(ROOT / "sections-geometric.npz")["density"]
names = [str(name) for name in data["names"]]
labels = [str(name) for name in data["labels"]]
EDGE, CELL = 54.0, 6.0
centres = np.arange(-EDGE + CELL / 2, EDGE, CELL)
grid = np.stack(np.meshgrid(centres, centres, centres, indexing="ij"), -1).reshape(-1, 3)
CORE, RING, LOW, HIGH = 12.0, 30.0, -24.0, 12.0
SECTORS = ("anteromedial", "anterior", "anterolateral", "lateral", "posterolateral", "posterior", "posteromedial", "medial")
BANDS = ("tibia", "joint zone", "upper")
angle = np.degrees(np.arctan2(grid[:, 1], grid[:, 0]))
radius = np.hypot(grid[:, 0], grid[:, 1])
sector = np.floor(((angle - 22.5) % 360.0) / 45.0).astype(int)
band = np.searchsorted(np.array([LOW, HIGH]), grid[:, 2])
ring = np.where(radius < CORE, 0, np.where(radius < RING, 1, 2))
section = np.where(ring == 0, band, 3 + ((ring - 1) * 8 + sector) * 3 + band)
COUNT = 3 + 2 * 8 * 3
COLOURS = {
    "ACL": "#5aa469", "MCL": "#8e5ea2", "Medial Meniscus": "#d1495b", "Lateral Meniscus": "#3a7ca5",
    "Medial OA": "#e07a5f", "Lateral OA": "#5b8fb9", "PF OA": "#f2c14e", "Effusion": "#e9a03b",
    "Synovitis": "#c77dba", "Baker's": "#2a9d8f", "Contusion": "#7a8b99", "Fracture": "#7a8b99",
}
LEVELS = ((-2.0, -1.05), (-0.55, 0.35), (0.85, 1.8))
RADII = (0.36, 0.9, 1.5)


def wedge(inner, outer, start, stop, low, high, steps=5):
    angles = np.radians(np.linspace(start, stop, steps + 1))
    faces = []
    for z in (low, high):
        faces.append(
            [(outer * np.cos(a), outer * np.sin(a), z) for a in angles]
            + [(inner * np.cos(a), inner * np.sin(a), z) for a in angles[::-1]]
        )
    for value in (inner, outer):
        if value == 0:
            continue
        for a, b in zip(angles[:-1], angles[1:]):
            faces.append([
                (value * np.cos(a), value * np.sin(a), low), (value * np.cos(b), value * np.sin(b), low),
                (value * np.cos(b), value * np.sin(b), high), (value * np.cos(a), value * np.sin(a), high),
            ])
    for a in (angles[0], angles[-1]):
        faces.append([
            (inner * np.cos(a), inner * np.sin(a), low), (outer * np.cos(a), outer * np.sin(a), low),
            (outer * np.cos(a), outer * np.sin(a), high), (inner * np.cos(a), inner * np.sin(a), high),
        ])
    return faces


def shape(index):
    if index < 3:
        low, high = LEVELS[index]
        return wedge(0.0, RADII[0] - 0.03, 0.0, 360.0, low, high, steps=20)
    rest = index - 3
    level, cell = rest % 3, rest // 3
    which_ring, which_sector = cell // 8, cell % 8
    low, high = LEVELS[level]
    start = 22.5 + 45.0 * which_sector
    return wedge(RADII[which_ring] + 0.03, RADII[which_ring + 1], start + 2.0, start + 43.0, low, high)


def name_of(index):
    if index < 3:
        return f"core, {BANDS[index]}"
    rest = index - 3
    return f"{SECTORS[(rest // 3) % 8]}, {'inner' if rest // 24 == 0 else 'outer'} ring, {BANDS[rest % 3]}"


figure = plt.figure(figsize=(22, 18))
for number, label in enumerate(labels, 1):
    values = density[names.index(label)]
    share = np.bincount(section, weights=values, minlength=COUNT)
    share = share / share.sum()
    chosen = np.flatnonzero(share >= 0.05)
    axis = figure.add_subplot(3, 4, number, projection="3d")
    for index in range(COUNT):
        used = index in chosen
        axis.add_collection3d(
            Poly3DCollection(
                shape(index),
                facecolors=COLOURS[label] if used else "#c9c9c9",
                edgecolors="#222222" if used else "#aaaaaa",
                linewidths=0.6 if used else 0.2,
                alpha=min(0.95, 0.45 + 1.6 * share[index]) if used else 0.08,
            )
        )
    limit = 1.9
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit, limit)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xticks([-1.0, 1.0], ["lateral", "medial"])
    axis.set_yticks([-1.0, 1.0], ["back", "front"])
    axis.set_zticks([(low + high) / 2 for low, high in LEVELS], list(BANDS))
    axis.tick_params(labelsize=8, pad=0)
    axis.view_init(elev=30, azim=-58)
    order = chosen[np.argsort(-share[chosen])]
    listing = "\n".join(f"{100 * share[index]:.0f}% {name_of(index)}" for index in order[:4])
    axis.set_title(
        f"{label}: {len(chosen)} of {COUNT} sections hold {100 * share[chosen].sum():.0f}%\n{listing}",
        fontsize=8.5,
        pad=0,
    )
    print(label, len(chosen), round(float(share[chosen].sum()), 2), [(name_of(i), round(float(share[i]), 2)) for i in order])
figure.suptitle(
    "The division read from the heads' attention: eight sectors centred on the axes, a core and an outer ring, at three heights (51 sections).\n"
    "Per target, the sections that hold at least 5% of its ten most attended tokens (run 11, positive validation knees); stronger colour = larger share. "
    "Schematic; the heights are drawn apart.",
    fontsize=13,
)
figure.tight_layout(rect=(0, 0, 1, 0.95))
figure.savefig(ROOT / "rings-per-target-3d.png", dpi=80)
print(ROOT / "rings-per-target-3d.png")
