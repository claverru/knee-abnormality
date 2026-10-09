from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path("outputs/analysis/o/octant-sketches-20261006")
QUADRANTS = {
    (1, 1): ("medial front", "#e2783a"),
    (1, -1): ("medial back", "#cc4055"),
    (-1, 1): ("lateral front", "#4db09a"),
    (-1, -1): ("lateral back", "#386bb0"),
}
BANDS = (("tibia", -1.75, -0.75), ("joint line", -0.28, 0.28), ("femur", 0.75, 1.75))
CORE, SIZE, GAP = 0.42, 1.25, 0.05
GREY = "#c9c9c9"


def prism(outline, low, high):
    faces = [[(x, y, z) for x, y in outline] for z in (low, high)]
    for (x0, y0), (x1, y1) in zip(outline, outline[1:] + outline[:1]):
        faces.append([(x0, y0, low), (x1, y1, low), (x1, y1, high), (x0, y0, high)])
    return faces


def quadrant(medial, front):
    start = np.arctan2(front, 0.0)
    stop = np.arctan2(0.0, medial)
    if abs(stop - start) > np.pi:
        stop += 2 * np.pi * np.sign(start - stop)
    arc = [
        (CORE * np.cos(a) + GAP * medial, CORE * np.sin(a) + GAP * front)
        for a in np.linspace(start, stop, 9)
    ]
    return [
        (medial * SIZE, GAP * front),
        (medial * SIZE, front * SIZE),
        (GAP * medial, front * SIZE),
        *arc,
    ]


def core():
    return [
        ((CORE - GAP) * np.cos(a), (CORE - GAP) * np.sin(a))
        for a in np.linspace(0, 2 * np.pi, 28, endpoint=False)
    ]


def draw(axis, colour_of, title):
    for band, low, high in BANDS:
        for key in (*QUADRANTS, "core"):
            colour = colour_of(key, band)
            outline = core() if key == "core" else quadrant(*key)
            axis.add_collection3d(
                Poly3DCollection(
                    prism(outline, low, high),
                    facecolors=colour or GREY,
                    edgecolors="#222222" if colour else "#999999",
                    linewidths=0.7 if colour else 0.3,
                    alpha=0.92 if colour else 0.13,
                )
            )
    limit = 1.75
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit, limit)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xticks([-0.9, 0.9], ["lateral", "medial"])
    axis.set_yticks([-0.9, 0.9], ["back", "front"])
    axis.set_zticks([(low + high) / 2 for _, low, high in BANDS], [name for name, _, _ in BANDS])
    axis.tick_params(labelsize=10, pad=1)
    axis.view_init(elev=26, azim=-58)
    axis.set_title(title, fontsize=12, pad=0)


def regions(key, band):
    return "#f1f1f1" if key == "core" else QUADRANTS[key][1]


def meniscus(key, band):
    if key == "core" or band != "joint line":
        return None
    return "#d1495b" if key[0] > 0 else "#3a7ca5"


def osteoarthritis(key, band):
    if key == "core":
        return None
    if band == "femur" and key[1] > 0:
        return "#f2c14e"
    return "#e07a5f" if key[0] > 0 else "#5b8fb9"


def cruciate(key, band):
    return "#5aa469" if key == "core" else None


def collateral(key, band):
    return "#8e5ea2" if key != "core" and key[0] > 0 else None


def baker(key, band):
    return "#2a9d8f" if key == (1, -1) else None


PANELS = (
    (regions, "The regions: four quadrants and a core,\nat three heights (15)"),
    (meniscus, "Medial Meniscus (red), Lateral Meniscus (blue):\ntheir side, joint-line band"),
    (osteoarthritis, "PF OA (yellow; Effusion reads the same),\nMedial OA (orange), Lateral OA (light blue)"),
    (cruciate, "ACL: the core"),
    (collateral, "MCL: the medial quadrants"),
    (baker, "Baker's: the medial-back quadrant"),
)
figure = plt.figure(figsize=(19, 12.5))
for number, (rule, title) in enumerate(PANELS, 1):
    draw(figure.add_subplot(2, 3, number, projection="3d"), rule, title)
figure.suptitle(
    "Proposed division around one centre placed at the joint line: four quadrants plus a core, at three heights. "
    "Schematic, not to scale; the bands are drawn apart.\n"
    "First panel: the regions. Other panels: the regions each label reads, with the rest in grey.",
    fontsize=13,
)
figure.tight_layout(rect=(0, 0, 1, 0.94))
figure.savefig(OUT / "quadrants-core-3d.png", dpi=90)
print(OUT / "quadrants-core-3d.png")
