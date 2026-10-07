from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path("outputs/analysis/octant-sketches-20261006")
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


def medial_side(key):
    return key != "core" and key[0] > 0


def lateral_side(key):
    return key != "core" and key[0] < 0


def kneecap(key, band):
    return key != "core" and key[1] > 0 and band == "femur"


TARGETS = (
    ("ACL", "the core", "#5aa469", lambda key, band: key == "core"),
    ("MCL", "the medial quadrants", "#8e5ea2", lambda key, band: medial_side(key)),
    ("Medial Meniscus", "medial side, joint-line band", "#d1495b", lambda key, band: medial_side(key) and band == "joint line"),
    ("Lateral Meniscus", "lateral side, joint-line band", "#3a7ca5", lambda key, band: lateral_side(key) and band == "joint line"),
    ("Medial OA", "medial side, without the kneecap part", "#e07a5f", lambda key, band: medial_side(key) and not kneecap(key, band)),
    ("Lateral OA", "lateral side, without the kneecap part", "#5b8fb9", lambda key, band: lateral_side(key) and not kneecap(key, band)),
    ("PF OA", "front quadrants, femur band", "#f2c14e", kneecap),
    ("Effusion", "front quadrants, femur band", "#e9a03b", kneecap),
    ("Baker's", "the medial-back quadrant", "#2a9d8f", lambda key, band: key == (1, -1)),
    ("Synovitis", "everything (not routed)", "#9aa0a6", lambda key, band: True),
    ("Contusion", "everything (not routed)", "#9aa0a6", lambda key, band: True),
    ("Fracture", "everything (not routed)", "#9aa0a6", lambda key, band: True),
)
figure = plt.figure(figsize=(22, 17))
for number, (name, reads, colour, rule) in enumerate(TARGETS, 1):
    count = sum(rule(key, band) for band, _, _ in BANDS for key in (*QUADRANTS, "core"))
    draw(
        figure.add_subplot(3, 4, number, projection="3d"),
        lambda key, band, rule=rule, colour=colour: colour if rule(key, band) else None,
        f"{name}\n{reads}\n{count} of 15 regions",
    )
figure.suptitle(
    "What each of the twelve targets would read: four quadrants plus a core, at three heights, around one centre at the joint line.\n"
    "Schematic, not to scale; the height bands are drawn apart. Coloured regions are read, grey ones are not.",
    fontsize=14,
)
figure.tight_layout(rect=(0, 0, 1, 0.95))
figure.savefig(OUT / "quadrants-core-per-target-3d.png", dpi=80)
print(OUT / "quadrants-core-per-target-3d.png")
