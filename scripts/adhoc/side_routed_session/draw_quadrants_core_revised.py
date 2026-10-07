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


def core(medial):
    return [
        (
            (CORE - GAP) * np.cos(a) + 0.5 * GAP * medial,
            (CORE - GAP) * np.sin(a),
        )
        for a in np.linspace(-np.pi / 2, np.pi / 2, 15) + (0 if medial > 0 else np.pi)
    ]


KEYS = (*QUADRANTS, ("core", 1), ("core", -1))


def draw(axis, colour_of, title):
    for band, low, high in BANDS:
        for key in KEYS:
            colour = colour_of(key, band)
            outline = core(key[1]) if key[0] == "core" else quadrant(*key)
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


def is_core(key):
    return key[0] == "core"


def medial(key):
    return key[1] > 0 if is_core(key) else key[0] > 0


def front(key):
    return not is_core(key) and key[1] > 0


def back(key):
    return not is_core(key) and key[1] < 0


UPPER = ("femur", "joint line")
TARGETS = (
    ("ACL", "core and lateral quadrants,\nfemur and joint bands", "#5aa469",
     lambda key, band: (is_core(key) or not medial(key)) and band in UPPER),
    ("MCL", "medial quadrants and medial half of the core,\nall heights", "#8e5ea2",
     lambda key, band: medial(key)),
    ("Medial Meniscus", "medial side with its half of the core,\nfemur and joint bands", "#d1495b",
     lambda key, band: medial(key) and band in UPPER),
    ("Lateral Meniscus", "lateral side with its half of the core,\nfemur and joint bands", "#3a7ca5",
     lambda key, band: not medial(key) and band in UPPER),
    ("Medial OA", "medial side with its half of the core,\nall heights, without the kneecap part", "#e07a5f",
     lambda key, band: medial(key) and not (front(key) and band == "femur")),
    ("Lateral OA", "lateral side with its half of the core,\nall heights, without the kneecap part", "#5b8fb9",
     lambda key, band: not medial(key) and not (front(key) and band == "femur")),
    ("PF OA", "front quadrants and the core,\nfemur band", "#f2c14e",
     lambda key, band: (front(key) or is_core(key)) and band == "femur"),
    ("Effusion", "front quadrants and the core,\nfemur band", "#e9a03b",
     lambda key, band: (front(key) or is_core(key)) and band == "femur"),
    ("Baker's", "medial-back quadrant,\nfemur and joint bands", "#2a9d8f",
     lambda key, band: key == (1, -1) and band in UPPER),
    ("Synovitis", "everything\n(not routed)", "#9aa0a6", lambda key, band: True),
    ("Contusion", "everything\n(not routed)", "#9aa0a6", lambda key, band: True),
    ("Fracture", "everything\n(not routed)", "#9aa0a6", lambda key, band: True),
)
figure = plt.figure(figsize=(22, 17.5))
for number, (name, reads, colour, rule) in enumerate(TARGETS, 1):
    count = sum(rule(key, band) for band, _, _ in BANDS for key in KEYS)
    draw(
        figure.add_subplot(3, 4, number, projection="3d"),
        lambda key, band, rule=rule, colour=colour: colour if rule(key, band) else None,
        f"{name}: {count} of 18 regions\n{reads}",
    )
figure.suptitle(
    "Revised sections per target, after measuring where each label's attention falls (run 11, positive validation knees).\n"
    "Four quadrants plus a core split at the midline, at three heights: 18 regions around one centre at the joint line. "
    "Schematic, not to scale; coloured regions are read, grey ones are not.",
    fontsize=14,
)
figure.tight_layout(rect=(0, 0, 1, 0.95))
figure.savefig(OUT / "quadrants-core-per-target-revised-3d.png", dpi=80)
print(OUT / "quadrants-core-per-target-revised-3d.png")
