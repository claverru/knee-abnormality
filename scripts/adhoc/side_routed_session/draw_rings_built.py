from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path("outputs/analysis/s/side-routing-attention-shape-20261006")
SECTORS = ("AM", "A", "AL", "L", "PL", "P", "PM", "M")
BANDS = ("tibia", "joint", "upper")
LEVELS = ((-2.0, -1.05), (-0.55, 0.35), (0.85, 1.8))
RADII = (0.4, 0.95, 1.5)
MEDIAL, LATERAL, FRONT = ("AM", "M", "PM"), ("AL", "L", "PL"), ("AM", "A", "AL")
ALL = set(BANDS)


def table(sectors=(), rings=(), bands=(), core=(), core_bands=()):
    chosen = {("ring", ring, sector, band) for ring in rings for sector in sectors for band in bands}
    return chosen | {("core", half, band) for half in core for band in core_bands}


REGIONS = {
    "ACL": ("#5aa469", "core; inner ring of the front, back and lateral sectors\njoint and upper", table(("A", "P", *LATERAL), ("inner",), ("joint", "upper"), ("medial", "lateral"), ("joint", "upper"))),
    "MCL": ("#8e5ea2", "medial and anteromedial sectors, both rings\nall heights", table(("AM", "M"), ("inner", "outer"), ALL)),
    "Medial Meniscus": ("#d1495b", "the three medial sectors, both rings, its half of the core\ntibia and joint", table(MEDIAL, ("inner", "outer"), ("tibia", "joint"), ("medial",), ("tibia", "joint"))),
    "Lateral Meniscus": ("#3a7ca5", "the three lateral sectors, both rings, its half of the core\ntibia and joint", table(LATERAL, ("inner", "outer"), ("tibia", "joint"), ("lateral",), ("tibia", "joint"))),
    "Medial OA": ("#e07a5f", "same region as Medial Meniscus", table(MEDIAL, ("inner", "outer"), ("tibia", "joint"), ("medial",), ("tibia", "joint"))),
    "Lateral OA": ("#5b8fb9", "same region as Lateral Meniscus", table(LATERAL, ("inner", "outer"), ("tibia", "joint"), ("lateral",), ("tibia", "joint"))),
    "PF OA": ("#f2c14e", "the three front sectors, both rings\njoint and upper", table(FRONT, ("inner", "outer"), ("joint", "upper"))),
    "Effusion": ("#e9a03b", "same region as PF OA", table(FRONT, ("inner", "outer"), ("joint", "upper"))),
    "Synovitis": ("#c77dba", "the three front sectors, both rings, and the core\njoint and upper", table(FRONT, ("inner", "outer"), ("joint", "upper"), ("medial", "lateral"), ("joint", "upper"))),
    "Baker's": ("#2a9d8f", "posterior, posteromedial and medial sectors, both rings\nall heights", table(("P", "PM", "M"), ("inner", "outer"), ALL)),
    "Contusion": ("#7a8b99", "everything (not routed)", None),
    "Fracture": ("#7a8b99", "everything (not routed)", None),
}


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


SECTIONS = []
for level, band in enumerate(BANDS):
    low, high = LEVELS[level]
    SECTIONS.append((("core", "medial", band), wedge(0.0, RADII[0] - 0.03, -88.0, 88.0, low, high, steps=10)))
    SECTIONS.append((("core", "lateral", band), wedge(0.0, RADII[0] - 0.03, 92.0, 268.0, low, high, steps=10)))
    for which, ring in enumerate(("inner", "outer")):
        for number, sector in enumerate(SECTORS):
            start = 22.5 + 45.0 * number
            SECTIONS.append(
                (("ring", ring, sector, band), wedge(RADII[which] + 0.03, RADII[which + 1], start + 2.0, start + 43.0, low, high))
            )
figure = plt.figure(figsize=(22, 18))
for number, (label, (colour, text, chosen)) in enumerate(REGIONS.items(), 1):
    axis = figure.add_subplot(3, 4, number, projection="3d")
    used = 0
    for key, faces in SECTIONS:
        inside = chosen is None or key in chosen
        used += inside
        axis.add_collection3d(
            Poly3DCollection(
                faces,
                facecolors=colour if inside else "#c9c9c9",
                edgecolors="#222222" if inside else "#aaaaaa",
                linewidths=0.6 if inside else 0.2,
                alpha=0.9 if inside else 0.08,
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
    axis.set_title(f"{label}: {used} of {len(SECTIONS)} sections\n{text}", fontsize=9, pad=0)
figure.suptitle(
    "The regions as built in run 24: eight sectors centred on the axes, a core (drawn in two halves) and an outer ring, at three heights.\n"
    "Coloured sections are read by the target, grey ones are not. Schematic; the heights are drawn apart.",
    fontsize=13,
)
figure.tight_layout(rect=(0, 0, 1, 0.95))
figure.savefig(OUT / "rings-as-built-per-target-3d.png", dpi=80)
print(OUT / "rings-as-built-per-target-3d.png")
