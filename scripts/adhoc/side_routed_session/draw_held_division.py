import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path("outputs/analysis/side-routing-cut-forces-20261006")
division = json.loads((OUT / "symmetric-division-0.20.json").read_text())
ANGLES = division["angles"]
SECTORS = ("front-medial", "front-lateral", "back-lateral", "back-medial")
BANDS = ("joint", "kneecap", "pouch")
LEVELS = ((-1.9, -0.75), (-0.45, 0.35), (0.65, 1.8))
SCALE = 1.5 / 45.0
CORE = division["radii"][0] * SCALE
COLOURS = {
    "ACL": "#5aa469", "MCL": "#8e5ea2", "Medial Meniscus": "#d1495b", "Lateral Meniscus": "#3a7ca5",
    "Medial OA": "#e07a5f", "Lateral OA": "#5b8fb9", "PF OA": "#f2c14e", "Effusion": "#e9a03b",
    "Synovitis": "#c77dba", "Baker's": "#2a9d8f", "Contusion": "#7a8b99", "Fracture": "#7a8b99",
}


def wedge(inner, outer, start, stop, low, high):
    steps = max(4, int((stop - start) / 10))
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
for band, (low, high) in enumerate(LEVELS):
    SECTIONS.append((("core", band), wedge(0.0, CORE - 0.03, 0.0, 360.0, low, high)))
    for number in range(len(ANGLES)):
        start = ANGLES[number]
        stop = ANGLES[(number + 1) % len(ANGLES)] + (360.0 if number == len(ANGLES) - 1 else 0.0)
        SECTIONS.append((("ring", band, number), wedge(CORE + 0.03, 1.5, start + 2.0, stop - 2.0, low, high)))
figure = plt.figure(figsize=(22, 18))
for number, (label, colour) in enumerate(COLOURS.items(), 1):
    axis = figure.add_subplot(3, 4, number, projection="3d")
    reads = division["reads"].get(label)
    used = 0
    for key, faces in SECTIONS:
        if reads is None:
            inside = True
        elif key[0] == "core":
            inside = 0 in reads["radii"] and key[1] in reads["heights"]
        else:
            inside = 1 in reads["radii"] and key[1] in reads["heights"] and key[2] in reads["sectors"]
        used += inside
        axis.add_collection3d(
            Poly3DCollection(
                faces,
                facecolors=colour if inside else "#c9c9c9",
                edgecolors="#222222" if inside else "#aaaaaa",
                linewidths=0.6 if inside else 0.2,
                alpha=0.9 if inside else 0.07,
            )
        )
    if reads is None:
        text = "everything (not routed)"
    else:
        parts = []
        if 0 in reads["radii"]:
            parts.append("core")
        if 1 in reads["radii"]:
            parts.append(", ".join(SECTORS[i] for i in reads["sectors"]))
        text = " + ".join(parts) + "\n" + ", ".join(BANDS[i] for i in reads["heights"])
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
    "The finest division in which labels hold every cut from both sides, medial and lateral mirrored: three heights (cuts at +9 and +27 mm), a core of 21 mm and four sectors.\n"
    "Coloured sections are read by the target (at least 20% of its free attention), grey ones are not. Schematic; the heights are drawn apart.",
    fontsize=13,
)
figure.tight_layout(rect=(0, 0, 1, 0.95))
figure.savefig(OUT / "symmetric-division-per-target-3d.png", dpi=80)
print(OUT / "symmetric-division-per-target-3d.png")
