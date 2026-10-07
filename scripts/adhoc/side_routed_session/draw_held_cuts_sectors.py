from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path("outputs/analysis/side-routing-cut-forces-20261006")
CORE, RADIUS, LOW, CUT, HIGH, GAP = 0.4, 1.5, -1.9, 0.2, 1.8, 0.22
TILT = np.radians(15.0)
LABELS = {
    "ACL": ("#5aa469", ("core",), "the core, at all heights"),
    "MCL": ("#8e5ea2", ("medial",), "the medial sector, below the cut"),
    "Medial Meniscus": ("#d1495b", ("medial",), "the medial sector, below the cut"),
    "Lateral Meniscus": ("#3a7ca5", ("lateral",), "the lateral sector, below the cut"),
    "Medial OA": ("#e07a5f", ("medial",), "the medial sector, below the cut"),
    "Lateral OA": ("#5b8fb9", ("lateral",), "the lateral sector, below the cut"),
    "PF OA": ("#f2c14e", ("upper medial", "upper lateral"), "both sectors, above the cut"),
    "Effusion": ("#e9a03b", ("upper medial", "upper lateral"), "both sectors, above the cut"),
    "Synovitis": ("#c77dba", ("upper medial", "upper lateral"), "both sectors, above the cut"),
    "Baker's": ("#2a9d8f", ("medial",), "the medial sector, below the cut"),
    "Contusion": ("#7a8b99", None, "everything (not routed)"),
    "Fracture": ("#7a8b99", None, "everything (not routed)"),
}


def wedge(inner, outer, start, stop, low, high, steps=14):
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


def nod(faces):
    return [
        [(x, y * np.cos(TILT) + z * np.sin(TILT), z * np.cos(TILT) - y * np.sin(TILT)) for x, y, z in face]
        for face in faces
    ]


SECTIONS = {
    "core": wedge(0.0, CORE - 0.03, 0.0, 360.0, LOW, HIGH + GAP, steps=28),
    "medial": wedge(CORE + 0.03, RADIUS, -88.0, 88.0, LOW, CUT),
    "lateral": wedge(CORE + 0.03, RADIUS, 92.0, 268.0, LOW, CUT),
    "upper medial": wedge(CORE + 0.03, RADIUS, -88.0, 88.0, CUT + GAP, HIGH + GAP),
    "upper lateral": wedge(CORE + 0.03, RADIUS, 92.0, 268.0, CUT + GAP, HIGH + GAP),
}
figure = plt.figure(figsize=(22, 18))
for number, (label, (colour, read, text)) in enumerate(LABELS.items(), 1):
    axis = figure.add_subplot(3, 4, number, projection="3d")
    for name, faces in SECTIONS.items():
        inside = read is None or name in read
        axis.add_collection3d(
            Poly3DCollection(
                nod(faces),
                facecolors=colour if inside else "#c9c9c9",
                edgecolors="#222222" if inside else "#aaaaaa",
                linewidths=0.6 if inside else 0.2,
                alpha=0.9 if inside else 0.08,
            )
        )
    limit = 2.0
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit, limit)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xticks([-1.0, 1.0], ["lateral", "medial"])
    axis.set_yticks([-1.0, 1.0], ["back", "front"])
    axis.set_zticks([-0.9, 1.1], ["below", "above"])
    axis.tick_params(labelsize=8, pad=0)
    axis.view_init(elev=24, azim=-58)
    axis.set_title(f"{label}\n{text}", fontsize=10, pad=0)
figure.suptitle(
    "The same construction as the reference with only the held cuts: a core, two sectors (medial, lateral) and one height cut, in a frame tilted forward.\n"
    "Coloured sections are read by the target, grey ones are not. Schematic: the two heights are drawn apart; the tilt (15 degrees here), the cut's height and the core's radius are learned.",
    fontsize=13,
)
figure.tight_layout(rect=(0, 0, 1, 0.95))
figure.savefig(OUT / "held-cuts-sectors-per-target-3d.png", dpi=80)
print(OUT / "held-cuts-sectors-per-target-3d.png")
