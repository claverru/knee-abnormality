from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path("outputs/analysis/octant-sketches-20261006")
QUADRANTS = {
    (1, 1): ("medial\nfront", (0.89, 0.47, 0.20)),
    (1, -1): ("medial\nback", (0.80, 0.25, 0.33)),
    (-1, 1): ("lateral\nfront", (0.30, 0.69, 0.60)),
    (-1, -1): ("lateral\nback", (0.22, 0.42, 0.69)),
}
SECTORS = (
    ("medial", (0.84, 0.19, 0.15)),
    ("antero-\nmedial", (0.96, 0.55, 0.20)),
    ("anterior", (0.95, 0.80, 0.25)),
    ("antero-\nlateral", (0.55, 0.78, 0.35)),
    ("lateral", (0.20, 0.63, 0.55)),
    ("postero-\nlateral", (0.20, 0.47, 0.75)),
    ("posterior", (0.42, 0.32, 0.70)),
    ("postero-\nmedial", (0.72, 0.30, 0.62)),
)
BANDS = (("tibia", -2.0, -1.05), ("joint line", -0.5, -0.1), ("femur", 0.45, 1.4))
CORE, RING, EDGE = 0.32, 0.86, 1.2


def box(low, high):
    (x0, y0, z0), (x1, y1, z1) = low, high
    c = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    return [[c[i] for i in face] for face in (
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)
    )]


def wedge(inner, outer, start, stop, low, high, steps=6):
    angles = np.radians(np.linspace(start, stop, steps + 1))
    faces = []
    for z in (low, high):
        faces.append(
            [(outer * np.cos(a), outer * np.sin(a), z) for a in angles]
            + [(inner * np.cos(a), inner * np.sin(a), z) for a in angles[::-1]]
        )
    for radius in (inner, outer):
        if radius == 0:
            continue
        for a, b in zip(angles[:-1], angles[1:]):
            faces.append([
                (radius * np.cos(a), radius * np.sin(a), low),
                (radius * np.cos(b), radius * np.sin(b), low),
                (radius * np.cos(b), radius * np.sin(b), high),
                (radius * np.cos(a), radius * np.sin(a), high),
            ])
    for a in (angles[0], angles[-1]):
        faces.append([
            (inner * np.cos(a), inner * np.sin(a), low),
            (outer * np.cos(a), outer * np.sin(a), low),
            (outer * np.cos(a), outer * np.sin(a), high),
            (inner * np.cos(a), inner * np.sin(a), high),
        ])
    return faces


def style(axis, limit, title):
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit, limit)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xticks([-limit * 0.6, limit * 0.6], ["lateral", "medial"])
    axis.set_yticks([-limit * 0.6, limit * 0.6], ["back", "front"])
    axis.set_zticks([-limit * 0.6, limit * 0.6], ["down", "up"])
    axis.tick_params(labelsize=11, pad=1)
    axis.view_init(elev=27, azim=-58)
    axis.set_title(title, fontsize=13, pad=2)


figure = plt.figure(figsize=(19, 9.6))
axis = figure.add_subplot(1, 2, 1, projection="3d")
gap = 0.07
for medial, front, up in product((1, -1), repeat=3):
    name, shade = QUADRANTS[(medial, front)]
    low = [gap if s > 0 else -1.0 - gap for s in (medial, front, up)]
    high = [value + 1.0 for value in low]
    tone = np.array(shade) * (1.0 if up > 0 else 0.72)
    axis.add_collection3d(
        Poly3DCollection(box(low, high), facecolors=tone, edgecolors="#222222", linewidths=0.8, alpha=0.8)
    )
    if front < 0 or medial > 0 or up > 0:
        axis.text(
            medial * 0.62, front * 0.62, up * 1.22, f"{name}\n{'up' if up > 0 else 'down'}",
            fontsize=9, ha="center", va="center", zorder=20,
            bbox={"facecolor": "white", "alpha": 0.75, "pad": 1.5, "edgecolor": "none"},
        )
style(axis, 1.15, "Octants: three flat cuts through the centre\n8 regions")

axis = figure.add_subplot(1, 2, 2, projection="3d")
for band, low, high in BANDS:
    for number, (name, shade) in enumerate(SECTORS):
        start, stop = 45.0 * number - 21.0, 45.0 * number + 21.0
        axis.add_collection3d(
            Poly3DCollection(
                wedge(CORE + 0.05, RING, start, stop, low, high),
                facecolors=shade, edgecolors="#222222", linewidths=0.5, alpha=0.9,
            )
        )
        axis.add_collection3d(
            Poly3DCollection(
                wedge(RING + 0.05, EDGE, start, stop, low, high),
                facecolors=np.array(shade) * 0.7, edgecolors="#222222", linewidths=0.4, alpha=0.3,
            )
        )
    axis.add_collection3d(
        Poly3DCollection(
            wedge(0.0, CORE, 0.0, 360.0, low, high, steps=24),
            facecolors=(0.93, 0.93, 0.93), edgecolors="#444444", linewidths=0.4, alpha=0.95,
        )
    )
    axis.text(
        1.75, 1.1, (low + high) / 2, band, fontsize=12, ha="left", va="center", fontweight="bold"
    )
for number, (name, _) in enumerate(SECTORS):
    angle = np.radians(45.0 * number)
    axis.text(
        1.7 * np.cos(angle), 1.7 * np.sin(angle), 1.75, name,
        fontsize=9.5, ha="center", va="center",
        bbox={"facecolor": "white", "alpha": 0.75, "pad": 1.5, "edgecolor": "none"},
    )
axis.text(0, 0, 1.6, "core", fontsize=10, ha="center", fontweight="bold")
style(
    axis, 1.95,
    "Cylindrical regions around the leg axis\n8 sectors x 3 height bands x (inner ring, periphery) + core",
)
figure.suptitle(
    "The same knee space divided two ways (schematic, not to scale). "
    "Right: solid wedges are the inner ring, the faint outer wedges the periphery, the pale column the core.",
    fontsize=12.5,
)
figure.tight_layout(rect=(0, 0, 1, 0.95))
figure.savefig(OUT / "octants-vs-cylinder-3d.png", dpi=95)
print(OUT / "octants-vs-cylinder-3d.png")
