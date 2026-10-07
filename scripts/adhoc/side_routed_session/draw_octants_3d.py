from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path("outputs/analysis/octant-sketches-20261006")
OUT.mkdir(parents=True, exist_ok=True)
GAP = 0.14
COLOURS = {
    "Medial Meniscus": "#d1495b",
    "Lateral Meniscus": "#3a7ca5",
    "Medial OA": "#e07a5f",
    "Lateral OA": "#5b8fb9",
    "PF OA": "#f2c14e",
}


def meniscus(medial, front, up):
    return "Medial Meniscus" if medial > 0 else "Lateral Meniscus"


def oa_now(medial, front, up):
    if front > 0 and up > 0:
        return "PF OA"
    return "Medial OA" if medial > 0 else "Lateral OA"


def oa_run5(medial, front, up):
    if front > 0:
        return "PF OA"
    return "Medial OA" if medial > 0 else "Lateral OA"


def cube(origin, size=1.0):
    x, y, z = origin
    s = size
    corners = [
        (x, y, z),
        (x + s, y, z),
        (x + s, y + s, z),
        (x, y + s, z),
        (x, y, z + s),
        (x + s, y, z + s),
        (x + s, y + s, z + s),
        (x, y + s, z + s),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 3, 7, 4),
    ]
    return [[corners[i] for i in face] for face in faces]


def draw(axis, rule, title):
    for medial, front, up in product((-1, 1), repeat=3):
        label = rule(medial, front, up)
        origin = (
            GAP / 2 if medial > 0 else -1 - GAP / 2,
            GAP / 2 if front > 0 else -1 - GAP / 2,
            GAP / 2 if up > 0 else -1 - GAP / 2,
        )
        axis.add_collection3d(
            Poly3DCollection(
                cube(origin),
                facecolors=COLOURS[label],
                edgecolors="#333333",
                linewidths=0.8,
                alpha=0.78,
            )
        )
    limit = 1 + GAP
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit, limit)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xticks([-0.6, 0.6], ["lateral", "medial"])
    axis.set_yticks([-0.6, 0.6], ["back", "front"])
    axis.set_zticks([-0.6, 0.6], ["down", "up"])
    axis.tick_params(labelsize=11, pad=2)
    axis.view_init(elev=20, azim=-52)
    axis.set_title(title, fontsize=13, pad=4)
    used = []
    for medial, front, up in product((-1, 1), repeat=3):
        label = rule(medial, front, up)
        if label not in used:
            used.append(label)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOURS[name], edgecolor="#333333")
        for name in used
    ]
    counts = {
        name: sum(rule(*octant) == name for octant in product((-1, 1), repeat=3))
        for name in used
    }
    axis.legend(
        handles,
        [f"{name}: {counts[name]} octants" for name in used],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        fontsize=10,
        frameon=False,
        ncol=1,
    )


figure = plt.figure(figsize=(17, 7.4))
panels = (
    (meniscus, "Meniscus: Medial and Lateral"),
    (oa_now, "OA in the adopted model:\nPF OA takes front-and-up"),
    (oa_run5, "OA in run 5 (single frame):\nPF OA takes the whole front"),
)
for number, (rule, title) in enumerate(panels, 1):
    draw(figure.add_subplot(1, 3, number, projection="3d"), rule, title)
figure.suptitle(
    "The eight octants around the knee centre and the routed label that reads each one\n"
    "(each octant is read by one meniscus label and one OA label; cuts are soft, about 5 mm)",
    fontsize=13,
)
figure.tight_layout(rect=(0, 0.02, 1, 0.93))
figure.savefig(OUT / "octants-3d.png", dpi=110)
print(OUT / "octants-3d.png")
