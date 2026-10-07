from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

LABEL, REPORT, BUILD, EXTRACT = "#1a1a1a", "#1f6fb5", "#2a8f3c", "#d9730d"
CUTS = [
    (
        "Medial | lateral",
        "axial",
        "side",
        "the lateral side",
        "the medial side",
        [("Lateral Meniscus", LABEL), ("Lateral OA", LABEL), ("tear place, lateral ×3", REPORT), ("lateral facet", REPORT)],
        [("Medial Meniscus", LABEL), ("Medial OA", LABEL), ("MCL", LABEL), ("Baker's", LABEL), ("tear place, medial ×3", REPORT), ("medial facet", REPORT)],
    ),
    (
        "Front | back",
        "axial",
        "front",
        "the back",
        "the front",
        [("Baker's", LABEL), ("posterior horn ×2", REPORT)],
        [("PF OA", LABEL), ("Effusion", LABEL), ("Synovitis", LABEL), ("anterior horn ×2", REPORT), ("facets ×2", REPORT)],
    ),
    (
        "Core radius",
        "axial",
        "core",
        "inside the core",
        "outside the core",
        [("ACL", LABEL), ("Synovitis (partly)", LABEL), ("ACL + PCL status", BUILD)],
        [("MCL", LABEL), ("Baker's", LABEL), ("PF OA", LABEL), ("Effusion", LABEL), ("tear place ×6", REPORT)],
    ),
    (
        "Ring radius",
        "axial",
        "ring",
        "the inner ring",
        "the outer ring",
        [("ACL", LABEL)],
        [("MCL + lateral ligament status", BUILD)],
    ),
    (
        "Upper cut (joint | above)",
        "side",
        "top",
        "below the cut",
        "above the cut",
        [("Medial Meniscus", LABEL), ("Lateral Meniscus", LABEL), ("Medial OA", LABEL), ("Lateral OA", LABEL), ("tear place ×6", REPORT)],
        [("quadriceps tendon", EXTRACT)],
    ),
    (
        "Lower cut (tibia | joint)",
        "side",
        "bottom",
        "below the cut",
        "above the cut",
        [("marrow edema in the tibia", EXTRACT)],
        [("PF OA", LABEL), ("Effusion", LABEL), ("Synovitis", LABEL), ("ACL", LABEL), ("tear place ×6", REPORT), ("facets ×2", REPORT)],
    ),
]


def sketch(axis, view, cut):
    axis.set_aspect("equal")
    axis.axis("off")
    grey, red = "#b9b9b9", "#d62728"
    if view == "axial":
        axis.set_xlim(-50, 50)
        axis.set_ylim(-50, 50)
        axis.add_patch(Circle((0, 0), 45, fill=False, color=grey, linewidth=1))
        axis.add_patch(Circle((0, 0), 27, fill=False, color=red if cut == "ring" else grey, linewidth=2.6 if cut == "ring" else 1))
        axis.add_patch(Circle((0, 0), 14, fill=False, color=red if cut == "core" else grey, linewidth=2.6 if cut == "core" else 1))
        for number in range(8):
            angle = np.radians(22.5 + 45 * number)
            axis.plot([14 * np.cos(angle), 45 * np.cos(angle)], [14 * np.sin(angle), 45 * np.sin(angle)], color=grey, linewidth=0.8)
        if cut == "side":
            axis.plot([0, 0], [-48, 48], color=red, linewidth=2.6)
        if cut == "front":
            axis.plot([-48, 48], [0, 0], color=red, linewidth=2.6)
        axis.text(0, 47, "front", ha="center", va="bottom", fontsize=7, color="#666666")
        axis.text(47, 0, " medial", ha="left", va="center", fontsize=7, color="#666666")
        axis.text(0, -49, "seen from above", ha="center", va="top", fontsize=7, color="#666666")
    else:
        axis.set_xlim(-50, 50)
        axis.set_ylim(-50, 50)
        axis.add_patch(Rectangle((-40, -45), 80, 90, fill=False, color=grey, linewidth=1))
        axis.plot([-40, 40], [12, 12], color=red if cut == "top" else grey, linewidth=2.6 if cut == "top" else 1)
        axis.plot([-40, 40], [-12, -12], color=red if cut == "bottom" else grey, linewidth=2.6 if cut == "bottom" else 1)
        for height, name in ((29, "above"), (0, "joint"), (-29, "tibia")):
            axis.text(0, height, name, ha="center", va="center", fontsize=7, color="#666666")
        axis.text(0, -49, "seen from the side", ha="center", va="top", fontsize=7, color="#666666")


figure = plt.figure(figsize=(15, 13.5))
grid = figure.add_gridspec(len(CUTS), 2, width_ratios=(1, 5.2), hspace=0.32, wspace=0.02, left=0.02, right=0.98, top=0.9, bottom=0.03)
for row, (title, view, cut, left_name, right_name, left, right) in enumerate(CUTS):
    sketch(figure.add_subplot(grid[row, 0]), view, cut)
    axis = figure.add_subplot(grid[row, 1])
    axis.set_xlim(-10, 10)
    axis.set_ylim(-3.6, 3.9)
    axis.axis("off")
    axis.plot([0, 0], [-3.2, 2.6], color="#d62728", linewidth=3)
    axis.text(0, 3.0, title, ha="center", va="bottom", fontsize=12, fontweight="bold")
    axis.text(-5, 3.0, f"reads only {left_name}", ha="center", va="bottom", fontsize=9, color="#666666")
    axis.text(5, 3.0, f"reads only {right_name}", ha="center", va="bottom", fontsize=9, color="#666666")
    for side, items in ((-1, left), (1, right)):
        present = [item for item in items if item[1] in (LABEL, REPORT)]
        for number, (name, colour) in enumerate(items):
            height = 2.1 - number * 0.86
            axis.add_patch(
                FancyArrowPatch(
                    (side * 3.2, height),
                    (side * 0.25, height),
                    arrowstyle="-|>",
                    mutation_scale=16,
                    color=colour,
                    linewidth=3.2 if colour == LABEL else 1.6,
                    linestyle="-" if colour in (LABEL, REPORT) else (0, (4, 3)),
                )
            )
            axis.text(side * 3.4, height, name, ha="left" if side > 0 else "right", va="center", fontsize=10, color=colour)
        if not present:
            axis.text(side * 5, 2.1 - 0.86 * len(items) - 0.2, "nothing today", ha="center", va="center", fontsize=10, color="#d62728", fontweight="bold")
figure.suptitle("What holds each cut: targets that read only one side of it, with the targets we have", fontsize=15, y=0.975)
figure.text(
    0.5,
    0.935,
    "thick black: one of the 12 labels (weight 1)      thin blue: report target in place (weight 0.1)      "
    "dashed green: from existing answers, not built      dashed orange: needs the v5 extraction",
    ha="center",
    fontsize=10,
)
out = Path("outputs/analysis/side-routing-cut-forces-20261006/cut-forces.png")
out.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(out, dpi=95)
print(out)
