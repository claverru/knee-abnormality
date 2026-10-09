from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FIXED, SPOTS, LEARNED = "#8a8a8a", "#1f6fb5", "#d9730d"
OUTER, CORE, RING, TOP, BOTTOM, LOW, HIGH = 45.0, 16.0, 30.0, 11.0, -13.0, -42.0, 40.0
turn = np.linspace(0, 2 * np.pi, 120)
figure = plt.figure(figsize=(19, 11))
axis = figure.add_axes((0.0, 0.02, 0.6, 0.9), projection="3d")


def circle(radius, height, colour, width, style="-"):
    axis.plot(radius * np.cos(turn), radius * np.sin(turn), height, color=colour, linewidth=width, linestyle=style)


def disc(height, colour, alpha):
    points = [(OUTER * np.cos(a), OUTER * np.sin(a), height) for a in turn]
    axis.add_collection3d(Poly3DCollection([points], facecolors=colour, edgecolors="none", alpha=alpha))


for height in (LOW, HIGH):
    circle(OUTER, height, FIXED, 1.0)
for angle in np.radians(22.5 + 45.0 * np.arange(8)):
    x, y = np.cos(angle), np.sin(angle)
    axis.plot([OUTER * x] * 2, [OUTER * y] * 2, [LOW, HIGH], color=FIXED, linewidth=0.8)
    for height in (TOP, BOTTOM):
        axis.plot([CORE * x, OUTER * x], [CORE * y, OUTER * y], [height, height], color=FIXED, linewidth=1.6)
for height in (TOP, BOTTOM):
    disc(height, SPOTS, 0.13)
    circle(OUTER, height, SPOTS, 2.6)
    circle(CORE, height, LEARNED, 2.6)
    circle(RING, height, LEARNED, 2.6)
for radius in (CORE, RING):
    for angle in np.radians((200.0, 340.0)):
        axis.plot([radius * np.cos(angle)] * 2, [radius * np.sin(angle)] * 2, [BOTTOM, TOP], color=LEARNED, linewidth=1.2, linestyle=":")
spots = {"medial meniscus spot": (24.0, 0.0, 0.0), "lateral meniscus spot": (-24.0, 0.0, 0.0), "kneecap spot": (0.0, 31.0, 22.0)}
for name, (x, y, z) in spots.items():
    axis.scatter([x], [y], [z], color=SPOTS, s=150, depthshade=False, edgecolors="white", linewidths=1.2)
axis.plot([-24, 24], [0, 0], [0, 0], color=SPOTS, linewidth=3.0)
axis.scatter([0], [0], [0], color=SPOTS, s=90, marker="D", depthshade=False, edgecolors="white")
axis.quiver(0, 0, 0, 5, -4, 4, color=LEARNED, linewidth=2.6, arrow_length_ratio=0.35)
axis.plot([0, 0], [0, 0], [LOW, HIGH], color=FIXED, linewidth=0.8, linestyle="--")
nod = np.radians(np.linspace(0, 22, 20))
axis.plot(np.zeros_like(nod) + 46, 0 + 34 * np.sin(nod) * 0 + 0 * nod, 0 * nod, alpha=0)
arc_y, arc_z = 52 * np.cos(nod), 52 * np.sin(nod)
axis.plot(np.zeros_like(nod), arc_y, arc_z, color=LEARNED, linewidth=2.6)
axis.quiver(0, arc_y[-2], arc_z[-2], 0, arc_y[-1] - arc_y[-2], arc_z[-1] - arc_z[-2], color=LEARNED, linewidth=2.6, arrow_length_ratio=6.0)
def at(radius, degrees, height):
    return radius * np.cos(np.radians(degrees)), radius * np.sin(np.radians(degrees)), height


numbers = (
    ((0.0, -4.0, 4.0), "1", SPOTS),
    ((12.0, -3.0, -4.0), "2", SPOTS),
    (at(OUTER + 4, 195, TOP), "3", SPOTS),
    (at(OUTER + 4, 195, BOTTOM), "4", SPOTS),
    ((0.0, 58.0, 12.0), "5", LEARNED),
    (at(CORE, 250, BOTTOM), "6", LEARNED),
    (at(RING, 250, BOTTOM), "7", LEARNED),
    ((7.0, -6.0, 7.0), "8", LEARNED),
    (at(OUTER + 3, 67.5, HIGH), "9", FIXED),
)
for (x, y, z), text, colour in numbers:
    axis.text(x, y, z, text, color="white", fontsize=11, fontweight="bold", ha="center", va="center",
              bbox={"boxstyle": "circle,pad=0.25", "facecolor": colour, "edgecolor": "white"}, zorder=20)
for (x, y, z), text in (((28, 4, -6), "medial meniscus spot"), ((-78, 0, -8), "lateral meniscus spot"), ((-36, 31, 27), "kneecap spot")):
    axis.text(x, y, z, text, color=SPOTS, fontsize=10.5, fontweight="bold")
axis.set_xlim(-50, 50)
axis.set_ylim(-50, 50)
axis.set_zlim(-45, 45)
axis.set_box_aspect((1, 1, 0.9))
axis.set_xticks([-35, 35], ["lateral", "medial"])
axis.set_yticks([-35, 35], ["back", "front"])
axis.set_zticks([-28, 0, 26], ["tibia band", "joint band", "upper band"])
axis.tick_params(labelsize=9, pad=1)
axis.view_init(elev=22, azim=-62)
panel = figure.add_axes((0.6, 0.03, 0.39, 0.9))
panel.axis("off")
rows = (
    (SPOTS, "PLACED PER KNEE FROM THE HEADS' ATTENTION, NO GRADIENT", None),
    (SPOTS, "1  Centre", "Midpoint of the two meniscus spots."),
    (SPOTS, "2  Side axis (roll and turn)", "The line through the two meniscus spots."),
    (SPOTS, "3  Upper cut", "Between the menisci's height and the kneecap spot's, weighted by\nhow spread each spot is. Validation mean +11 mm."),
    (SPOTS, "4  Lower cut", "Mirror of the upper cut about the menisci. Validation mean −13 mm."),
    (LEARNED, "LEARNED BY GRADIENT FROM THE LABELS", None),
    (LEARNED, "5  Tilt forward (nod)", "Per knee, regressed from the tokens, plus one shared rotation.\nValidation mean 6°."),
    (LEARNED, "6, 7  Core radius, ring radius", "Per knee: a shared start (14 and 28 mm), a shared offset and a\nregression from the tokens. Validation means 16 and 30 mm."),
    (LEARNED, "8  Shared shift of the centre", "Three numbers, the same for every knee."),
    (LEARNED, "Which side is medial", "Scan header where it is certain, otherwise a small head on the tokens."),
    (FIXED, "FIXED BY CONSTRUCTION", None),
    (FIXED, "9  Sector boundaries", "Eight sectors of 45°, centred on the frame's axes."),
    (FIXED, "Region table", "Which sectors, rings and bands each label reads; set once from\nthe attention analysis of 2026-10-06."),
    (FIXED, "How a cut works", "A token belongs wholly to the section its centre is in; the gradient\npasses through a soft edge as wide as the token."),
)
y = 0.97
for colour, title, text in rows:
    if text is None:
        y -= 0.02
        panel.text(0.0, y, title, color=colour, fontsize=12.5, fontweight="bold", va="top")
        y -= 0.05
        continue
    panel.text(0.02, y, title, color=colour, fontsize=11.5, fontweight="bold", va="top")
    panel.text(0.02, y - 0.028, text, color="#222222", fontsize=10.5, va="top")
    y -= 0.062 + 0.026 * text.count("\n")
figure.suptitle("The incumbent's frame: what is placed from the heads' attention, what is learned, and what is fixed", fontsize=15, y=0.975)
out = Path("outputs/analysis/r/routing-and-shifts-frames-20261007/fixed-and-learned-3d.png")
out.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(out, dpi=90)
print(out)
