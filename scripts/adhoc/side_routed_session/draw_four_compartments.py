from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

OUT = Path("outputs/analysis/side-routing-cut-forces-20261006")
RADIUS, CORE, LOW, HIGH = 45.0, 13.0, -45.0, 45.0
PLANE_DEG, PLANE_OFFSET = 15.0, 6.0
STEP = 4.5
edges_xy = np.arange(-RADIUS, RADIUS + STEP, STEP)
edges_z = np.arange(LOW, HIGH + STEP, STEP)
centre_xy = (edges_xy[:-1] + edges_xy[1:]) / 2
centre_z = (edges_z[:-1] + edges_z[1:]) / 2
front, lateral, up = np.meshgrid(centre_xy, centre_xy, centre_z, indexing="ij")
medial = -lateral
distance = np.hypot(medial, front)
knee = distance <= RADIUS
core = distance <= CORE
angle = np.radians(PLANE_DEG)
kneecap = knee & ~core & (front * np.sin(angle) + up * np.cos(angle) > PLANE_OFFSET)
COMPARTMENTS = {
    "core": core,
    "kneecap": kneecap,
    "medial": knee & ~core & ~kneecap & (medial > 0),
    "lateral": knee & ~core & ~kneecap & (medial <= 0),
}
LABELS = {
    "ACL": ("#5aa469", ("core",), "the core"),
    "MCL": ("#8e5ea2", ("medial",), "the medial compartment"),
    "Medial Meniscus": ("#d1495b", ("medial",), "the medial compartment"),
    "Lateral Meniscus": ("#3a7ca5", ("lateral",), "the lateral compartment"),
    "Medial OA": ("#e07a5f", ("medial",), "the medial compartment"),
    "Lateral OA": ("#5b8fb9", ("lateral",), "the lateral compartment"),
    "PF OA": ("#f2c14e", ("kneecap",), "the kneecap compartment"),
    "Effusion": ("#e9a03b", ("kneecap",), "the kneecap compartment"),
    "Synovitis": ("#c77dba", ("kneecap",), "the kneecap compartment"),
    "Baker's": ("#2a9d8f", ("medial",), "the medial compartment"),
    "Contusion": ("#7a8b99", tuple(COMPARTMENTS), "everything (not routed)"),
    "Fracture": ("#7a8b99", tuple(COMPARTMENTS), "everything (not routed)"),
}
grid_x, grid_y, grid_z = np.meshgrid(edges_xy, edges_xy, edges_z, indexing="ij")
figure = plt.figure(figsize=(22, 18))
for number, (label, (colour, read, text)) in enumerate(LABELS.items(), 1):
    axis = figure.add_subplot(3, 4, number, projection="3d")
    chosen = np.zeros_like(knee)
    for name in read:
        chosen |= COMPARTMENTS[name]
    axis.voxels(grid_x, grid_y, grid_z, chosen, facecolors=to_rgba(colour, 0.95), edgecolors=to_rgba("#222222", 0.3), linewidth=0.3)
    turn = np.linspace(0, 2 * np.pi, 80)
    for height in (LOW, HIGH):
        axis.plot(RADIUS * np.cos(turn), RADIUS * np.sin(turn), height, color="#888888", linewidth=0.9)
        axis.plot(CORE * np.cos(turn), CORE * np.sin(turn), height, color="#888888", linewidth=0.6, linestyle=":")
    for spoke in np.radians(np.arange(0, 360, 45)):
        axis.plot([RADIUS * np.cos(spoke)] * 2, [RADIUS * np.sin(spoke)] * 2, [LOW, HIGH], color="#bbbbbb", linewidth=0.6)
    axis.set_xlim(-RADIUS, RADIUS)
    axis.set_ylim(-RADIUS, RADIUS)
    axis.set_zlim(LOW, HIGH)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xticks([-30, 30], ["back", "front"])
    axis.set_yticks([-30, 30], ["medial", "lateral"])
    axis.set_zticks([-30, 0, 30], ["tibia", "joint", "above"])
    axis.tick_params(labelsize=8, pad=0)
    axis.view_init(elev=20, azim=-28)
    axis.set_title(f"{label}\n{text}", fontsize=10, pad=0)
figure.suptitle(
    "Four compartments from three cuts, each held by labels on both sides: the core cylinder, one tilted kneecap plane, the midline.\n"
    "Coloured blocks are read by the target; the grey outline is the knee. The plane is drawn where the reference put its cuts (tilted 15 degrees, 6 mm above the centre);\nin the model its tilt and height, the core's radius and the midline are learned.",
    fontsize=13,
)
figure.tight_layout(rect=(0, 0, 1, 0.95))
figure.savefig(OUT / "four-compartments-per-target-3d.png", dpi=80)
print(OUT / "four-compartments-per-target-3d.png")
