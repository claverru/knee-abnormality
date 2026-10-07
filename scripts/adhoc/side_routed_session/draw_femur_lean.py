from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path("outputs/analysis/routing-and-shifts-frames-20261007")
CORE, RING, OUTER = 15.4, 28.7, 45.0
BOTTOM, TOP, LOW, HIGH = -11.2, 10.7, -42.0, 46.0
LEAN, NOD = np.radians(-25.0), np.radians(7.0)
SECTORS = {
    "anteromedial": 22.5, "anterior": 67.5, "anterolateral": 112.5, "lateral": 157.5,
    "posterolateral": 202.5, "posterior": 247.5, "posteromedial": 292.5, "medial": 337.5,
}
MEDIAL = ("anteromedial", "medial", "posteromedial")
LATERAL = ("anterolateral", "lateral", "posterolateral")
FRONT = ("anteromedial", "anterior", "anterolateral")
BANDS = {"tibia": (LOW, BOTTOM), "joint": (BOTTOM, TOP), "upper": (TOP, HIGH)}


def wedge(inner, outer, start, stop, low, high, steps=6):
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


def place(faces):
    out = []
    for face in faces:
        points = []
        for x, y, z in face:
            y = y + max(z - TOP, 0.0) * np.tan(LEAN)
            points.append((x, y * np.cos(NOD) + z * np.sin(NOD), z * np.cos(NOD) - y * np.sin(NOD)))
        out.append(points)
    return out


CELLS = {}
for band, (low, high) in BANDS.items():
    pad = 0.6
    for side, (start, stop) in (("medial", (-90.0, 90.0)), ("lateral", (90.0, 270.0))):
        CELLS[("core", side, band)] = wedge(0.0, CORE - pad, start + 1, stop - 1, low + pad, high - pad, steps=12)
    for ring, (inner, outer) in (("inner", (CORE, RING)), ("outer", (RING, OUTER))):
        for sector, start in SECTORS.items():
            CELLS[(ring, sector, band)] = wedge(inner + pad, outer - pad, start + 1.5, start + 43.5, low + pad, high - pad)


def compartment(names):
    def read(ring, part, band):
        if band == "upper":
            return False
        return part == names[1] if ring == "core" else part in names[0]
    return read


def kneecap(ring, part, band):
    return ring != "core" and part in FRONT and band != "tibia"


def ligament(ring, part, band):
    if band == "tibia":
        return False
    return ring == "core" or (ring == "inner" and part in ("anterior", "posterior", *LATERAL))


LABELS = {
    "ACL": ("#5aa469", ligament, "core, and the inner ring's front, back and lateral sectors; joint and upper bands"),
    "MCL": ("#8e5ea2", lambda r, p, b: r != "core" and p in ("anteromedial", "medial"), "medial and anteromedial sectors outside the core; all heights"),
    "Medial Meniscus": ("#d1495b", compartment((MEDIAL, "medial")), "three medial sectors and the core's medial half; tibia and joint bands"),
    "Lateral Meniscus": ("#3a7ca5", compartment((LATERAL, "lateral")), "three lateral sectors and the core's lateral half; tibia and joint bands"),
    "Medial OA": ("#e07a5f", compartment((MEDIAL, "medial")), "three medial sectors and the core's medial half; tibia and joint bands"),
    "Lateral OA": ("#5b8fb9", compartment((LATERAL, "lateral")), "three lateral sectors and the core's lateral half; tibia and joint bands"),
    "PF OA": ("#f2c14e", kneecap, "three front sectors outside the core; joint and upper bands"),
    "Effusion": ("#e9a03b", kneecap, "three front sectors outside the core; joint and upper bands"),
    "Synovitis": ("#c77dba", lambda r, p, b: b != "tibia" and (r == "core" or p in FRONT), "three front sectors and the core; joint and upper bands"),
    "Baker's": ("#2a9d8f", lambda r, p, b: r != "core" and p in ("posterior", "posteromedial", "medial"), "back, posteromedial and medial sectors outside the core; all heights"),
    "Contusion": ("#7a8b99", lambda r, p, b: True, "everything (not routed)"),
    "Fracture": ("#7a8b99", lambda r, p, b: True, "everything (not routed)"),
}
figure = plt.figure(figsize=(23, 18))
for number, (label, (colour, read, text)) in enumerate(LABELS.items(), 1):
    axis = figure.add_subplot(3, 4, number, projection="3d")
    for (ring, part, band), faces in CELLS.items():
        inside = read(ring, part, band)
        leaning = band == "upper"
        axis.add_collection3d(
            Poly3DCollection(
                place(faces),
                facecolors=colour if inside else "#c9c9c9",
                edgecolors=("#222222" if inside else "#9a9a9a"),
                linewidths=0.5 if inside else 0.25,
                alpha=(0.85 if inside else (0.10 if leaning else 0.05)),
            )
        )
    hinge = np.array([0.0, 0.0, TOP])
    tip = np.array([0.0, (HIGH - TOP) * np.tan(LEAN), HIGH])
    for a, b, style in ((np.array([0.0, 0.0, LOW]), hinge, "-"), (hinge, tip, "-")):
        points = np.array(place([[tuple(a), tuple(b)]])[0])
        axis.plot(points[:, 0], points[:, 1], points[:, 2], color="black", linewidth=1.6, linestyle=style)
    limit = 50
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit, limit)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xticks([-30, 30], ["lateral", "medial"])
    axis.set_yticks([-30, 30], ["back", "front"])
    axis.set_zticks([-27, 0, 28], ["tibia", "joint", "femur"])
    axis.tick_params(labelsize=8, pad=0)
    axis.view_init(elev=14, azim=-18)
    axis.set_title(f"{label}\n{text}", fontsize=9.5, pad=0)
figure.suptitle(
    "The frame of the femur-lean run at epoch 7, per label. Coloured cells are read by the label, grey ones are not; the black line is the axis of each segment.\n"
    "Tibia and joint bands: the incumbent's cylinder (core 15 mm, ring 29 mm, cuts at -11 and +11 mm, tilted forward 7 degrees). Femur band: the same sectors around an axis leaning 25 degrees backwards from the upper cut.",
    fontsize=13,
)
figure.tight_layout(rect=(0, 0, 1, 0.95))
figure.savefig(OUT / "femur-lean-per-label-3d.png", dpi=80)
print(OUT / "femur-lean-per-label-3d.png")
