import itertools
import json
import sys
from pathlib import Path

import numpy as np

READ = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
USE_REPORTS = len(sys.argv) > 2 and sys.argv[2] == "reports"
PITCH_MM, SECTOR_DEG = 12.0, 45.0
ROOT = Path("outputs/analysis")
old = np.load(ROOT / "side-routing-old-vs-reference-20261006/old.npz", allow_pickle=True)
reference = np.load(ROOT / "side-routing-old-vs-reference-20261006/reference.npz", allow_pickle=True)
free = np.load(ROOT / "side-routing-attention-shape-20261006/attention.npz", allow_pickle=False)
labels = [str(name) for name in old["targets"]]
assert (old["study"] == free["study"]).all()
truth = free["truth"]
assert [str(name) for name in free["labels"]] == labels
knees = len(old["study"])


def label_points(number):
    points = np.zeros((knees, 10, 3), dtype=np.float32)
    weights = np.zeros((knees, 10), dtype=np.float32)
    for knee in range(knees):
        chosen = old["index"][knee, number, :10]
        points[knee] = reference["position"][knee][chosen]
        weights[knee] = old["weight"][knee, number, :10]
    return points, weights / weights.sum(-1, keepdims=True)


medial_points, medial_weights = label_points(labels.index("Medial Meniscus"))
lateral_points, lateral_weights = label_points(labels.index("Lateral Meniscus"))
centre = (
    (medial_weights[..., None] * medial_points).sum(1) + (lateral_weights[..., None] * lateral_points).sum(1)
) / 2
targets = {}
for number, name in enumerate(labels):
    if name in ("Contusion", "Fracture"):
        continue
    points, weights = label_points(number)
    chosen = truth[:, number]
    targets[name] = ((points - centre[:, None])[chosen].reshape(-1, 3), weights[chosen].reshape(-1), float(chosen.sum()))
if USE_REPORTS:
    names = [str(name) for name in free["names"]]
    weight = free["weight"] / free["weight"].sum(-1, keepdims=True)
    for column, question in enumerate(str(name) for name in free["questions"]):
        head = names.index(f"report:{question}")
        chosen = free["abnormal"][:, column] > 0.5
        if chosen.sum() < 40 or question in ("study_context", "overall_osteoarthritis", "osteophytes", "marrow_edema", "fracture_status"):
            continue
        targets[f"r:{question}"] = (
            (free["fitted"][:, head] - centre[:, None])[chosen].reshape(-1, 3),
            weight[chosen, head].reshape(-1),
            0.1 * float(chosen.sum()),
        )
order = list(targets)
cases = np.array([targets[name][2] for name in order])


def masses(values, weights_list, edges, circular=False):
    result = np.zeros((len(order), len(edges) if circular else len(edges) + 1))
    for row, (value, weight) in enumerate(zip(values, weights_list, strict=True)):
        place = np.searchsorted(edges, value, side="right")
        if circular:
            place = (place - 1) % len(edges)
        result[row] = np.bincount(place, weights=weight, minlength=result.shape[1]) / weight.sum()
    return result


def judge(mass, circular):
    reads = mass >= READ
    pairs = [(i, (i + 1) % mass.shape[1]) for i in range(mass.shape[1] if circular else mass.shape[1] - 1)]
    strengths, holders = [], []
    for low, high in pairs:
        below = reads[:, low] & ~reads[:, high]
        above = reads[:, high] & ~reads[:, low]
        strengths.append(min(cases[below].sum(), cases[above].sum()))
        holders.append(([order[i] for i in np.flatnonzero(below)], [order[i] for i in np.flatnonzero(above)]))
    coverage = (mass * reads).sum(1)
    return strengths, holders, coverage, reads


def search(values, weights_list, candidates, most, spacing, circular=False):
    best = None
    for count in range(1 if not circular else 2, most + 1):
        for edges in itertools.combinations(candidates, count):
            gaps = np.diff(edges)
            if circular:
                gaps = np.append(gaps, edges[0] + 360.0 - edges[-1])
            if len(gaps) and gaps.min() < spacing:
                continue
            mass = masses(values, weights_list, np.array(edges), circular)
            strengths, holders, coverage, reads = judge(mass, circular)
            if min(strengths) <= 0 or coverage.min() < 0.8:
                continue
            key = (mass.shape[1], min(strengths))
            if best is None or key > best[0]:
                best = (key, edges, strengths, holders, coverage, reads, mass)
    return best


def report(title, best, unit, circular=False):
    if best is None:
        print(f"{title}: no division with every cut held")
        return 1
    (count, weakest), edges, strengths, holders, coverage, reads, mass = best
    print(f"\n{title}: {count} parts, cuts at {[float(e) for e in edges]} {unit}; weakest side {weakest:.0f} cases; lowest coverage {coverage.min():.2f}")
    for edge, strength, (below, above) in zip(edges if not circular else edges, strengths, holders, strict=False):
        print(f"  cut {edge:6.1f}: {strength:6.0f} | one side: {', '.join(below)}  ||  other side: {', '.join(above)}")
    for name, row, share in zip(order, reads, mass, strict=True):
        print(f"    {name:34} reads parts {np.flatnonzero(row).tolist()}  shares {np.round(share, 2).tolist()}")
    return count


heights = [targets[name][0][:, 2] for name in order]
radii = [np.hypot(targets[name][0][:, 0], targets[name][0][:, 1]) for name in order]
weights_list = [targets[name][1] for name in order]
print(f"read threshold {READ:.2f}; targets: {len(order)} ({'labels and report questions' if USE_REPORTS else 'labels only'})")
height_best = search(heights, weights_list, np.arange(-30.0, 45.0, 3.0), 4, PITCH_MM)
bands = report("Heights (parts from the bottom)", height_best, "mm")
radius_best = search(radii, weights_list, np.arange(6.0, 45.0, 3.0), 3, PITCH_MM)
rings = report("Radii (parts from the axis)", radius_best, "mm")
core = radius_best[1][0] if radius_best else 0.0
angles, angle_weights = [], []
for name in order:
    points, weight, _ = targets[name]
    outside = np.hypot(points[:, 0], points[:, 1]) > core
    if weight[outside].sum() < 0.3 * weight.sum():
        outside = np.ones_like(outside)
    angles.append(np.degrees(np.arctan2(points[outside, 1], points[outside, 0])) % 360.0)
    angle_weights.append(weight[outside])
angle_best = search(angles, angle_weights, np.arange(0.0, 360.0, 15.0), 8, SECTOR_DEG, circular=True)
sectors = report("Sectors outside the core (degrees from medial, counter-clockwise through the front)", angle_best, "deg", circular=True)
print(f"\nSections: {bands} heights x ({max(rings - 1, 1)} rings x {sectors} sectors{' + core' if rings > 1 else ''}) = {bands * (max(rings - 1, 1) * sectors + (rings > 1))}")
out = ROOT / "side-routing-cut-forces-20261006" / f"held-division-{'reports' if USE_REPORTS else 'labels'}-{READ:.2f}.json"
out.write_text(
    json.dumps(
        {
            "heights": [float(e) for e in height_best[1]] if height_best else [],
            "radii": [float(e) for e in radius_best[1]] if radius_best else [],
            "angles": [float(e) for e in angle_best[1]] if angle_best else [],
            "reads": {
                name: {
                    "heights": np.flatnonzero(height_best[5][i]).tolist() if height_best else [0],
                    "radii": np.flatnonzero(radius_best[5][i]).tolist() if radius_best else [0],
                    "sectors": np.flatnonzero(angle_best[5][i]).tolist() if angle_best else [0],
                }
                for i, name in enumerate(order)
            },
        },
        indent=1,
    )
)
print(out)
