import itertools
import json
from pathlib import Path

import numpy as np

PITCH_MM, SECTOR_DEG, COVERAGE = 12.0, 45.0, 0.8
ROOT = Path("outputs/analysis")
old = np.load(ROOT / "side-routing-old-vs-reference-20261006/old.npz", allow_pickle=True)
reference = np.load(ROOT / "side-routing-old-vs-reference-20261006/reference.npz", allow_pickle=True)
free = np.load(ROOT / "side-routing-attention-shape-20261006/attention.npz", allow_pickle=False)
labels = [str(name) for name in old["targets"]]
truth = free["truth"]
knees = len(old["study"])
PAIRS = (("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA"))


def label_points(number):
    points = np.zeros((knees, 10, 3), dtype=np.float32)
    weights = np.zeros((knees, 10), dtype=np.float32)
    for knee in range(knees):
        chosen = old["index"][knee, number, :10]
        points[knee] = reference["position"][knee][chosen]
        weights[knee] = old["weight"][knee, number, :10]
    return points, weights / weights.sum(-1, keepdims=True)


cache = {name: label_points(number) for number, name in enumerate(labels)}
centre = sum((cache[name][1][..., None] * cache[name][0]).sum(1) for name in PAIRS[0]) / 2
order = [name for name in labels if name not in ("Contusion", "Fracture")]
cases = np.array([truth[:, labels.index(name)].sum() for name in order], dtype=float)
HEIGHTS = np.arange(-30.0, 45.0, 3.0)
RADII = np.arange(6.0, 45.0, 3.0)
ANGLE_STEP = 15.0
height_bins = np.zeros((len(order), len(HEIGHTS) + 1))
radius_bins = np.zeros((len(order), len(RADII) + 1))
points_by_target = {}
for row, name in enumerate(order):
    points, weights = cache[name]
    chosen = truth[:, labels.index(name)]
    points = (points - centre[:, None])[chosen].reshape(-1, 3)
    weights = weights[chosen].reshape(-1)
    weights = weights / weights.sum()
    points_by_target[name] = (points, weights)
    height_bins[row] = np.bincount(np.searchsorted(HEIGHTS, points[:, 2], side="right"), weights=weights, minlength=len(HEIGHTS) + 1)
    radius_bins[row] = np.bincount(np.searchsorted(RADII, np.hypot(points[:, 0], points[:, 1]), side="right"), weights=weights, minlength=len(RADII) + 1)


def pair_rows():
    return [(order.index(a), order.index(b)) for a, b in PAIRS]


def judge(mass, reads, circular):
    count = mass.shape[1]
    pairs = [(i, (i + 1) % count) for i in range(count if circular else count - 1)]
    strengths, holders = [], []
    for low, high in pairs:
        below = reads[:, low] & ~reads[:, high]
        above = reads[:, high] & ~reads[:, low]
        strengths.append(min(cases[below].sum(), cases[above].sum()))
        holders.append(([order[i] for i in np.flatnonzero(below)], [order[i] for i in np.flatnonzero(above)]))
    return strengths, holders, (mass * reads).sum(1)


def linear(bins, candidates, most, read):
    best = None
    for count in range(1, most + 1):
        for chosen in itertools.combinations(range(len(candidates)), count):
            edges = candidates[list(chosen)]
            if count > 1 and np.diff(edges).min() < PITCH_MM:
                continue
            bounds = [0, *[c + 1 for c in chosen], bins.shape[1]]
            mass = np.stack([bins[:, a:b].sum(1) for a, b in zip(bounds[:-1], bounds[1:])], 1)
            reads = mass >= read
            for a, b in pair_rows():
                reads[a] = reads[b] = reads[a] | reads[b]
            strengths, holders, coverage = judge(mass, reads, False)
            if min(strengths) <= 0 or coverage.min() < COVERAGE:
                continue
            key = (mass.shape[1], min(strengths))
            if best is None or key > best[0]:
                best = (key, edges.tolist(), strengths, holders, coverage, reads, mass)
    return best


def angular(core, read):
    slots = int(360 / ANGLE_STEP)
    bins = np.zeros((len(order), slots))
    for row, name in enumerate(order):
        points, weights = points_by_target[name]
        outside = np.hypot(points[:, 0], points[:, 1]) > core
        if weights[outside].sum() < 0.3:
            outside = np.ones_like(outside)
        angle = np.degrees(np.arctan2(points[outside, 1], points[outside, 0])) % 360.0
        bins[row] = np.bincount((angle // ANGLE_STEP).astype(int) % slots, weights=weights[outside], minlength=slots) / weights[outside].sum()
    groups = sorted({tuple(sorted({slot, (slots // 2 - slot) % slots})) for slot in range(slots)})
    best = None
    for size in range(1, len(groups) + 1):
        for chosen in itertools.combinations(groups, size):
            cuts = sorted({slot for group in chosen for slot in group})
            if len(cuts) < 2 or len(cuts) > 8:
                continue
            gaps = np.diff([*cuts, cuts[0] + slots]) * ANGLE_STEP
            if gaps.min() < SECTOR_DEG:
                continue
            spans = [list(range(a, b)) for a, b in zip(cuts, [*cuts[1:], cuts[0] + slots])]
            mass = np.stack([bins[:, [slot % slots for slot in span]].sum(1) for span in spans], 1)
            reads = mass >= read
            mirror = [
                next(j for j, other in enumerate(spans) if {(slots // 2 - 1 - slot) % slots for slot in span} == {slot % slots for slot in other})
                for span in spans
            ]
            for a, b in pair_rows():
                union = reads[a] | reads[b][mirror]
                reads[a], reads[b] = union, union[mirror]
            strengths, holders, coverage = judge(mass, reads, True)
            if min(strengths) <= 0 or coverage.min() < COVERAGE:
                continue
            key = (len(cuts), min(strengths))
            if best is None or key > best[0]:
                edges = [cut * ANGLE_STEP for cut in cuts]
                best = (key, edges, strengths[-1:] + strengths[:-1], holders[-1:] + holders[:-1], coverage, reads, mass)
    return best


def show(title, best, names=None):
    if best is None:
        print(f"{title}: no division with every cut held")
        return 1
    (count, weakest), edges, strengths, holders, coverage, reads, mass = best
    print(f"{title}: {count} parts, cuts at {edges}; weakest side {weakest:.0f} cases; lowest coverage {coverage.min():.2f}")
    for edge, strength, (below, above) in zip(edges, strengths, holders, strict=True):
        print(f"   cut {edge:6.1f}: {strength:5.0f} | before: {', '.join(below)} || after: {', '.join(above)}")
    for name, row, share in zip(order, reads, mass, strict=True):
        print(f"      {name:17} reads {np.flatnonzero(row).tolist()}  shares {np.round(share, 2).tolist()}")
    return count


for read in (0.10, 0.15, 0.20):
    print(f"\n===== a label reads a part holding at least {read:.2f} of its attention; pairs mirrored =====")
    height_best = linear(height_bins, HEIGHTS, 4, read)
    bands = show("Heights", height_best)
    radius_best = linear(radius_bins, RADII, 3, read)
    rings = show("Radii", radius_best)
    core = radius_best[1][0] if radius_best else 0.0
    angle_best = angular(core, read)
    sectors = show("Sectors (degrees from medial through the front)", angle_best)
    outer = max(rings - 1, 1)
    print(f"Sections: {bands} heights x ({outer} rings x {sectors} sectors{' + core' if rings > 1 else ''}) = {bands * (outer * sectors + (rings > 1))}")
    (ROOT / "side-routing-cut-forces-20261006" / f"symmetric-division-{read:.2f}.json").write_text(
        json.dumps(
            {
                "heights": height_best[1] if height_best else [],
                "radii": radius_best[1] if radius_best else [],
                "angles": angle_best[1] if angle_best else [],
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
