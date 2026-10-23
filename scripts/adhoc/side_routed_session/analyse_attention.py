import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering

data = np.load(sys.argv[1], allow_pickle=False)
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
names = [str(name) for name in data["names"]]
labels = [str(name) for name in data["labels"]]
ROUTED = {
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Baker's",
}
free = np.array([name not in ROUTED for name in names])
truth, abnormal = data["truth"], data["abnormal"]
present = np.concatenate((truth, abnormal > 0.5), 1)
weight = data["weight"] / data["weight"].sum(-1, keepdims=True)
EDGE, CELL = 54.0, 6.0
bins = np.arange(-EDGE, EDGE + CELL, CELL)
count = len(bins) - 1
centres = (bins[:-1] + bins[1:]) / 2
grid = np.stack(np.meshgrid(centres, centres, centres, indexing="ij"), -1).reshape(
    -1, 3
)


def densities(points):
    result = np.zeros((len(names), count**3))
    knees = np.zeros(len(names), dtype=int)
    for head in range(len(names)):
        chosen = present[:, head]
        if chosen.sum() < 25:
            chosen = np.ones(len(present), dtype=bool)
        knees[head] = chosen.sum()
        values = points[chosen, head].reshape(-1, 3)
        histogram, _ = np.histogramdd(
            values, bins=(bins, bins, bins), weights=weight[chosen, head].reshape(-1)
        )
        result[head] = histogram.reshape(-1) / max(histogram.sum(), 1e-9)
    return result, knees


def information(density, section):
    joint = np.zeros((density.shape[0], section.max() + 1))
    np.add.at(joint.T, section, density.T)
    joint = joint / joint.sum()
    heads = joint.sum(1, keepdims=True)
    parts = joint.sum(0, keepdims=True)
    ratio = np.where(joint > 0, joint / (heads * parts + 1e-30), 1.0)
    return float((joint * np.log2(ratio)).sum())


fitted = data["fitted"]
medial = names.index("report:medial_meniscus_status")
lateral = names.index("report:lateral_meniscus_status")
spot = lambda head: (weight[:, head, :, None] * fitted[:, head]).sum(1)
midpoint = (spot(medial) + spot(lateral)) / 2
span = np.linalg.norm(spot(medial) - spot(lateral), axis=1)
print(
    f"knees {len(fitted)}; free meniscus-status spots: distance {span.mean():.1f} mm (SD {span.std():.1f}), midpoint mean {midpoint.mean(0).round(1)} SD {midpoint.std(0).round(1)}"
)
frames = {
    "geometric frame": fitted,
    "centred on the meniscus spots": fitted - midpoint[:, None, None],
}
summary = {}
for frame_name, points in frames.items():
    density, knees = densities(points)
    ceiling = information(density[free], np.arange(count**3))
    ceiling_labels = information(density[: len(labels)], np.arange(count**3))
    m, f, z = grid[:, 0], grid[:, 1], grid[:, 2]
    angle = np.degrees(np.arctan2(f, m))
    radius = np.hypot(m, f)
    levels = np.arange(-36.0, 37.0, 6.0)

    def bands(cuts):
        return np.searchsorted(np.array(cuts), z)

    def sectors(number, offset):
        return np.floor(((angle - offset) % 360.0) / (360.0 / number)).astype(int)

    def best(build, options):
        scored = [
            (information(density[free], build(*option)), option) for option in options
        ]
        value, option = max(scored)
        section = build(*option)
        return value, option, section

    shapes = []
    shapes.append(("whole knee", 0.0, (), np.zeros(count**3, dtype=int)))
    shapes.append(
        (
            "octants, cuts through the centre",
            *(
                (
                    information(density[free], sectors(4, 0) * 2 + bands((0.0,))),
                    (),
                    sectors(4, 0) * 2 + bands((0.0,)),
                )
            ),
        )
    )
    shapes.append(
        (
            "quadrants x 2 heights, height cut placed",
            *best(
                lambda cut: sectors(4, 0) * 2 + bands((cut,)), [(c,) for c in levels]
            ),
        )
    )
    pairs = [(a, b) for a, b in itertools.combinations(levels, 2)]
    triples = [c for c in itertools.combinations(levels[::2], 3)]
    shapes.append(
        (
            "quadrants x 3 heights",
            *best(lambda a, b: sectors(4, 0) * 3 + bands((a, b)), pairs),
        )
    )
    shapes.append(
        (
            "quadrants x 4 heights",
            *best(lambda a, b, c: sectors(4, 0) * 4 + bands((a, b, c)), triples),
        )
    )
    for number, offsets in ((6, (0.0, 30.0)), (8, (0.0, 22.5)), (12, (0.0, 15.0))):
        shapes.append(
            (
                f"{number} sectors x 3 heights",
                *best(
                    lambda a, b, o, n=number: sectors(n, o) * 3 + bands((a, b)),
                    [(a, b, o) for a, b in pairs for o in offsets],
                ),
            )
        )
    low, high = shapes[3][2]
    for number, offsets in ((4, (0.0,)), (8, (0.0, 22.5))):

        def with_core(r, o, n=number):
            section = sectors(n, o) * 3 + bands((low, high))
            return np.where(radius < r, n * 3 + bands((low, high)), section)

        shapes.append(
            (
                f"{number} sectors + core x 3 heights",
                *best(
                    with_core,
                    [
                        (r, o)
                        for r in (9.0, 12.0, 15.0, 18.0, 21.0, 24.0)
                        for o in offsets
                    ],
                ),
            )
        )

    def rings(r, big):
        section = sectors(8, 0.0) * 3 + bands((low, high))
        section = np.where(radius > big, section + 24, section)
        return np.where(radius < r, 48 + bands((low, high)), section)

    shapes.append(
        (
            "8 sectors + core + outer ring x 3 heights",
            *best(
                rings,
                [(r, big) for r in (12.0, 15.0, 18.0) for big in (30.0, 36.0, 42.0)],
            ),
        )
    )
    print(
        f"\n=== {frame_name}: how much of what separates the free heads each shape captures (ceiling {ceiling:.3f} bits at {CELL:.0f} mm cells) ==="
    )
    table = []
    for name, value, option, section in shapes:
        labels_value = information(density[: len(labels)], section)
        table.append(
            {
                "shape": name,
                "sections": int(len(np.unique(section))),
                "free_heads": round(value / ceiling, 3),
                "labels": round(labels_value / ceiling_labels, 3),
                "parameters": [float(v) for v in option],
            }
        )
        print(
            f"{name:44} sections {len(np.unique(section)):3}  free heads {100 * value / ceiling:5.1f}%  diagnosis labels {100 * labels_value / ceiling_labels:5.1f}%  cuts {tuple(float(v) for v in option)}"
        )
    mass = density[free].sum(0)
    kept = (
        mass
        > np.sort(mass)[::-1][
            np.searchsorted(np.cumsum(np.sort(mass)[::-1]), 0.97 * mass.sum())
        ]
    )
    index = np.full(count**3, -1)
    index[kept] = np.arange(kept.sum())
    cells = np.argwhere(kept.reshape(count, count, count))
    links = []
    for axis in range(3):
        step = np.zeros(3, dtype=int)
        step[axis] = 1
        other = cells + step
        inside = (other < count).all(1)
        a = index.reshape(count, count, count)[tuple(cells[inside].T)]
        b = index.reshape(count, count, count)[tuple(other[inside].T)]
        good = b >= 0
        links.append(np.stack((a[good], b[good])))
    links = np.concatenate(links, 1)
    connectivity = sparse.coo_matrix(
        (np.ones(links.shape[1]), (links[0], links[1])), shape=(kept.sum(), kept.sum())
    )
    connectivity = connectivity + connectivity.T
    features = density[free][:, kept].T
    features = np.sqrt(features)
    print(
        f"\n--- {frame_name}: sections that emerge from grouping neighbouring cells looked at by the same heads ---"
    )
    emergent = {}
    for groups in (4, 6, 8, 10, 12, 15, 18, 24):
        model = AgglomerativeClustering(
            n_clusters=groups, connectivity=connectivity, linkage="ward"
        ).fit(features)
        section = np.full(count**3, groups)
        section[kept] = model.labels_
        value = information(density[free], section)
        labels_value = information(density[: len(labels)], section)
        emergent[groups] = section
        print(
            f"emergent, {groups:2} sections: free heads {100 * value / ceiling:5.1f}%  diagnosis labels {100 * labels_value / ceiling_labels:5.1f}%"
        )
    summary[frame_name] = {"ceiling": ceiling, "shapes": table}
    np.savez_compressed(
        OUT
        / f"sections-{'geometric' if frame_name.startswith('geo') else 'spots'}.npz",
        density=density,
        knees=knees,
        **{f"emergent_{k}": v for k, v in emergent.items()},
    )
    for groups in (8, 12, 18):
        section = emergent[groups].reshape(count, count, count).astype(float)
        section[section == groups] = np.nan
        heights = (27.0, 15.0, 3.0, -9.0, -21.0, -33.0)
        figure, axes = plt.subplots(2, len(heights), figsize=(3.1 * len(heights), 6.6))
        for column, height in enumerate(heights):
            layer = int(np.argmin(np.abs(centres - height)))
            axes[0][column].imshow(
                section[:, :, layer].T,
                origin="lower",
                cmap="tab20",
                vmin=0,
                vmax=19,
                extent=(-EDGE, EDGE, -EDGE, EDGE),
                interpolation="nearest",
            )
            axes[0][column].set_title(f"axial, {height:+.0f} mm", fontsize=9)
            axes[0][column].set_xlabel("lateral  <-  medial (mm)", fontsize=7)
            axes[0][column].set_ylabel("back  <-  front (mm)", fontsize=7)
        for column, (depth, name, axis) in enumerate(
            (
                (-15.0, "coronal, 15 mm behind", 1),
                (3.0, "coronal, centre", 1),
                (21.0, "coronal, 21 mm in front", 1),
                (-27.0, "sagittal, lateral", 0),
                (-3.0, "sagittal, middle", 0),
                (27.0, "sagittal, medial", 0),
            )
        ):
            layer = int(np.argmin(np.abs(centres - depth)))
            image = section[:, layer, :] if axis == 1 else section[layer, :, :]
            axes[1][column].imshow(
                image.T,
                origin="lower",
                cmap="tab20",
                vmin=0,
                vmax=19,
                extent=(-EDGE, EDGE, -EDGE, EDGE),
                interpolation="nearest",
            )
            axes[1][column].set_title(name, fontsize=9)
            axes[1][column].set_xlabel(
                "lateral <- medial" if axis == 1 else "back <- front", fontsize=7
            )
            axes[1][column].set_ylabel("down <- up (mm)", fontsize=7)
        for panel in axes.flat:
            panel.axhline(0, color="black", linewidth=0.4)
            panel.axvline(0, color="black", linewidth=0.4)
            panel.tick_params(labelsize=6)
        figure.suptitle(
            f"Sections that emerge from the free heads' most attended tokens ({groups} sections, {frame_name}). One colour per section; white = hardly attended.",
            fontsize=11,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        figure.savefig(
            OUT
            / f"emergent-{groups}-{'geometric' if frame_name.startswith('geo') else 'spots'}.png",
            dpi=85,
        )
        plt.close(figure)
    if frame_name.startswith("geo"):
        shown = [*labels, *(name for name in names[len(labels) :])]
        for part, chunk in enumerate(np.array_split(np.arange(len(shown)), 4)):
            figure, axes = plt.subplots(
                len(chunk), 3, figsize=(7.8, 2.45 * len(chunk)), squeeze=False
            )
            for line, head in enumerate(chunk):
                volume = density[names.index(shown[head])].reshape(count, count, count)
                for column, (axis, xlabel, ylabel) in enumerate(
                    (
                        (2, "lateral <- medial", "back <- front"),
                        (1, "lateral <- medial", "down <- up"),
                        (0, "back <- front", "down <- up"),
                    )
                ):
                    panel = axes[line][column]
                    panel.imshow(
                        volume.sum(axis).T,
                        origin="lower",
                        cmap="magma",
                        extent=(-EDGE, EDGE, -EDGE, EDGE),
                        interpolation="nearest",
                    )
                    panel.axhline(0, color="white", linewidth=0.4)
                    panel.axvline(0, color="white", linewidth=0.4)
                    panel.set_xlabel(xlabel, fontsize=6)
                    panel.set_ylabel(ylabel, fontsize=6)
                    panel.tick_params(labelsize=5)
                tag = (
                    "" if free[names.index(shown[head])] else " (routed in this model)"
                )
                axes[line][0].set_title(
                    f"{shown[head].removeprefix('report:')}{tag}, {knees[names.index(shown[head])]} knees",
                    fontsize=8,
                    loc="left",
                )
            figure.suptitle(
                "Where each head's ten most attended tokens fall (geometric frame; axial, coronal and sagittal views)",
                fontsize=10,
            )
            figure.tight_layout(rect=(0, 0, 1, 0.985))
            figure.savefig(OUT / f"heads-{part + 1}.png", dpi=80)
            plt.close(figure)
(OUT / "shapes.json").write_text(json.dumps(summary, indent=1))
