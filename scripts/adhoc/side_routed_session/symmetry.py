from pathlib import Path

import numpy as np

ROOT = Path("outputs/analysis/side-routing-attention-shape-20261006")
data = np.load(ROOT / "attention.npz")
density = np.load(ROOT / "sections-geometric.npz")["density"]
names = [str(name) for name in data["names"]]
truth = data["truth"]
labels = [str(name) for name in data["labels"]]
weight = data["weight"] / data["weight"].sum(-1, keepdims=True)
fitted = data["fitted"]
count = round(density.shape[1] ** (1 / 3))
volume = lambda name: density[names.index(name)].reshape(count, count, count)
for medial, lateral in (("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA"), ("report:medial_meniscus_status", "report:lateral_meniscus_status"), ("report:medial_cartilage_depth", "report:lateral_cartilage_depth")):
    a, b = volume(medial), volume(lateral)[::-1]
    best = max((float(np.corrcoef(a.ravel(), np.roll(b, shift, 0).ravel())[0, 1]), shift) for shift in range(-3, 4))
    spots = []
    for name in (medial, lateral):
        head = names.index(name)
        chosen = truth[:, labels.index(name)] if name in labels else np.ones(len(truth), dtype=bool)
        spots.append((weight[chosen, head, :, None] * fitted[chosen, head]).sum(1).mean(0))
    print(f"{medial:32} vs mirrored {lateral:32} map correlation {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.2f}; best {best[0]:.2f} with the lateral map moved {6 * best[1]:+d} mm; spots {spots[0].round(1)} and {spots[1].round(1)}")
acl = names.index("ACL")
print("ACL spot", (weight[:, acl, :, None] * fitted[:, acl]).sum(1).mean(0).round(1), " n medial/lateral meniscus", int(truth[:, labels.index("Medial Meniscus")].sum()), int(truth[:, labels.index("Lateral Meniscus")].sum()))
