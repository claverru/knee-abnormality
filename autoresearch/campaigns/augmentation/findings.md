# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/augmentation/findings.md` (not tracked).

Incumbent: `dsu-feature-statistics-20260926T034836Z-6298ea10`

- `phase-axis-resolution-degradation`: reject, macro AUC delta +0.001191, adoption 80% CI [-0.000783, +0.003200], soft AUC delta +0.001060
- `unsharp-sharpening-augmentation`: reject, macro AUC delta +0.000436, adoption 80% CI [-0.001558, +0.002471], soft AUC delta +0.000200
- `aligned-image-mixup`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `token-feature-mixup`: reject, macro AUC delta -0.000112, adoption 80% CI [-0.002210, +0.002010], soft AUC delta +0.000163
- `pooled-output-mixup`: reject, macro AUC delta +0.000704, adoption 80% CI [-0.001525, +0.002836], soft AUC delta +0.001860
- `late-affine-shutdown`: reject, macro AUC delta -0.000545, adoption 80% CI [-0.002137, +0.001034], soft AUC delta +0.000878
- `smooth-bias-field-augmentation`: reject, macro AUC delta +0.000571, adoption 80% CI [-0.001339, +0.002414], soft AUC delta +0.000933
- `triplet-channel-reversal`: reject, macro AUC delta +0.001159, adoption 80% CI [-0.000943, +0.003125], soft AUC delta +0.000306
- `bilateral-smoothing-augmentation`: provisional_accept, macro AUC delta +0.002039, adoption 80% CI [+0.000184, +0.003854], soft AUC delta +0.000618
- `intra-study-contrast-mixing`: reject, macro AUC delta +0.001503, adoption 80% CI [-0.000539, +0.003475], soft AUC delta +0.001555
- `dsu-feature-statistics`: provisional_accept (original verdict: reject), macro AUC delta +0.002155, adoption 80% CI [-0.000478, +0.004619], soft AUC delta +0.004300
- `rexnet-stochastic-depth`: reject, macro AUC delta +0.001078, adoption 80% CI [-0.000775, +0.002970], soft AUC delta -0.000352
- `aligned-image-mixup-rerun`: reject, macro AUC delta +0.000584, adoption 80% CI [-0.001699, +0.002875], soft AUC delta -0.002351
- `correlated-style-uncertainty`: reject, macro AUC delta +0.000711, adoption 80% CI [-0.001376, +0.002965], soft AUC delta -0.001898
- `study-coherent-feature-statistics`: reject, macro AUC delta +0.001254, adoption 80% CI [-0.001043, +0.003703], soft AUC delta -0.002012
- `gin-intensity-augmentation-v3`: reject, macro AUC delta +0.001332, adoption 80% CI [-0.001163, +0.003813], soft AUC delta -0.001585
- `noise-referenced-bilateral-smoothing`: reject, macro AUC delta +0.001813, adoption 80% CI [-0.000342, +0.004002], soft AUC delta -0.001564
