# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/mambavision-settings/findings.md` (not tracked).

Incumbent: `late-anchor-follows-rate-on-linear-decay-20261014T095230Z-c63b90a0`

- `stage-rates-by-weight-scale`: reject, macro AUC delta x4 -0.002762, x1 -0.001544, 80% CI of x4 [-0.004675, -0.000919], soft AUC delta -0.003082
- `backbone-rate-2e-4`: provisional_accept, macro AUC delta x4 +0.002441, x1 +0.004004, 80% CI of x4 [+0.001044, +0.003780], soft AUC delta +0.001653
- `anchors-by-weight-scale`: provisional_accept, macro AUC delta x4 +0.010086, x1 +0.009323, 80% CI of x4 [+0.008250, +0.011935], soft AUC delta +0.007914
- `backbone-rate-4e-4`: reject, macro AUC delta x4 +0.000352, x1 -0.000343, 80% CI of x4 [-0.000931, +0.001514], soft AUC delta -0.000568
- `twelve-epochs`: provisional_accept, macro AUC delta x4 +0.001530, x1 +0.002300, 80% CI of x4 [+0.000437, +0.002629], soft AUC delta +0.001537
- `stage-rates-on-freed-anchors`: reject, macro AUC delta x4 -0.000324, x1 +0.000022, 80% CI of x4 [-0.001684, +0.001033], soft AUC delta -0.000725
- `feature-statistics-augmentation-off`: reject, macro AUC delta x4 -0.001107, x1 -0.001304, 80% CI of x4 [-0.002457, +0.000383], soft AUC delta -0.000491
- `gradient-clipping-at-10`: reject, macro AUC delta x4 -0.000334, x1 -0.000251, 80% CI of x4 [-0.001451, +0.000727], soft AUC delta -0.000172
- `late-anchor-follows-learning-rate`: provisional_accept, macro AUC delta x4 +0.001447, x1 +0.000054, 80% CI of x4 [+0.000462, +0.002492], soft AUC delta +0.000690
- `twelve-epoch-linear-decay`: provisional_accept, macro AUC delta x4 +0.001020, x1 +0.001192, 80% CI of x4 [-0.000078, +0.002234], soft AUC delta +0.000565
- `feature-statistics-after-first-epoch`: reject, macro AUC delta x4 -0.000320, x1 -0.002331, 80% CI of x4 [-0.001435, +0.000748], soft AUC delta +0.000010
- `two-epoch-warmup-on-weight-scale-anchors`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `late-anchor-follows-rate-on-linear-decay`: provisional_accept (original verdict: reject), macro AUC delta x4 +0.000593, x1 -0.001946, 80% CI of x4 [-0.000418, +0.001691], soft AUC delta +0.000422
- `anchors-per-stage-by-weight-scale`: reject, macro AUC delta x4 -0.000711, x1 +0.002171, 80% CI of x4 [-0.001863, +0.000420], soft AUC delta -0.000280
- `stochastic-depth-0-1-on-a-fitted-model`: reject, macro AUC delta x4 -0.000916, x1 +0.001138, 80% CI of x4 [-0.002489, +0.000571], soft AUC delta -0.000034
- `two-warm-up-epochs`: reject, macro AUC delta x4 -0.000428, x1 +0.001342, 80% CI of x4 [-0.001725, +0.000759], soft AUC delta -0.000135
- `head-rate-twice`: reject, macro AUC delta x4 -0.001062, x1 +0.000429, 80% CI of x4 [-0.002413, +0.000240], soft AUC delta -0.000419
