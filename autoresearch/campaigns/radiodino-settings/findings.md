# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/radiodino-settings/findings.md` (not tracked).

Incumbent: `top-down-backbone-warm-up-20261015T034829Z-c8b4e10b`

- `radiodino-on-mamba-recipe`: provisional_accept (original verdict: reject), macro AUC delta x4 -0.008581, x1 -0.006238, 80% CI of x4 [-0.010865, -0.006304], soft AUC delta -0.006135
- `layer-rate-decay-from-radimagenet-profile`: reject, macro AUC delta x4 -0.000010, x1 +0.001549, 80% CI of x4 [-0.001221, +0.001196], soft AUC delta +0.000173
- `anchors-by-weight-scale`: reject, macro AUC delta x4 -0.002436, x1 -0.000067, 80% CI of x4 [-0.003585, -0.001292], soft AUC delta -0.001476
- `feature-statistics-before-position-embedding`: reject, macro AUC delta x4 -0.000385, x1 +0.000353, 80% CI of x4 [-0.001339, +0.000601], soft AUC delta -0.000817
- `stage-rates-4e-4-top-to-2e-4-stem`: reject, macro AUC delta x4 -0.000164, x1 +0.000822, 80% CI of x4 [-0.001284, +0.001028], soft AUC delta +0.000168
- `backbone-held-for-the-head-epoch`: reject, macro AUC delta x4 +0.000060, x1 -0.000027, 80% CI of x4 [-0.001331, +0.001445], soft AUC delta +0.000087
- `top-down-backbone-warm-up`: provisional_accept, macro AUC delta x4 +0.000422, x1 +0.000614, 80% CI of x4 [-0.000555, +0.001392], soft AUC delta +0.000525
- `backbone-rate-1-4e-4-by-relative-step`: reject, macro AUC delta x4 -0.000455, x1 -0.000532, 80% CI of x4 [-0.001361, +0.000497], soft AUC delta -0.000784
- `position-embedding-rate-by-weight-scale`: reject, macro AUC delta x4 -0.000549, x1 +0.000188, 80% CI of x4 [-0.001443, +0.000340], soft AUC delta -0.000696
- `middle-read-at-block-10`: reject, macro AUC delta x4 -0.000515, x1 +0.000292, 80% CI of x4 [-0.001511, +0.000518], soft AUC delta -0.000326
- `final-read-by-learned-two-by-two-merge`: reject, macro AUC delta x4 -0.001885, x1 -0.001319, 80% CI of x4 [-0.003148, -0.000671], soft AUC delta -0.001784
- `early-anchor-by-gradient-scale`: reject, macro AUC delta x4 -0.000664, x1 -0.001351, 80% CI of x4 [-0.001559, +0.000219], soft AUC delta -0.000479
- `stochastic-depth-0-1`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `convolutional-spatial-prior-injection`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `blockwise-top-down-warm-up`: reject, macro AUC delta x4 -0.000187, x1 +0.000077, 80% CI of x4 [-0.001156, +0.000754], soft AUC delta -0.000352
- `convolutional-spatial-prior-frozen-branch`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `convolutional-spatial-prior-frozen-branch`: reject, macro AUC delta x4 -0.000144, x1 -0.000318, 80% CI of x4 [-0.000997, +0.000690], soft AUC delta -0.000528
- `overlapping-patch-embedding`: reject, macro AUC delta x4 -0.001604, x1 -0.001168, 80% CI of x4 [-0.002746, -0.000504], soft AUC delta -0.001938
- `depthwise-convolution-in-block-mlps`: reject, macro AUC delta x4 -0.000026, x1 +0.001180, 80% CI of x4 [-0.000926, +0.000830], soft AUC delta -0.000738
