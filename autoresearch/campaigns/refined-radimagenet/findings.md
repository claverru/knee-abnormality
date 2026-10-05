# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/refined-radimagenet/findings.md` (not tracked).

Incumbent: `radimagenet-input-statistics-20261004T103433Z-ff2f3786`

- `radimagenet-input-statistics`: provisional_accept, macro AUC delta +0.000581, adoption 80% CI [+0.000182, +0.001016], soft AUC delta +0.000335
- `rn10t-policy-weights`: reject, macro AUC delta -0.004281, adoption 80% CI [-0.006272, -0.002393], soft AUC delta -0.002850
- `rn10t-policy-ft-weights`: reject, macro AUC delta -0.004387, adoption 80% CI [-0.006357, -0.002488], soft AUC delta -0.002960
- `backbone-eval-pretrained-statistics`: reject, macro AUC delta -0.003119, adoption 80% CI [-0.004573, -0.001732], soft AUC delta -0.002514
- `policy-mask-rn10t`: reject, macro AUC delta -0.002292, adoption 80% CI [-0.003594, -0.001087], soft AUC delta -0.001916
- `policy-mask-two-stage`: reject, macro AUC delta -0.001733, adoption 80% CI [-0.002623, -0.000827], soft AUC delta -0.001134
- `policy-mask-pretrained-policy`: reject, macro AUC delta -0.002590, adoption 80% CI [-0.003837, -0.001417], soft AUC delta -0.002489
- `random-smooth-mask`: reject, macro AUC delta -0.002325, adoption 80% CI [-0.003573, -0.001116], soft AUC delta -0.002279
- `knee-logit-token-features`: reject, macro AUC delta -0.001496, adoption 80% CI [-0.002371, -0.000728], soft AUC delta -0.000946
- `knee-classifier-readout`: reject, macro AUC delta +0.000048, adoption 80% CI [-0.000351, +0.000442], soft AUC delta +0.000113
- `pretrained-slice-teacher`: reject, macro AUC delta +0.000333, adoption 80% CI [-0.000588, +0.001210], soft AUC delta +0.000229
- `knee-logit-attention-prior`: reject, macro AUC delta +0.000297, adoption 80% CI [-0.000114, +0.000679], soft AUC delta +0.000183
- `crossd-resnet18-drop-in`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `crossd-3d-stem`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `crossd-static-kernels`: reject, macro AUC delta -0.003903, adoption 80% CI [-0.005739, -0.002093], soft AUC delta -0.003802
- `crossd-gated-depth-stem`: reject, macro AUC delta -0.004200, adoption 80% CI [-0.006117, -0.002313], soft AUC delta -0.003866
- `acs-stem-layer1`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `acs-gated-stem-block1`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `acs-gated-stem`: reject, macro AUC delta +0.000027, adoption 80% CI [-0.000328, +0.000374], soft AUC delta +0.000080
- `knee-normal-contrast-attention`: reject, macro AUC delta +0.000095, adoption 80% CI [-0.000229, +0.000396], soft AUC delta -0.000040
