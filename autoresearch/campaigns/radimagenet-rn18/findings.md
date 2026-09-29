# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/radimagenet-rn18/findings.md` (not tracked).

Incumbent: `rn18-more-triplets-20260928T235020Z-fd2c7bef`

- `rn18-no-series-dropout`: provisional_accept, macro AUC delta +0.002330, adoption 80% CI [+0.000897, +0.003798], soft AUC delta +0.002037
- `rn18-lighter-dsu`: reject, macro AUC delta +0.000052, adoption 80% CI [-0.000679, +0.000745], soft AUC delta -0.000193
- `rn18-layerwise-lr-decay`: reject, macro AUC delta -0.000568, adoption 80% CI [-0.001661, +0.000515], soft AUC delta -0.000799
- `rn18-linear-probe-then-finetune`: reject, macro AUC delta -0.001244, adoption 80% CI [-0.003147, +0.000477], soft AUC delta -0.001435
- `rn18-l2-sp`: reject, macro AUC delta +0.000626, adoption 80% CI [-0.000322, +0.001631], soft AUC delta +0.000007
- `rn18-mean-init-adjacent-slices`: reject, macro AUC delta -0.003869, adoption 80% CI [-0.005988, -0.001954], soft AUC delta -0.002712
- `rn18-freeze-stem-layer1`: reject, macro AUC delta -0.002890, adoption 80% CI [-0.004411, -0.001384], soft AUC delta -0.003170
- `rn18-more-triplets`: provisional_accept, macro AUC delta +0.001780, adoption 80% CI [+0.000237, +0.003360], soft AUC delta +0.000456
- `rn18-backbone-lr-1e-4`: reject, macro AUC delta -0.001720, adoption 80% CI [-0.003233, -0.000168], soft AUC delta -0.001342
- `rn18-multiscale-tokens`: reject, macro AUC delta -0.002260, adoption 80% CI [-0.003408, -0.001083], soft AUC delta -0.001488
- `rn18-warmup-stable-decay`: reject, macro AUC delta -0.000252, adoption 80% CI [-0.001521, +0.001071], soft AUC delta +0.000007
- `rn10t-double-batch-and-rates`: reject, macro AUC delta -0.003689, adoption 80% CI [-0.006099, -0.001439], soft AUC delta -0.002932
- `rn18-plane-backbones-dense-fusion`: reject, macro AUC delta -0.003292, adoption 80% CI [-0.005205, -0.001348], soft AUC delta -0.002194
- `asymmetric-focal-loss`: reject, macro AUC delta -0.002434, adoption 80% CI [-0.003696, -0.001192], soft AUC delta -0.001825
- `soft-label-101-bins`: reject, macro AUC delta -0.013902, adoption 80% CI [-0.017378, -0.010251], soft AUC delta -0.004512
- `rn18-native-contrast-preservation`: reject, macro AUC delta +0.000551, adoption 80% CI [-0.001057, +0.002199], soft AUC delta +0.000861
- `rn18-early-stage-anchor`: reject, macro AUC delta -0.000353, adoption 80% CI [-0.000991, +0.000265], soft AUC delta +0.000124
- `rn18-forty-triplets-nine-epochs`: reject, macro AUC delta -0.001533, adoption 80% CI [-0.003129, +0.000069], soft AUC delta -0.000953
- `rn18-bn-adapted-head-start`: reject, macro AUC delta -0.000398, adoption 80% CI [-0.001805, +0.001045], soft AUC delta -0.000148
- `rn18-gated-adjacent-residuals`: reject, macro AUC delta -0.000047, adoption 80% CI [-0.000803, +0.000627], soft AUC delta +0.000126
- `rn18-separate-module-clipping`: reject, macro AUC delta -0.002412, adoption 80% CI [-0.003914, -0.000906], soft AUC delta -0.001373
- `rn18-upper-stage-batch-renorm`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `rn18-full-width-projection`: reject, macro AUC delta -0.001212, adoption 80% CI [-0.002521, +0.000136], soft AUC delta -0.000469
