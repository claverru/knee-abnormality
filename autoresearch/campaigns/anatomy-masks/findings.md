# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/anatomy-masks/findings.md` (not tracked).

Incumbent: `joint-centre-with-equal-view-exposure-20261008T050113Z-9937e29a`

- `turned-crops-cut-with-bicubic-interpolation`: provisional_accept, macro AUC delta +0.001154, adoption 80% CI [+0.000387, +0.001976], soft AUC delta +0.000783
- `joint-centre-from-the-map-instead-of-the-spots`: provisional_accept (original verdict: reject), macro AUC delta +0.000044, adoption 80% CI [-0.000810, +0.000896], soft AUC delta -0.000275
- `a-tenth-of-the-map-gradient-to-the-backbone`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `the-map-loss-without-its-dice-term`: reject, macro AUC delta -0.000335, adoption 80% CI [-0.001151, +0.000513], soft AUC delta -0.000109
- `a-tenth-of-the-map-gradient-to-the-backbone-2`: reject, macro AUC delta -0.000004, adoption 80% CI [-0.001067, +0.001089], soft AUC delta +0.000126
- `the-first-look-on-a-third-of-the-slices`: reject, macro AUC delta -0.001528, adoption 80% CI [-0.002672, -0.000356], soft AUC delta -0.001814
- `fluid-sagittal-crops-moved-forward-for-the-kneecap`: reject, macro AUC delta -0.000491, adoption 80% CI [-0.001254, +0.000284], soft AUC delta -0.000406
- `joint-gap-per-side-as-a-target`: reject, macro AUC delta -0.000582, adoption 80% CI [-0.001430, +0.000243], soft AUC delta -0.000461
- `the-first-look-on-the-whole-canvas-at-224-px`: reject, macro AUC delta -0.000114, adoption 80% CI [-0.001074, +0.000807], soft AUC delta -0.000213
- `sagittal-crops-a-ninth-wider`: reject, macro AUC delta -0.000617, adoption 80% CI [-0.001478, +0.000280], soft AUC delta -0.000320
- `coronal-crops-a-ninth-wider`: reject, macro AUC delta +0.000153, adoption 80% CI [-0.000776, +0.001089], soft AUC delta +0.000054
- `posed-crops-keep-their-subpixel-shifts`: reject, macro AUC delta -0.000885, adoption 80% CI [-0.001523, -0.000215], soft AUC delta -0.000465
- `soft-bone-shares-fit-the-joint-plane`: reject, macro AUC delta +0.000136, adoption 80% CI [-0.000683, +0.000945], soft AUC delta -0.000253
- `map-gradient-projected-away-from-task-conflict`: reject, macro AUC delta +0.000094, adoption 80% CI [-0.000510, +0.000773], soft AUC delta -0.000209
- `patellar-routing-uses-unsaturated-shares`: reject, macro AUC delta +0.000332, adoption 80% CI [-0.000268, +0.000979], soft AUC delta +0.000216
- `joint-centre-with-equal-view-exposure`: provisional_accept (original verdict: reject), macro AUC delta +0.000180, adoption 80% CI [-0.000444, +0.000837], soft AUC delta -0.000156
- `sparse-first-look-at-interior-depth-quantiles`: reject, macro AUC delta +0.000177, adoption 80% CI [-0.000651, +0.001010], soft AUC delta +0.000186
