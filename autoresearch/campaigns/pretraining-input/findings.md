# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/pretraining-input/findings.md` (not tracked).

Incumbent: `no-contrast-mixing-20261005T012244Z-811efd8a`

- `loader-upper-quantile-099`: reject, macro AUC delta +0.000439, adoption 80% CI [-0.000048, +0.000911], soft AUC delta +0.000369
- `loader-upper-quantile-098`: reject, macro AUC delta +0.000420, adoption 80% CI [-0.000230, +0.001087], soft AUC delta +0.000195
- `heated-weights`: reject, macro AUC delta +0.001039, adoption 80% CI [-0.000650, +0.002647], soft AUC delta +0.001549
- `slice-upper-quantile-098`: provisional_accept, macro AUC delta +0.000716, adoption 80% CI [+0.000009, +0.001413], soft AUC delta +0.000478
- `autoaugment-intensity`: reject, macro AUC delta -0.002907, adoption 80% CI [-0.004085, -0.001671], soft AUC delta -0.002114
- `fluid-type-upper-quantile`: reject, macro AUC delta -0.000247, adoption 80% CI [-0.000703, +0.000205], soft AUC delta -0.000276
- `other-type-upper-quantile`: reject, macro AUC delta -0.000217, adoption 80% CI [-0.000477, +0.000031], soft AUC delta -0.000127
- `slice-quantile-other-types`: reject, macro AUC delta -0.000952, adoption 80% CI [-0.001677, -0.000240], soft AUC delta -0.000526
- `fluid-brightness`: reject, macro AUC delta -0.001079, adoption 80% CI [-0.001827, -0.000349], soft AUC delta -0.000734
- `no-bilateral-smoothing`: reject, macro AUC delta +0.000902, adoption 80% CI [-0.000073, +0.001923], soft AUC delta +0.000517
- `slice-quantile-soft-knee`: reject, macro AUC delta +0.000025, adoption 80% CI [-0.000266, +0.000314], soft AUC delta +0.000128
- `no-contrast-mixing`: provisional_accept, macro AUC delta +0.001952, adoption 80% CI [+0.000994, +0.002938], soft AUC delta +0.001184
- `no-gin`: reject, macro AUC delta -0.000560, adoption 80% CI [-0.001830, +0.000706], soft AUC delta -0.000149
- `no-bilateral-smoothing-on-no-mixing`: reject, macro AUC delta -0.001948, adoption 80% CI [-0.002988, -0.000937], soft AUC delta -0.001416
- `series-quantile-stretch`: reject, macro AUC delta -0.000560, adoption 80% CI [-0.001196, +0.000060], soft AUC delta -0.000527
- `rotation-30`: reject, macro AUC delta -0.001765, adoption 80% CI [-0.002582, -0.000974], soft AUC delta -0.001150
- `autoaugment-intensity-025`: reject, macro AUC delta -0.003720, adoption 80% CI [-0.005028, -0.002470], soft AUC delta -0.001842
- `study-cutmix`: reject, macro AUC delta -0.003917, adoption 80% CI [-0.005064, -0.002773], soft AUC delta -0.002856
- `union-bag-loss`: reject, macro AUC delta -0.003706, adoption 80% CI [-0.005447, -0.001899], soft AUC delta -0.002935
- `input-mask-centred-one`: reject, macro AUC delta -0.002893, adoption 80% CI [-0.004008, -0.001777], soft AUC delta -0.001876
- `heated-weights-mask-release`: reject, macro AUC delta -0.002152, adoption 80% CI [-0.003888, -0.000445], soft AUC delta -0.000622
- `split-knee-bag-loss`: reject, macro AUC delta -0.004919, adoption 80% CI [-0.006380, -0.003556], soft AUC delta -0.002141
- `split-knee-bag-average`: reject, macro AUC delta -0.002651, adoption 80% CI [-0.003977, -0.001359], soft AUC delta -0.001331
- `split-knee-bag-or`: reject, macro AUC delta -0.003378, adoption 80% CI [-0.004968, -0.001745], soft AUC delta -0.002359
- `front-back-bag-loss`: reject, macro AUC delta -0.004358, adoption 80% CI [-0.006103, -0.002634], soft AUC delta -0.003179
- `transplant-bag-loss`: reject, macro AUC delta -0.003555, adoption 80% CI [-0.005228, -0.001847], soft AUC delta -0.002417
- `learned-mask-policy`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `incumbent-rerun`: reject, macro AUC delta -0.000119, adoption 80% CI [-0.000346, +0.000124], soft AUC delta -0.000042
- `slice-quantile-keep-axial-fluid`: reject, macro AUC delta -0.000370, adoption 80% CI [-0.000980, +0.000236], soft AUC delta +0.000005
- `heated-weights-on-incumbent`: reject, macro AUC delta -0.001669, adoption 80% CI [-0.003376, -0.000016], soft AUC delta -0.000312
- `slice-quantile-keep-sagittal-fluid`: reject, macro AUC delta -0.000934, adoption 80% CI [-0.001449, -0.000422], soft AUC delta -0.000369
- `heated-teacher-distillation`: reject, macro AUC delta +0.000105, adoption 80% CI [-0.000486, +0.000680], soft AUC delta +0.000187
- `slice-window-channels`: reject, macro AUC delta -0.001046, adoption 80% CI [-0.001748, -0.000385], soft AUC delta -0.000555
