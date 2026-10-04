# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/learnable-views/findings.md` (not tracked).

Incumbent: `hand-views-frozen-with-label-bias-20261003T085657Z-6dd457aa`

- `two-views-sagittal-fluid`: reject, macro AUC delta -0.000274, adoption 80% CI [-0.001777, +0.001179], soft AUC delta +0.000275
- `two-views-every-type`: reject, macro AUC delta -0.000116, adoption 80% CI [-0.001558, +0.001296], soft AUC delta +0.000524
- `knee-anchored-learned-offsets`: reject, macro AUC delta -0.000497, adoption 80% CI [-0.001643, +0.000633], soft AUC delta +0.000336
- `knee-frame-3d-position`: reject, macro AUC delta -0.000412, adoption 80% CI [-0.001900, +0.001004], soft AUC delta +0.000605
- `learned-centre-fit`: reject, macro AUC delta +0.000616, adoption 80% CI [-0.000466, +0.001745], soft AUC delta +0.000479
- `knee-frame-3d-position-with-1d-2d`: reject, macro AUC delta +0.000507, adoption 80% CI [-0.000687, +0.001670], soft AUC delta +0.000783
- `knee-frame-3d-position-ema`: reject, macro AUC delta -0.000705, adoption 80% CI [-0.001955, +0.000550], soft AUC delta +0.000373
- `computed-centre-3d-position`: reject, macro AUC delta -0.000120, adoption 80% CI [-0.001516, +0.001275], soft AUC delta +0.000604
- `knee-frame-3d-position-centre-long-average`: reject, macro AUC delta -0.001731, adoption 80% CI [-0.003163, -0.000406], soft AUC delta +0.000113
- `knee-frame-3d-position-centre-noise`: reject, macro AUC delta -0.000737, adoption 80% CI [-0.002300, +0.000724], soft AUC delta +0.001023
- `learned-slice-spread`: reject, macro AUC delta -0.002414, adoption 80% CI [-0.004339, -0.000540], soft AUC delta +0.000351
- `knee-frame-3d-position-first-order-oa-detector`: reject, macro AUC delta +0.000955, adoption 80% CI [-0.000381, +0.002322], soft AUC delta +0.000702
- `views-learned-on-held-out-studies`: reject, macro AUC delta -0.002400, adoption 80% CI [-0.003989, -0.000856], soft AUC delta -0.000956
- `view-optimizer-long-average`: reject, macro AUC delta -0.000494, adoption 80% CI [-0.001710, +0.000767], soft AUC delta +0.000460
- `views-start-late`: reject, macro AUC delta -0.000646, adoption 80% CI [-0.001845, +0.000501], soft AUC delta +0.000689
- `label-bias-off`: reject, macro AUC delta -0.001686, adoption 80% CI [-0.003002, -0.000400], soft AUC delta -0.000076
- `normal-jitter`: reject, macro AUC delta -0.000765, adoption 80% CI [-0.002158, +0.000595], soft AUC delta +0.000547
- `views-only-fixed-budget`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `start-from-hand-recipe`: reject, macro AUC delta -0.000761, adoption 80% CI [-0.002158, +0.000558], soft AUC delta +0.000391
- `views-only-fixed-budget`: reject, macro AUC delta -0.000547, adoption 80% CI [-0.001918, +0.000869], soft AUC delta +0.000479
- `views-frozen-late`: reject, macro AUC delta -0.000183, adoption 80% CI [-0.001415, +0.000994], soft AUC delta +0.000187
- `learned-centre-soft-boundaries`: reject, macro AUC delta -0.000568, adoption 80% CI [-0.001894, +0.000710], soft AUC delta +0.001047
- `sagittal-fluid-ordered-view-widths`: reject, macro AUC delta +0.000419, adoption 80% CI [-0.000819, +0.001620], soft AUC delta +0.000727
- `first-order-oa-detector-image-views`: reject, macro AUC delta -0.000522, adoption 80% CI [-0.001668, +0.000586], soft AUC delta +0.000098
- `slice-budget-all-token-density`: reject, macro AUC delta -0.000936, adoption 80% CI [-0.002283, +0.000378], soft AUC delta +0.000381
- `sagittal-fluid-evidence-conditioned-routing`: reject, macro AUC delta +0.000654, adoption 80% CI [-0.000621, +0.001862], soft AUC delta +0.000967
- `learned-centre-with-slice-spread`: reject, macro AUC delta -0.001009, adoption 80% CI [-0.002392, +0.000312], soft AUC delta +0.000333
- `view-clipping-feasibility-gradient`: reject, macro AUC delta -0.000855, adoption 80% CI [-0.002150, +0.000473], soft AUC delta -0.000151
- `ordered-sagittal-pair-with-learned-depth-spread`: reject, macro AUC delta -0.000155, adoption 80% CI [-0.001411, +0.001054], soft AUC delta +0.000932
- `learned-centre-with-slice-spread-widen-only`: reject, macro AUC delta -0.000052, adoption 80% CI [-0.001339, +0.001105], soft AUC delta -0.000046
- `view-learning-through-photometric-jacobians`: reject, macro AUC delta +0.000408, adoption 80% CI [-0.000732, +0.001579], soft AUC delta +0.000308
- `learned-centre-spread-with-reference-mixture`: reject, macro AUC delta -0.002084, adoption 80% CI [-0.003638, -0.000484], soft AUC delta +0.000481
- `view-clipping-inward-image-gradient`: reject, macro AUC delta -0.000943, adoption 80% CI [-0.002206, +0.000355], soft AUC delta -0.000127
- `ordered-sagittal-pair-with-complementary-depth-routing`: reject, macro AUC delta -0.001420, adoption 80% CI [-0.002782, -0.000196], soft AUC delta +0.000082
- `learned-per-axis-anchor-interpolation`: reject, macro AUC delta -0.000278, adoption 80% CI [-0.001373, +0.000812], soft AUC delta +0.000329
- `oa-detector-3d-with-sagittal-evidence-routing`: reject, macro AUC delta -0.000436, adoption 80% CI [-0.001907, +0.000974], soft AUC delta +0.000182
- `hand-views-frozen-with-label-bias`: provisional_accept (original verdict: reject), macro AUC delta +0.000818, adoption 80% CI [-0.000691, +0.002342], soft AUC delta +0.000729
- `hand-views-sagittal-fluid-pair-learned-budget`: reject, macro AUC delta -0.001955, adoption 80% CI [-0.003431, -0.000586], soft AUC delta -0.000409
- `hand-views-learned-budget`: reject, macro AUC delta -0.001623, adoption 80% CI [-0.002960, -0.000418], soft AUC delta -0.000434
- `hand-views-label-bias-faster-rate`: reject, macro AUC delta +0.000082, adoption 80% CI [-0.000417, +0.000557], soft AUC delta +0.000261
- `hand-sizes-learned-shifts`: reject, macro AUC delta -0.000401, adoption 80% CI [-0.001391, +0.000541], soft AUC delta +0.000170
- `hand-start-views-held-early`: reject, macro AUC delta -0.000342, adoption 80% CI [-0.001413, +0.000773], soft AUC delta -0.000088
- `view-size-menu-bias-guided`: reject, macro AUC delta +0.000349, adoption 80% CI [-0.001025, +0.001708], soft AUC delta -0.000068
- `hand-views-training-randomness`: reject, macro AUC delta -0.000880, adoption 80% CI [-0.002151, +0.000257], soft AUC delta +0.000550
- `hand-views-patellofemoral-axial-zoom`: reject, macro AUC delta -0.001290, adoption 80% CI [-0.002663, -0.000082], soft AUC delta -0.000155
- `hand-views-label-depth-bias`: reject, macro AUC delta +0.000270, adoption 80% CI [-0.000115, +0.000632], soft AUC delta +0.000074
- `learned-centre-with-protected-strata-depth-transport`: reject, macro AUC delta +0.000047, adoption 80% CI [-0.001394, +0.001440], soft AUC delta +0.000397
