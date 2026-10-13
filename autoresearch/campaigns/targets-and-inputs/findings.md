# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/targets-and-inputs/findings.md` (not tracked).

Incumbent: `average-pool-shortcuts-in-the-resnet-20261013T122134Z-1fc0acea`

- `soft-prevalence-class-weights`: provisional_accept, macro AUC delta x4 +0.000509, x1 +0.001327, 80% CI of x4 [-0.000222, +0.001285], soft AUC delta +0.000871
- `stable-intensity-bounds`: reject, macro AUC delta x4 -0.000270, x1 +0.000608, 80% CI of x4 [-0.001035, +0.000511], soft AUC delta -0.000057
- `uncertain-report-targets-weigh-less`: reject, macro AUC delta x4 -0.000953, x1 -0.000141, 80% CI of x4 [-0.002044, +0.000157], soft AUC delta +0.000178
- `hard-report-targets`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `mask-decoder-beside-the-labels`: provisional_accept, macro AUC delta x4 +0.000529, x1 +0.001424, 80% CI of x4 [-0.000239, +0.001311], soft AUC delta +0.000389
- `intensity-augmentation-off`: reject, macro AUC delta x4 -0.000381, x1 -0.000914, 80% CI of x4 [-0.001422, +0.000680], soft AUC delta -0.000108
- `noise-augmentation-off`: reject, macro AUC delta x4 +0.000376, x1 -0.000387, 80% CI of x4 [-0.000666, +0.001392], soft AUC delta -0.000029
- `hard-pair-final-phase`: reject, macro AUC delta x4 +0.000199, x1 -0.000665, 80% CI of x4 [-0.000590, +0.001025], soft AUC delta -0.000094
- `plane-contrast-readouts-off`: reject, macro AUC delta x4 +0.000249, x1 -0.000381, 80% CI of x4 [-0.000769, +0.001305], soft AUC delta -0.000308
- `oa-detector-beside-the-shared-head`: provisional_accept, macro AUC delta x4 +0.001089, x1 +0.001362, 80% CI of x4 [+0.000431, +0.001775], soft AUC delta +0.000570
- `oa-detector-instead-of-the-shared-head`: reject, macro AUC delta x4 -0.000652, x1 -0.002039, 80% CI of x4 [-0.001664, +0.000277], soft AUC delta +0.000164
- `report-attribute-loss-on`: reject, macro AUC delta x4 -0.000295, x1 -0.001865, 80% CI of x4 [-0.000977, +0.000329], soft AUC delta -0.000102
- `scanner-confusion-head-beside-l2sp`: reject, macro AUC delta x4 -0.001575, x1 -0.001292, 80% CI of x4 [-0.002398, -0.000713], soft AUC delta -0.000963
- `view-label-bias-off`: reject, macro AUC delta x4 -0.000596, x1 -0.001442, 80% CI of x4 [-0.001303, +0.000130], soft AUC delta -0.000374
- `anatomical-attention-bias-back`: reject, macro AUC delta x4 -0.000615, x1 -0.001570, 80% CI of x4 [-0.001297, +0.000059], soft AUC delta -0.000338
- `sagittal-fluid-detail-view-off`: reject, macro AUC delta x4 -0.001193, x1 -0.002963, 80% CI of x4 [-0.002095, -0.000334], soft AUC delta -0.000979
- `detail-view-embedding-off`: reject, macro AUC delta x4 -0.000512, x1 -0.001195, 80% CI of x4 [-0.001275, +0.000224], soft AUC delta -0.000159
- `gin-augmentation-off`: reject, macro AUC delta x4 +0.001111, x1 -0.001359, 80% CI of x4 [-0.000198, +0.002423], soft AUC delta +0.000699
- `resolution-augmentation-off`: reject, macro AUC delta x4 -0.001044, x1 -0.000687, 80% CI of x4 [-0.002223, +0.000098], soft AUC delta -0.000603
- `mask-decoder-loss-weight-equal-to-the-map-heads`: reject, macro AUC delta x4 -0.000996, x1 -0.001522, 80% CI of x4 [-0.001690, -0.000317], soft AUC delta -0.000718
- `mask-decoder-stage-one-level`: reject, macro AUC delta x4 -0.000543, x1 -0.001122, 80% CI of x4 [-0.001279, +0.000209], soft AUC delta -0.000121
- `squeeze-excitation-in-the-resnet-blocks`: reject, macro AUC delta x4 -0.000998, x1 -0.001638, 80% CI of x4 [-0.001972, -0.000012], soft AUC delta -0.000287
- `global-context-in-the-resnet-blocks`: reject, macro AUC delta x4 -0.000341, x1 -0.000590, 80% CI of x4 [-0.001423, +0.000722], soft AUC delta -0.000351
- `soft-prevalence-class-weights-on-the-incumbent`: reject, macro AUC delta x4 -0.000938, x1 -0.002065, 80% CI of x4 [-0.001728, -0.000183], soft AUC delta +0.000018
- `global-context-pulled-to-identity`: reject, macro AUC delta x4 -0.000216, x1 -0.001119, 80% CI of x4 [-0.001110, +0.000621], soft AUC delta -0.000598
- `coordinate-attention-late-stages-pulled-to-identity`: reject, macro AUC delta x4 -0.000667, x1 -0.001175, 80% CI of x4 [-0.001384, +0.000033], soft AUC delta -0.000501
- `average-pool-shortcuts-in-the-resnet`: provisional_accept (original verdict: reject), macro AUC delta x4 +0.001016, x1 -0.000345, 80% CI of x4 [+0.000172, +0.001885], soft AUC delta +0.000244
- `blur-pool-at-the-resnet-stage-transitions`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `squeeze-excitation-late-stages-pulled-to-identity`: reject, macro AUC delta x4 -0.000994, x1 -0.001310, 80% CI of x4 [-0.001610, -0.000421], soft AUC delta -0.000260
