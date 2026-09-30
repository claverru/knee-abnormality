# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/site-generalization/findings.md` (not tracked).

Incumbent: `site-l2-sp-without-arcface-20260930T102231Z-507e2e97`

- `site-arcface-prototypes`: reject, macro AUC delta -0.000030, adoption 80% CI [-0.001847, +0.001680], soft AUC delta -0.001556
- `site-l2-sp-anchor`: provisional_accept, macro AUC delta +0.001514, adoption 80% CI [+0.000557, +0.002474], soft AUC delta +0.000664
- `site-fourier-amplitude-swap`: reject, macro AUC delta +0.000691, adoption 80% CI [-0.000721, +0.002086], soft AUC delta +0.000218
- `site-cross-mixstyle`: reject, macro AUC delta -0.000984, adoption 80% CI [-0.002492, +0.000536], soft AUC delta -0.001213
- `site-conditional-coral`: reject, macro AUC delta +0.000083, adoption 80% CI [-0.000778, +0.000933], soft AUC delta -0.000444
- `site-augmix-consistency`: reject, macro AUC delta +0.000259, adoption 80% CI [-0.000318, +0.000842], soft AUC delta +0.000204
- `site-confusion-retarget`: reject, macro AUC delta +0.001559, adoption 80% CI [-0.000068, +0.003209], soft AUC delta +0.000095
- `site-group-dro`: reject, macro AUC delta -0.006930, adoption 80% CI [-0.008859, -0.004820], soft AUC delta -0.005450
- `site-arcface-per-label-pooled`: provisional_accept, macro AUC delta +0.001841, adoption 80% CI [+0.000152, +0.003453], soft AUC delta -0.000076
- `site-scanner-subspace-projection`: reject, macro AUC delta -0.000495, adoption 80% CI [-0.001572, +0.000675], soft AUC delta +0.000209
- `site-scanner-subspace-instead-of-arcface`: reject, macro AUC delta -0.000670, adoption 80% CI [-0.002335, +0.001136], soft AUC delta +0.000489
- `site-no-scanner-confusion`: reject, macro AUC delta -0.000098, adoption 80% CI [-0.001243, +0.001052], soft AUC delta +0.000440
- `site-scanner-leace-token-erasure`: reject, macro AUC delta -0.008853, adoption 80% CI [-0.011600, -0.006355], soft AUC delta -0.005352
- `site-arcface-balanced-site-states`: reject, macro AUC delta -0.000363, adoption 80% CI [-0.001549, +0.000976], soft AUC delta +0.000370
- `site-pretrained-feature-direction-anchor`: reject, macro AUC delta -0.001852, adoption 80% CI [-0.002776, -0.000919], soft AUC delta -0.000806
- `site-fourier-matched-shape-exchange`: reject, macro AUC delta +0.000028, adoption 80% CI [-0.001193, +0.001149], soft AUC delta +0.000244
- `site-arcface-antipodal-prototypes`: reject, macro AUC delta -0.000120, adoption 80% CI [-0.001116, +0.000983], soft AUC delta +0.000276
- `site-dro-fresh-bounded-risk`: reject, macro AUC delta -0.001562, adoption 80% CI [-0.002792, -0.000339], soft AUC delta -0.000525
- `site-pretrained-canonical-feature-anchor`: reject, macro AUC delta -0.001060, adoption 80% CI [-0.001835, -0.000223], soft AUC delta -0.000242
- `site-fourier-feasible-residual-exchange`: reject, macro AUC delta -0.001583, adoption 80% CI [-0.002866, -0.000259], soft AUC delta -0.000646
- `site-acquisition-half-channel-normalization`: reject, macro AUC delta +0.000995, adoption 80% CI [-0.000458, +0.002399], soft AUC delta +0.000696
- `site-dro-entropy-corrected-risk`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `site-dro-entropy-corrected-risk`: reject, macro AUC delta -0.000688, adoption 80% CI [-0.001423, +0.000105], soft AUC delta -0.000207
- `site-acquisition-channel-selective-normalization`: reject, macro AUC delta +0.001475, adoption 80% CI [+0.000148, +0.002802], soft AUC delta +0.000371
- `site-arcface-per-label-pooled`: provisional_accept, macro AUC delta -0.002178, adoption 80% CI unavailable
- `site-acquisition-channel-selective-reseed`: reject, macro AUC delta -0.000098, adoption 80% CI [-0.001361, +0.001079], soft AUC delta -0.000790
- `site-acquisition-image-dependent-restoration`: reject, macro AUC delta +0.000475, adoption 80% CI [-0.000777, +0.001645], soft AUC delta -0.000551
- `site-arcface-without-l2-sp`: reject, macro AUC delta -0.001396, adoption 80% CI [-0.002237, -0.000527], soft AUC delta -0.001057
- `site-no-scanner-confusion-reseed`: provisional_accept (original verdict: reject), macro AUC delta -0.000679, adoption 80% CI [-0.001985, +0.000600], soft AUC delta -0.001199
- `site-acquisition-mixture-contrast`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `site-l2-sp-without-arcface`: provisional_accept (original verdict: reject), macro AUC delta +0.000228, adoption 80% CI [-0.001367, +0.001870], soft AUC delta +0.001304
- `site-lp-ft`: reject, macro AUC delta +0.001573, adoption 80% CI [+0.000028, +0.003213], soft AUC delta +0.001488
- `site-acquisition-mixture-contrast-final`: reject, macro AUC delta -0.001487, adoption 80% CI [-0.002861, -0.000052], soft AUC delta -0.000584
- `site-unlearning-only`: reject, macro AUC delta -0.002696, adoption 80% CI [-0.004780, -0.000658], soft AUC delta -0.002471
- `site-scanner-unlearning-grl`: reject, macro AUC delta -0.064674, adoption 80% CI [-0.071135, -0.057925], soft AUC delta -0.044540
- `site-frozen-backbone-batchnorm`: reject, macro AUC delta -0.008621, adoption 80% CI [-0.011164, -0.006200], soft AUC delta -0.005276
- `site-lp-ft-on-l2-sp`: reject, macro AUC delta +0.000518, adoption 80% CI [-0.000663, +0.001755], soft AUC delta +0.000314
- `site-scanner-unlearning-kl`: reject, macro AUC delta -0.000265, adoption 80% CI [-0.001575, +0.001086], soft AUC delta +0.000253
