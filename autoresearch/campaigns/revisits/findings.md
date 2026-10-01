# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/revisits/findings.md` (not tracked).

Incumbent: `no-anatomical-bias-20260930T213145Z-58126164`

- `dense-plane-readouts`: reject, macro AUC delta -0.005248, adoption 80% CI [-0.007138, -0.003362], soft AUC delta -0.003558
- `plane-contrast-readouts`: provisional_accept (original verdict: reject), macro AUC delta +0.000918, adoption 80% CI [+0.000096, +0.001703], soft AUC delta +0.000049
- `no-dsu`: reject, macro AUC delta -0.001334, adoption 80% CI [-0.002651, -0.000054], soft AUC delta -0.000218
- `no-anatomical-bias`: provisional_accept (original verdict: reject), macro AUC delta +0.000855, adoption 80% CI [+0.000185, +0.001569], soft AUC delta +0.000351
- `no-bilateral-smoothing`: reject, macro AUC delta -0.000026, adoption 80% CI [-0.001087, +0.001069], soft AUC delta +0.000046
- `no-resolution-degradation`: reject, macro AUC delta -0.000452, adoption 80% CI [-0.001489, +0.000552], soft AUC delta -0.000062
- `plane-contrast-feature-adapters`: reject, macro AUC delta -0.000393, adoption 80% CI [-0.001032, +0.000224], soft AUC delta -0.000144
- `plane-conditioned-batchnorm`: reject, macro AUC delta -0.001007, adoption 80% CI [-0.001787, -0.000170], soft AUC delta -0.000410
- `rn18-stochastic-depth`: reject, macro AUC delta -0.003490, adoption 80% CI [-0.004684, -0.002428], soft AUC delta -0.002744
- `ema-0999`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `ema-09995`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `ema-0999`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `ema-09995`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `rn18-plane-head-lr-3e-4`: reject, macro AUC delta -0.000319, adoption 80% CI [-0.000901, +0.000258], soft AUC delta -0.000250
- `ema-0999-subset-refresh`: reject, macro AUC delta -0.000369, adoption 80% CI [-0.000965, +0.000218], soft AUC delta -0.000345
- `ema-0999-buffer-average`: reject, macro AUC delta -0.000370, adoption 80% CI [-0.000974, +0.000222], soft AUC delta -0.000371
- `ema-09995-buffer-average`: reject, macro AUC delta -0.003194, adoption 80% CI [-0.004419, -0.001952], soft AUC delta -0.002365
- `rn18-bias-free-half-knee-00375`: reject, macro AUC delta -0.000716, adoption 80% CI [-0.001250, -0.000177], soft AUC delta -0.000195
- `rn18-late-anchor-5e-4`: reject, macro AUC delta -0.000069, adoption 80% CI [-0.000632, +0.000504], soft AUC delta +0.000254
- `rn18-report-attributes-005`: reject, macro AUC delta -0.000361, adoption 80% CI [-0.000774, +0.000060], soft AUC delta -0.000171
- `rn18-global-clip-2`: reject, macro AUC delta -0.000326, adoption 80% CI [-0.000988, +0.000371], soft AUC delta -0.000712
- `rn18-head-weight-decay-005`: reject, macro AUC delta -0.000449, adoption 80% CI [-0.000811, -0.000107], soft AUC delta -0.000115
- `rn18-backbone-lr-2p5e-5`: reject, macro AUC delta -0.001331, adoption 80% CI [-0.002156, -0.000522], soft AUC delta -0.001094
- `rn18-early-anchor-5e-3`: reject, macro AUC delta -0.000126, adoption 80% CI [-0.000522, +0.000272], soft AUC delta -0.000169
- `rn18-twelve-epochs`: reject, macro AUC delta +0.000232, adoption 80% CI [-0.000428, +0.000907], soft AUC delta +0.000271
- `rn18-gin-blend-max-025`: reject, macro AUC delta -0.000645, adoption 80% CI [-0.001275, -0.000006], soft AUC delta -0.000206
- `rn18-warmup-start-ratio-05`: reject, macro AUC delta -0.000837, adoption 80% CI [-0.001514, -0.000132], soft AUC delta -0.000369
- `rn18-contrast-mixing-target-min-075`: reject, macro AUC delta -0.000649, adoption 80% CI [-0.001128, -0.000172], soft AUC delta -0.000198
- `rn18-affine-rotation-5deg`: reject, macro AUC delta -0.000905, adoption 80% CI [-0.001533, -0.000285], soft AUC delta -0.000325
- `rn18-noise-std-max-001`: reject, macro AUC delta -0.000110, adoption 80% CI [-0.000555, +0.000341], soft AUC delta -0.000040
- `rn18-cosine-floor-005`: reject, macro AUC delta -0.000242, adoption 80% CI [-0.000655, +0.000149], soft AUC delta +0.000162
- `rn18-batch-12-scaled-rates`: reject, macro AUC delta -0.000850, adoption 80% CI [-0.001987, +0.000346], soft AUC delta -0.000984
- `rn10t-incumbent-recipe`: reject, macro AUC delta -0.005309, adoption 80% CI [-0.007578, -0.003138], soft AUC delta -0.004072
- `rn10t-scaled-backbone-lr`: reject, macro AUC delta -0.004567, adoption 80% CI [-0.006845, -0.002493], soft AUC delta -0.003643
- `refined-rn50-incumbent-recipe`: provisional_accept, macro AUC delta +0.004588, adoption 80% CI [+0.002726, +0.006388], soft AUC delta +0.003023
- `rn18-intensity-probability-075`: reject, macro AUC delta -0.000645, adoption 80% CI [-0.001109, -0.000188], soft AUC delta -0.000350
- `refined-rn50-scaled-backbone-lr`: provisional_accept, macro AUC delta +0.003481, adoption 80% CI [+0.001293, +0.005648], soft AUC delta +0.002570
- `rn18-gin-probability-05`: reject, macro AUC delta -0.000817, adoption 80% CI [-0.001548, -0.000058], soft AUC delta -0.000425
- `rn18-plane-backbones-joint`: reject, macro AUC delta +0.000045, adoption 80% CI [-0.001111, +0.001129], soft AUC delta +0.000307
- `rn18-plane-backbones-dense-fusion`: reject, macro AUC delta -0.005700, adoption 80% CI [-0.007522, -0.003977], soft AUC delta -0.002641
- `rn10t-plane-backbones-joint`: reject, macro AUC delta -0.002272, adoption 80% CI [-0.004418, -0.000306], soft AUC delta -0.001613
- `rn10t-plane-backbones-dense-fusion`: reject, macro AUC delta -0.009468, adoption 80% CI [-0.011851, -0.007191], soft AUC delta -0.006175
- `no-report-attribute-heads`: reject, macro AUC delta -0.000567, adoption 80% CI [-0.001071, -0.000061], soft AUC delta -0.000505
