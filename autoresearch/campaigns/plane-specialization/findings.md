# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/plane-specialization/findings.md` (not tracked).

Incumbent: `dense-readouts-learned-fusion-20260927T231847Z-33417f97`

- `random-label-plane-partition`: reject, macro AUC delta -0.019017, adoption 80% CI [-0.022800, -0.015361], soft AUC delta -0.017912
- `random-partition-learned-fusion`: reject, macro AUC delta -0.014949, adoption 80% CI [-0.018380, -0.011638], soft AUC delta -0.015452
- `dense-plane-readouts`: provisional_accept, macro AUC delta +0.004699, adoption 80% CI [+0.002409, +0.006904], soft AUC delta +0.002388
- `dense-readouts-learned-fusion`: provisional_accept (original verdict: reject), macro AUC delta +0.000902, adoption 80% CI [-0.000552, +0.002376], soft AUC delta +0.000887
- `dense-plane-heads`: reject, macro AUC delta +0.000481, adoption 80% CI [-0.000860, +0.001857], soft AUC delta +0.000312
- `radiology-plane-mapping`: reject, macro AUC delta -0.001075, adoption 80% CI [-0.003605, +0.001452], soft AUC delta +0.000338
- `plane-auxiliary-readouts`: reject, macro AUC delta +0.000161, adoption 80% CI [-0.002080, +0.002339], soft AUC delta +0.001070
- `winner-take-all-plane-routing`: reject, macro AUC delta -0.137067, adoption 80% CI [-0.145738, -0.128734], soft AUC delta -0.104539
- `dense-plane-full-heads`: reject, macro AUC delta -0.000191, adoption 80% CI [-0.001779, +0.001370], soft AUC delta +0.000094
- `fusion-guided-dense-plane-routing`: reject, macro AUC delta -0.002065, adoption 80% CI [-0.003696, -0.000605], soft AUC delta -0.001129
- `dense-plane-detached-joint-fusion`: reject, macro AUC delta -0.000911, adoption 80% CI [-0.002940, +0.001080], soft AUC delta -0.000405
- `nonlinear-plane-logit-fusion`: reject, macro AUC delta -0.006603, adoption 80% CI [-0.008450, -0.004771], soft AUC delta -0.005000
- `nonlinear-plane-logit-fusion-no-dropout`: reject, macro AUC delta -0.000777, adoption 80% CI [-0.002567, +0.000907], soft AUC delta -0.000220
- `cross-label-plane-logit-fusion`: reject, macro AUC delta -0.000728, adoption 80% CI [-0.002240, +0.000842], soft AUC delta -0.000451
- `radiology-auxiliary-dense-readouts`: reject, macro AUC delta -0.000307, adoption 80% CI [-0.001788, +0.001112], soft AUC delta +0.000847
- `leave-one-plane-out-distillation`: reject, macro AUC delta -0.004268, adoption 80% CI [-0.006010, -0.002650], soft AUC delta -0.003568
- `plane-attention-entropy-auxiliary`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `plane-backbone-dsu`: reject, macro AUC delta -0.004163, adoption 80% CI [-0.005866, -0.002438], soft AUC delta -0.004277
- `plane-oa-detector`: reject, macro AUC delta -0.001586, adoption 80% CI [-0.002887, -0.000335], soft AUC delta -0.000232
- `plane-report-attribute-heads`: reject, macro AUC delta -0.002610, adoption 80% CI [-0.004096, -0.001168], soft AUC delta -0.001177
- `plane-half-knee-mil`: reject, macro AUC delta -0.000608, adoption 80% CI [-0.001854, +0.000579], soft AUC delta -0.000321
- `plane-scanner-confusion`: reject, macro AUC delta -0.000156, adoption 80% CI [-0.001510, +0.001233], soft AUC delta -0.000833
