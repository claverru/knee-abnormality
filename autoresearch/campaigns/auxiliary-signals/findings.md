# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/auxiliary-signals/findings.md` (not tracked).

Incumbent: `scanner-confusion-detached-head-20260926T165429Z-55c4db64`

- `attribute-loss-weight-0-2`: reject, macro AUC delta -0.000836, adoption 80% CI [-0.002466, +0.000800], soft AUC delta -0.000608
- `area-balanced-attribute-loss`: reject, macro AUC delta -0.000615, adoption 80% CI [-0.002162, +0.000940], soft AUC delta -0.001388
- `reliability-filtered-attribute-questions`: reject, macro AUC delta +0.001036, adoption 80% CI [-0.000423, +0.002430], soft AUC delta -0.000051
- `drop-report-completeness-options`: reject, macro AUC delta -0.000487, adoption 80% CI [-0.001976, +0.001095], soft AUC delta -0.002339
- `drop-hedged-options`: reject, macro AUC delta -0.001487, adoption 80% CI [-0.003071, +0.000056], soft AUC delta -0.002052
- `scanner-group-adversarial-head`: reject, macro AUC delta -0.006040, adoption 80% CI [-0.008598, -0.003494], soft AUC delta -0.004658
- `scanner-adversarial-decoupled-weight`: reject, macro AUC delta -0.004186, adoption 80% CI [-0.006477, -0.002028], soft AUC delta -0.003237
- `attribute-cell-reliability-filter`: reject, macro AUC delta -0.000623, adoption 80% CI [-0.001980, +0.000726], soft AUC delta -0.001141
- `attribute-threshold-grade-merges`: reject, macro AUC delta -0.001408, adoption 80% CI [-0.002748, -0.000077], soft AUC delta -0.001433
- `scanner-confusion-detached-head`: provisional_accept (original verdict: reject), macro AUC delta -0.000763, adoption 80% CI [-0.002737, +0.001097], soft AUC delta -0.001423
- `attribute-conditional-child-targets`: reject, macro AUC delta -0.001186, adoption 80% CI [-0.002656, +0.000252], soft AUC delta -0.000944
- `meniscus-status-location-joint-targets`: reject, macro AUC delta -0.000644, adoption 80% CI [-0.002150, +0.000873], soft AUC delta -0.001044
- `scanner-confusion-stronger-head`: reject, macro AUC delta -0.003297, adoption 80% CI [-0.005314, -0.001374], soft AUC delta -0.004172
- `scanner-confusion-backbone-features`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `site-bias-product-of-experts`: reject, macro AUC delta +0.000399, adoption 80% CI [-0.001065, +0.001902], soft AUC delta -0.000668
- `scanner-model-confusion-target`: reject, macro AUC delta -0.001343, adoption 80% CI [-0.003480, +0.000727], soft AUC delta -0.001190
- `attribute-consensus-target-hardening`: reject, macro AUC delta -0.000240, adoption 80% CI [-0.002099, +0.001646], soft AUC delta -0.000793
- `attribute-gate-family-shared-pooling`: reject, macro AUC delta -0.002356, adoption 80% CI [-0.004120, -0.000542], soft AUC delta -0.001999
- `predicted-attribute-residual-readout`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `model-bias-product-of-experts`: reject, macro AUC delta -0.002789, adoption 80% CI [-0.004578, -0.001036], soft AUC delta -0.002657
- `site-bias-with-model-confusion`: reject, macro AUC delta -0.000297, adoption 80% CI [-0.002372, +0.001690], soft AUC delta -0.000570
