# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/near-misses/findings.md` (not tracked).

Incumbent: `attention-token-dropout-random-20260925T003602Z-b69e50d5`

- `meniscus-shared-residual-head`: reject, macro AUC delta -0.000983, adoption 80% CI [-0.002508, +0.000557], soft AUC delta -0.000478
- `meniscus-folded-detector-isolated`: reject, macro AUC delta -0.000170, adoption 80% CI [-0.001881, +0.001517], soft AUC delta +0.000313
- `meniscus-shared-head-folded-detector`: reject, macro AUC delta +0.000498, adoption 80% CI [-0.001279, +0.002266], soft AUC delta +0.000834
- `meniscus-side-split-readout`: reject, macro AUC delta +0.000935, adoption 80% CI [-0.000615, +0.002453], soft AUC delta +0.000343
- `early-learning-regularization-current`: reject, macro AUC delta +0.000173, adoption 80% CI [-0.001457, +0.001775], soft AUC delta +0.000420
- `side-normalized-3d-position`: reject, macro AUC delta +0.000791, adoption 80% CI [-0.001564, +0.003217], soft AUC delta -0.000600
- `pf-oa-front-back-bias`: reject, macro AUC delta -0.000659, adoption 80% CI [-0.002748, +0.001368], soft AUC delta -0.002071
- `three-compartment-oa-detector`: reject, macro AUC delta +0.001346, adoption 80% CI [-0.000207, +0.002940], soft AUC delta +0.001414
- `attention-token-dropout-random`: provisional_accept, macro AUC delta +0.002477, adoption 80% CI [+0.000476, +0.004378], soft AUC delta +0.001866
- `attention-token-dropout-top`: reject, macro AUC delta -0.000709, adoption 80% CI [-0.002901, +0.001600], soft AUC delta +0.000263
- `plane-contrast-readouts`: reject, macro AUC delta -0.000634, adoption 80% CI [-0.002304, +0.001069], soft AUC delta -0.000381
- `any-oa-auxiliary`: reject, macro AUC delta -0.002104, adoption 80% CI [-0.004511, +0.000467], soft AUC delta -0.001133
- `slice-orientation-3d-position`: reject, macro AUC delta -0.001681, adoption 80% CI [-0.004379, +0.001251], soft AUC delta +0.001099
- `plane-negative-half-knee-mil`: reject, macro AUC delta -0.001674, adoption 80% CI [-0.003260, +0.000043], soft AUC delta -0.001477
- `attention-coupled-meniscus-mil`: reject, macro AUC delta -0.001563, adoption 80% CI [-0.003282, +0.000226], soft AUC delta -0.000015
- `oa-geometry-attention-residual`: reject, macro AUC delta -0.000317, adoption 80% CI [-0.001895, +0.001305], soft AUC delta +0.000446
- `oa-detector-isolated-union-supervision`: reject, macro AUC delta -0.002923, adoption 80% CI [-0.004790, -0.001005], soft AUC delta +0.000254
- `spatial-cluster-token-dropout`: reject, macro AUC delta -0.000172, adoption 80% CI [-0.001761, +0.001437], soft AUC delta -0.000669
- `oa-acquisition-mass-conserving-routing`: reject, macro AUC delta -0.001382, adoption 80% CI [-0.003231, +0.000431], soft AUC delta -0.000391
