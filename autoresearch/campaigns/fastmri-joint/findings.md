# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/fastmri-joint/findings.md` (not tracked).

Incumbent: `historical-control`

- `confident-pseudo-labels-weigh-more`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `kaggle-only-final-epochs`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `batch-norm-statistics-from-kaggle`: provisional_accept (original verdict: reject), macro AUC delta x4 +0.000626, x1 -0.000116, 80% CI of x4 [+0.000113, +0.001168], soft AUC delta +0.000334
- `fastmri-weight-follows-learning-rate`: provisional_accept (original verdict: reject), macro AUC delta x4 +0.001799, x1 -0.000063, 80% CI of x4 [+0.001132, +0.002497], soft AUC delta +0.001223
- `kaggle-only-final-epochs-rerun`: provisional_accept, macro AUC delta x4 +0.000316, x1 +0.000082, 80% CI of x4 [-0.000405, +0.001099], soft AUC delta -0.000338
- `fastmri-weight-follows-learning-rate-on-incumbent`: reject, macro AUC delta x4 -0.000231, x1 -0.000207, 80% CI of x4 [-0.001027, +0.000565], soft AUC delta +0.000343
- `confident-pseudo-labels-weigh-more-rerun`: reject, macro AUC delta x4 +0.000003, x1 -0.000724, 80% CI of x4 [-0.000838, +0.000812], soft AUC delta +0.000297
- `class-weights-over-both-datasets`: reject, macro AUC delta x4 -0.000202, x1 -0.000672, 80% CI of x4 [-0.001094, +0.000736], soft AUC delta +0.000675
- `no-loader-augmentation-on-fastmri`: reject, macro AUC delta x4 -0.000352, x1 +0.000334, 80% CI of x4 [-0.001363, +0.000663], soft AUC delta -0.000024
- `label-rate-matching-by-sampling`: reject, macro AUC delta x4 -0.000123, x1 +0.001508, 80% CI of x4 [-0.001062, +0.000812], soft AUC delta -0.000620
- `hard-pseudo-labels`: reject, macro AUC delta x4 -0.003609, x1 -0.002990, 80% CI of x4 [-0.004830, -0.002396], soft AUC delta -0.003575
- `batch-norm-statistics-from-kaggle-on-incumbent`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `l2-sp-off`: reject, macro AUC delta x4 +0.001806, x1 +0.001414, 80% CI of x4 [+0.001154, +0.002486], soft AUC delta +0.001079
- `fastmri-loss-weight-doubled`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
