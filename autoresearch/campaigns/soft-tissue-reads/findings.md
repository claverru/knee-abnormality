# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/soft-tissue-reads/findings.md` (not tracked).

Incumbent: `sagittal-fluid-detail-view-on-the-same-slices-20261011T114820Z-1bb81815`

- `residual-cell-read-on-a-detached-fourth-stage`: reject, macro AUC delta x4 -0.000918, x1 +0.000041, 80% CI of x4 [-0.001565, -0.000302], soft AUC delta -0.000440
- `oa-cartilage-and-bone-zone-pooled-apart`: reject, macro AUC delta x4 -0.001075, x1 -0.001020, 80% CI of x4 [-0.001894, -0.000245], soft AUC delta -0.000597
- `cell-position-inside-the-structure`: reject, macro AUC delta x4 -0.000799, x1 +0.000426, 80% CI of x4 [-0.001569, -0.000009], soft AUC delta -0.000298
- `pf-oa-own-cell-reader`: reject, macro AUC delta x4 -0.000665, x1 -0.000144, 80% CI of x4 [-0.001460, +0.000118], soft AUC delta +0.000043
- `soft-tissue-map-on-the-28-by-28-grid`: reject, macro AUC delta x4 -0.000478, x1 -0.000374, 80% CI of x4 [-0.001312, +0.000369], soft AUC delta -0.000267
- `oa-cartilage-read-from-second-stage-cells`: reject, macro AUC delta x4 -0.000315, x1 -0.000921, 80% CI of x4 [-0.000962, +0.000291], soft AUC delta -0.000127
- `bone-zone-taught-on-the-cell-grid`: reject, macro AUC delta x4 -0.000538, x1 -0.000577, 80% CI of x4 [-0.001113, +0.000047], soft AUC delta -0.000341
- `cell-read-relative-to-the-average-open-cell`: reject, macro AUC delta x4 -0.001373, x1 -0.000473, 80% CI of x4 [-0.001965, -0.000817], soft AUC delta -0.000413
- `cell-read-own-third-stage-projection`: reject, macro AUC delta x4 -0.000242, x1 +0.000252, 80% CI of x4 [-0.000863, +0.000398], soft AUC delta -0.000159
- `map-overlap-term-weighted-per-structure`: reject, macro AUC delta x4 -0.001028, x1 -0.000759, 80% CI of x4 [-0.001601, -0.000479], soft AUC delta -0.000402
- `cell-position-for-the-meniscus-reader-only`: reject, macro AUC delta x4 -0.000486, x1 -0.000380, 80% CI of x4 [-0.001079, +0.000079], soft AUC delta -0.000078
- `cell-projection-with-a-coarse-block-detail-adapter`: reject, macro AUC delta x4 -0.000061, x1 -0.000374, 80% CI of x4 [-0.000639, +0.000497], soft AUC delta +0.000026
- `oa-pools-with-one-total-attention-mass`: reject, macro AUC delta x4 -0.000808, x1 -0.000038, 80% CI of x4 [-0.001390, -0.000208], soft AUC delta -0.000269
- `cell-correction-weighted-by-map-certainty`: reject, macro AUC delta x4 -0.001004, x1 -0.001021, 80% CI of x4 [-0.001713, -0.000263], soft AUC delta -0.000146
- `fine-cartilage-read-with-mask-guided-subcell-weights`: reject, macro AUC delta x4 -0.000576, x1 -0.001034, 80% CI of x4 [-0.001277, +0.000192], soft AUC delta -0.000529
- `cell-attention-normalized-by-routed-area-per-view`: reject, macro AUC delta x4 -0.000019, x1 +0.000704, 80% CI of x4 [-0.000911, +0.000881], soft AUC delta -0.000070
- `cell-attention-excludes-exact-routing-zeros`: reject, macro AUC delta x4 -0.000291, x1 -0.000055, 80% CI of x4 [-0.001237, +0.000689], soft AUC delta -0.000383
- `soft-tissue-map-learns-adjacent-share-transitions`: reject, macro AUC delta x4 -0.000084, x1 -0.000244, 80% CI of x4 [-0.000742, +0.000495], soft AUC delta +0.000033
- `bundle-area-normalised-attention-with-the-detail-adapter`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `bundle-attention-adapter-cell-zone-own-pf-reader-and-position`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `bundle-the-three-runs-ahead-at-one-image-multiplier`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `map-taught-on-every-study-from-the-reference-pseudo-volumes`: reject, macro AUC delta x4 -0.001265, x1 -0.001663, 80% CI of x4 [-0.002137, -0.000425], soft AUC delta -0.000351
- `dense-context-slices-in-the-training-bag`: reject, macro AUC delta x4 -0.002286, x1 -0.002880, 80% CI of x4 [-0.003851, -0.000776], soft AUC delta -0.001396
- `map-agreement-across-planes-in-patient-space`: reject, macro AUC delta x4 -0.000454, x1 -0.000511, 80% CI of x4 [-0.001014, +0.000068], soft AUC delta -0.000227
- `structure-tokens-in-the-fourth-stage-bag`: reject, macro AUC delta x4 -0.000367, x1 -0.000743, 80% CI of x4 [-0.001075, +0.000405], soft AUC delta -0.000271
- `incumbent-recipe-retrained-unchanged`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `incumbent-recipe-retrained-unchanged`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `incumbent-recipe-retrained-unchanged`: reject, macro AUC delta x4 -0.000324, x1 +0.001507, 80% CI of x4 [-0.001019, +0.000340], soft AUC delta +0.000200
- `incumbent-recipe-with-twelve-epochs`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `incumbent-recipe-with-twelve-epochs`: reject (acceptance withdrawn), macro AUC delta x4 +0.000094, x1 +0.000496, 80% CI of x4 [-0.000835, +0.000957], soft AUC delta +0.000256
- `incumbent-with-a-small-label-loss-on-each-plane-read`: reject, macro AUC delta x4 -0.001550, x1 -0.001277, 80% CI of x4 [-0.002496, -0.000617], soft AUC delta -0.000772
- `sagittal-fluid-detail-view-on-the-same-slices`: provisional_accept, macro AUC delta x4 +0.000558, x1 +0.000462, 80% CI of x4 [-0.000316, +0.001510], soft AUC delta +0.000385
- `pseudo-volumes-on-the-untaught-views-with-the-masks-at-half`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `soft-tissue-map-as-a-side-task-with-the-transition-loss`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `pseudo-volumes-on-the-untaught-views-with-the-masks-at-half`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `detail-view-open-to-all-twelve-labels`: reject, macro AUC delta x4 -0.000198, x1 -0.000828, 80% CI of x4 [-0.001187, +0.000725], soft AUC delta +0.000051
- `detail-view-on-three-of-the-five-slices`: reject, macro AUC delta x4 -0.001134, x1 -0.002052, 80% CI of x4 [-0.001985, -0.000392], soft AUC delta -0.000927
- `detail-view-shares-the-feature-statistics-draw`: reject, macro AUC delta x4 -0.001039, x1 -0.000800, 80% CI of x4 [-0.001924, -0.000222], soft AUC delta -0.000275
- `soft-tissue-map-as-a-side-task-with-the-transition-loss`: reject, macro AUC delta x4 -0.000957, x1 -0.001255, 80% CI of x4 [-0.001903, -0.000061], soft AUC delta -0.000133
