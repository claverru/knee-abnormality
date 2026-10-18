# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/backbone-families-2/findings.md` (not tracked).

Incumbent: `nextvit-small-ssld-20261018T055544Z-d7530959`

- `medsam2-hiera-tiny`: reject, macro AUC delta x4 -0.021043, x1 -0.021869, 80% CI of x4 [-0.023745, -0.018365], soft AUC delta -0.016255
- `lingbot-vit-small-stride16`: reject, macro AUC delta x4 -0.010421, x1 -0.010156, 80% CI of x4 [-0.012429, -0.008433], soft AUC delta -0.007621
- `medsam2-stem-separated-clipping`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `lingbot-stage-decay-08`: reject, macro AUC delta x4 -0.006109, x1 -0.006119, 80% CI of x4 [-0.007981, -0.004254], soft AUC delta -0.004230
- `flash-internimage-seven-epochs`: reject, macro AUC delta x4 -0.002862, x1 -0.005331, 80% CI of x4 [-0.004930, -0.000868], soft AUC delta -0.002278
- `medsam2-thirteen-epochs`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `radimagenet-densenet121`: reject, macro AUC delta x4 -0.009849, x1 -0.010287, 80% CI of x4 [-0.011980, -0.007779], soft AUC delta -0.007025
- `radimagenet-inception-v3`: reject, macro AUC delta x4 -0.011513, x1 -0.015051, 80% CI of x4 [-0.014088, -0.009202], soft AUC delta -0.008310
- `radimagenet-densenet121-normalized-middle`: reject, macro AUC delta x4 -0.010275, x1 -0.010328, 80% CI of x4 [-0.012326, -0.008165], soft AUC delta -0.007316
- `radimagenet-inception-resnet-v2-grids`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `radimagenet-densenet121-stage-decay-06`: reject, macro AUC delta x4 -0.012781, x1 -0.013524, 80% CI of x4 [-0.015070, -0.010518], soft AUC delta -0.008858
- `radimagenet-inception-resnet-v2-rerun`: reject, macro AUC delta x4 -0.007405, x1 -0.010313, 80% CI of x4 [-0.009628, -0.005369], soft AUC delta -0.005143
- `lingbot-native-normalized-middle-read`: reject, macro AUC delta x4 -0.005366, x1 -0.006760, 80% CI of x4 [-0.007338, -0.003231], soft AUC delta -0.003379
- `vssd-mesa-five-epochs`: reject, macro AUC delta x4 -0.005736, x1 -0.006123, 80% CI of x4 [-0.007712, -0.003849], soft AUC delta -0.003574
- `flash-internimage-eight-epochs`: reject, macro AUC delta x4 -0.003099, x1 -0.005862, 80% CI of x4 [-0.004991, -0.001289], soft AUC delta -0.001554
- `internimage-half-stage-middle-read`: reject, macro AUC delta x4 -0.002965, x1 -0.005734, 80% CI of x4 [-0.004924, -0.001089], soft AUC delta -0.002410
- `lingbot-midpoint-normalized-middle-read`: reject, macro AUC delta x4 -0.006997, x1 -0.006472, 80% CI of x4 [-0.008907, -0.005040], soft AUC delta -0.004603
- `transnext-low-resolution-direct-four-epochs`: reject, macro AUC delta x4 -0.010809, x1 -0.010475, 80% CI of x4 [-0.013002, -0.008715], soft AUC delta -0.008549
- `focalnet-tiny-lrf`: reject, macro AUC delta x4 -0.001463, x1 -0.001082, 80% CI of x4 [-0.002868, -0.000077], soft AUC delta -0.000684
- `nextvit-small-ssld`: provisional_accept, macro AUC delta x4 +0.000034, x1 +0.001337, 80% CI of x4 [-0.001662, +0.001771], soft AUC delta +0.001165
- `focalnet-tiny-lrf-twelve-epochs`: reject, macro AUC delta x4 -0.000948, x1 -0.002588, 80% CI of x4 [-0.002646, +0.000789], soft AUC delta -0.000552
- `efficientnet-b3-noisy-student`: reject, macro AUC delta x4 -0.003113, x1 -0.005371, 80% CI of x4 [-0.005094, -0.001216], soft AUC delta -0.001672
- `nextvit-small-twelve-epochs`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `tinyvit-21m-in22k`: reject, macro AUC delta x4 -0.000043, x1 -0.002656, 80% CI of x4 [-0.001468, +0.001363], soft AUC delta -0.000215
