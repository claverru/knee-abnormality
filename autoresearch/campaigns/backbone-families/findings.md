# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/backbone-families/findings.md` (not tracked).

Incumbent: `focalnet-tiny-srf-20261015T172800Z-0542173d`

- `sam2-hiera-tiny`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `hrnet-w18`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `focalnet-tiny-srf`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `focalnet-tiny-srf`: provisional_accept, macro AUC delta x4 +0.003656, x1 +0.003160, 80% CI of x4 [+0.001673, +0.005663], soft AUC delta +0.001108
- `dinov3-convnext-tiny`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `dinov3-convnext-tiny-10ep`: reject, macro AUC delta x4 -0.004617, x1 -0.005448, 80% CI of x4 [-0.006512, -0.002901], soft AUC delta -0.002657
- `pvt-v2-b2-checkpointed`: reject, macro AUC delta x4 -0.005717, x1 -0.006290, 80% CI of x4 [-0.007636, -0.003859], soft AUC delta -0.002566
- `rdnet-tiny-checkpointed`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `focalnet-tiny-srf-10ep`: reject, macro AUC delta x4 -0.000662, x1 -0.000861, 80% CI of x4 [-0.002173, +0.000911], soft AUC delta -0.000228
- `davit-tiny`: reject, macro AUC delta x4 -0.002147, x1 -0.002884, 80% CI of x4 [-0.003773, -0.000519], soft AUC delta -0.001658
- `edgenext-base`: reject, macro AUC delta x4 -0.002783, x1 -0.003383, 80% CI of x4 [-0.004575, -0.001070], soft AUC delta -0.002298
- `efficientvit-b2-checkpointed`: reject, macro AUC delta x4 -0.001536, x1 -0.002149, 80% CI of x4 [-0.003179, +0.000179], soft AUC delta +0.000551
- `wtconvnext-tiny-9ep`: reject, macro AUC delta x4 -0.003169, x1 -0.004077, 80% CI of x4 [-0.004744, -0.001621], soft AUC delta -0.001845
- `efficientvit-real-row-batch-renorm`: reject, macro AUC delta x4 -0.002771, x1 -0.002546, 80% CI of x4 [-0.004576, -0.000786], soft AUC delta -0.000849
- `focalnet-early-anchor-rate-ten-epoch-combination`: reject, macro AUC delta x4 -0.000010, x1 -0.001708, 80% CI of x4 [-0.001263, +0.001249], soft AUC delta +0.000207
- `dinov3-convnext-twelve-epoch-fit`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `efficientvit-read-layernorm`: reject, macro AUC delta x4 -0.001953, x1 -0.004230, 80% CI of x4 [-0.003851, -0.000077], soft AUC delta -0.000333
- `efficientvit-bf16-attention`: reject, macro AUC delta x4 -0.001686, x1 -0.003090, 80% CI of x4 [-0.003468, +0.000171], soft AUC delta -0.000062
- `efficientvit-stochastic-depth`: reject, macro AUC delta x4 -0.000175, x1 -0.003256, 80% CI of x4 [-0.001896, +0.001568], soft AUC delta +0.000387
- `efficientvit-checkpoint-stage0`: invalid, macro AUC delta x4 unavailable, x1 unavailable, 80% CI of x4 unavailable
- `efficientvit-stochastic-depth-stage0`: reject, macro AUC delta x4 -0.001450, x1 -0.003417, 80% CI of x4 [-0.003248, +0.000417], soft AUC delta +0.000051
- `efficientvit-stochastic-depth-02`: reject, macro AUC delta x4 -0.000044, x1 -0.002353, 80% CI of x4 [-0.001777, +0.001756], soft AUC delta -0.000074
