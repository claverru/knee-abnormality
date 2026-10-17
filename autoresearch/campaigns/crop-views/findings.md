# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/crop-views/findings.md` (not tracked).

Incumbent: `coronal-crops-a-ninth-wider-20261017T022911Z-633ab764`

- `two-views-wide-and-posed`: provisional_accept, macro AUC delta x4 -0.000559, x1 +0.001448, 80% CI of x4 [-0.001640, +0.000553], soft AUC delta -0.000220
- `unposed-120-mm-everywhere`: reject, macro AUC delta x4 -0.001471, x1 -0.002522, 80% CI of x4 [-0.002935, +0.000007], soft AUC delta -0.000052
- `per-knee-scale-off`: provisional_accept, macro AUC delta x4 +0.001333, x1 -0.000154, 80% CI of x4 [+0.000076, +0.002614], soft AUC delta +0.000652
- `wide-view-on-distinct-depths`: provisional_accept, macro AUC delta x4 -0.000968, x1 -0.000666, 80% CI of x4 [-0.002126, +0.000201], soft AUC delta -0.000483
- `imagenet-resnet50`: reject, macro AUC delta x4 -0.004823, x1 -0.003359, 80% CI of x4 [-0.006937, -0.002655], soft AUC delta -0.003663
- `imagenet-resnet50-unposed-120`: reject, macro AUC delta x4 -0.005760, x1 -0.004362, 80% CI of x4 [-0.007891, -0.003584], soft AUC delta -0.003783
- `axial-views-unposed`: reject, macro AUC delta x4 -0.001010, x1 -0.000022, 80% CI of x4 [-0.002187, +0.000138], soft AUC delta -0.000307
- `wide-views-unposed-axial-scale-off`: reject, macro AUC delta x4 -0.001168, x1 +0.001046, 80% CI of x4 [-0.002014, -0.000304], soft AUC delta -0.000295
- `coronal-crops-a-ninth-wider`: provisional_accept, macro AUC delta x4 +0.001841, x1 +0.002212, 80% CI of x4 [+0.000746, +0.002894], soft AUC delta +0.000912
- `soft-bone-shares-for-the-joint-plane`: reject, macro AUC delta x4 -0.001218, x1 -0.002741, 80% CI of x4 [-0.002064, -0.000412], soft AUC delta -0.000775
- `coronal-wider-scale-off`: reject, macro AUC delta x4 -0.000927, x1 -0.001311, 80% CI of x4 [-0.001814, -0.000043], soft AUC delta -0.000384
- `canvas-locator-resnet10t`: reject, macro AUC delta x4 -0.001237, x1 -0.002041, 80% CI of x4 [-0.002152, -0.000371], soft AUC delta -0.000606
