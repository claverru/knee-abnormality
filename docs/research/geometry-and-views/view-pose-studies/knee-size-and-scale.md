# Study 1: knee size and per-knee scale

Masks on 550 knees; model estimates on the 150 fold-1 validation knees with the incumbent
(`report-attribute-loss-off-2`). Scripts in the session scratchpad (`knee-size/`).

**Knee size varies and the scanner does not normalise it.**

| Measure (550 knees) | Mean | sd | 5th to 95th |
|---|---:|---:|---|
| Femur width | 73.7 mm | 6.3 | 65.0 to 83.9 |
| Tibia width | 67.5 mm | 6.1 | 57.7 to 77.5 |
| Femur depth, front to back | 56.3 mm | 5.0 | 49.6 to 65.0 |

- Width and depth correlate 0.76; femur and tibia width 0.91. Size itself varies by 8.0%.
- Acquired field against knee size: r 0.15, 2% of the size variance. Native pixels average 0.34
  to 0.37 mm (95th percentile 0.56 to 0.66).

**How much of today's crop the anatomy fills** (mean, 5th to 95th):

| View | Measure | Fill | Knees over the field |
|---|---|---|---:|
| Coronal 84 mm | bone width | 0.87 (0.76 to 0.99) | 3.8% |
| Coronal 93.3 mm | bone width | 0.78 (0.68 to 0.89) | 0% |
| Sagittal 84 mm | femur back to kneecap front | 0.93 (0.81 to 1.05) | 20.2% |
| Sagittal 93.3 mm | femur back to kneecap front | 0.84 (0.73 to 0.94) | 0.7% |
| Axial 120 mm | femur width | 0.62 | 0% |
| Axial 120 mm | femur back to kneecap front | 0.65 | 0% |

**The model can size the knee from its first look.**

- Estimator: root of (share-weighted spread of femur and tibia tokens across) × (spread of femur
  tokens front to back), in the fitted frame. Against the masks: r 0.939, residual 2.1 mm (2.8%).
- Using two planes per direction is better than one (width from axial + coronal r 0.910; coronal
  alone 0.848).
- One shared size number leaves the fill varying by 2.7% (coronal) and 2.6% (sagittal); each
  plane's own measure gives 1.6% and 0%. Separate per-plane scales from the model gave no gain
  over the shared one (3.7% / 3.7% against 3.6% / 3.8%).

**Scaling equalises the framing; it does not widen it** (150 knees, share of series with the
structure fully inside):

| View | Structure | Today | Per-knee scale | Wider for all |
|---|---|---:|---:|---:|
| Coronal fluid | collateral region (bone edge + 10 mm) | 34.8% | 17.4% | 93.5% |
| Coronal 84 mm | femoral condyles | 91.6% | 98.5% | 100% |
| Sagittal fluid | whole kneecap | 26.0% | 15.3% | 88.7% |
| Sagittal 84 mm | whole kneecap | 7.8% | 1.1% | 56.4% |

Small knees lose the spare margin they have today; large knees gain (femoral condyles inside the
84 mm coronal crop, largest third: 73% to 95%).

**Axial: the problem is position, not size** (198 series, 150 knees).

- The bones sit 14.8 mm forward of the crop centre (sd 5.9); 6.8 mm of margin in front of the
  kneecap (sd 7.2; its front is cut in 13% of series), 36.4 mm behind the condyles.
- Kneecap front plus 25 mm behind the condyles inside: today 84.3%, resize only 84.3%, centre
  only 91.9%, centre and resize 98.5%. The turn was not measured.
- Centring on the bones themselves pushes the back of the knee out (12.6% inside).
- Axial has no canvas margin today.

**Run on it** (anatomy-recipe run 7, `crop-size-per-knee-from-the-first-look`): one shared scale
for the four sagittal and coronal views, axial untouched. ×4 0.927608 against 0.927291
(+0.0003; 80% −0.0009 to +0.0015), ×1 0.917680 against 0.917342. Mean applied scale 1.000,
spread 0.067, 5.1% of knees at the clamp. Up: MCL +0.0085, Synovitis +0.0025, Medial OA +0.0018.
Down: Fracture −0.0041, Lateral OA −0.0032, Lateral Meniscus −0.0028. It used a measured
reference (17.087) and clamp (0.87 to 1.13), which the agreed direction replaces by boxes read
from the map.

**Limits.** One checkpoint; the reference came from validation knees; coverage was measured in
the plane of the middle slice; the collateral and popliteal regions are proxies (bone edge plus a
margin), since the masks have bones only.

Drawings: `crops-coronal-fluid.png`, `crops-sagittal-fluid.png`, `crops-axial.png`,
`size-summary.png`.
