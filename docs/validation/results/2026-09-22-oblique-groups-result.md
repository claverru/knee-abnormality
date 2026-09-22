# 2026-09-22 oblique groups

**Giving oblique series their own slot budget did not measurably help.** Fold 0 at ×1 and τ 1
scored 0.897527 against 0.899042 for `k_05` (−0.0015, 95% interval [−0.0055, +0.0026]). At the
submission setting, ×4 and τ 0.75, it scored 0.90467 against 0.904375 (+0.0003). Publicly S46
scored 0.944 against 0.945 for `k_05` (S41), and S47 at ×5 and τ 0.70 scored 0.945, all inside the
[0.006 noise between separately trained runs](../../inference/submissions/2026-09-17-seed-control-submissions.md).
Under the shared rule it is not adopted. On 2026-09-22 the user kept the option under test as
the reference of an autoresearch campaign that looks for what the model needs to use these views;
the committed recipe does not include it.

## Run

`outputs/training/oblique`: `k_05`'s recipe plus
`oblique_tilt_deg`
30 on axial fluid and 20 on both coronal types, fold 0 of `all_5fold_scanner_v1`, combined labels,
seed 2263738815, ten epochs. Selected epoch 9. Training took 1,601 s (`k_05` 1,584 s) and peaked at
22.3 GB allocated (21.0 GB). On fold 0, 103 of 905 studies change: the mean study rises from 26.80
to 27.37 training triplets at ×1 and from 104.0 to 106.0 at ×4.

## Per label at ×1

Paired comparison: `outputs/analysis/oblique-vs-k05/comparison.json`
([fold-run comparison](../fold-run-comparison.md)). The inputs of the 802 unchanged studies are
identical in both runs, so their deltas measure training divergence alone.

| Label | All | 103 changed studies (positives) | 802 unchanged studies (positives) |
|---|---:|---:|---:|
| ACL | +0.0147 | +0.0062 (32) | +0.0166 (101) |
| MCL | −0.0116 | −0.0583 (11) | −0.0019 (35) |
| Lateral meniscus | +0.0068 | −0.0281 (22) | +0.0109 (116) |
| PF OA | −0.0170 | −0.0260 (32) | −0.0170 (247) |
| Baker's cyst | −0.0102 | −0.0017 (13) | −0.0113 (203) |
| Fracture | +0.0104 | +0.0126 (4) | +0.0111 (52) |

The ACL gain sits in the unchanged studies, so it is divergence, not the mechanism. Inside the
changed studies the deltas are small-sample noise leaning negative.

## Temperature and count

`outputs/analysis/tau-sweep-oblique/` (`run_sweep.sh`, one full fold-0 inference per setting).

| τ | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.90 | 1.00 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ×4 | 0.90402 | 0.90447 | 0.90464 | **0.90467** | 0.90448 | 0.90371 | 0.90237 |
| ×5 | 0.90446 | 0.90490 | **0.90495** | 0.90486 | 0.90456 | 0.90371 | 0.90226 |

`k_05` has only ×4 τ 0.75 on record (0.904375), so the ×4 comparison is one point. On the same
checkpoint, S47 at ×5 and τ 0.70 scored **0.945** publicly against 0.944 for S46 at ×4 and τ 0.75,
one rounding step, matching `k_05`'s best (S41, ×4 τ 0.75). Fold 0 put the same pair +0.0003 apart.

## Why the added coverage may not show

- The oblique groups keep their slot's plane and contrast embeddings, so the model can separate an
  ACL-oblique view from the standard series of the same slot only by appearance.
- Their training coverage is already high: over ten epochs the coronal fluid oblique groups see
  89% of their triplet windows and the coronal T1/PD ones 85%
  (`outputs/analysis/acquisition-geometry-20260922/oblique_reach.txt`). Denser sampling of the
  same series has little room; 13 of 54 coronal fluid oblique slabs sit off the study centre,
  reach under 10 mm and a narrow Gaussian width, without losing run coverage.
- The studies they serve are already ranked well: `k_05`'s ACL AUC is 0.930 inside studies with a
  coronal-family fluid oblique against 0.939 elsewhere.
