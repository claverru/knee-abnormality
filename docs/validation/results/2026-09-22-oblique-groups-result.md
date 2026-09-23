# 2026-09-22 oblique groups

**Closed 2026-09-23 without adoption.** Oblique groups and the group embedding were removed from
the recipe and the code after three paired submissions came out level or lower (see
[S52](#group-embedding-on-image-blend-labels-s52) and the [history](../../history/oblique-groups.md)).

**Giving oblique series their own slot budget did not measurably help.** Fold 0 at ×1 and τ 1
scored 0.897527 against 0.899042 for `k_05` (−0.0015, 95% interval [−0.0055, +0.0026]). At the
submission setting, ×4 and τ 0.75, it scored 0.90467 against 0.904375 (+0.0003). Publicly S46
scored 0.944 against 0.945 for `k_05` (S41), and S47 at ×5 and τ 0.70 scored 0.945, all inside the
[0.006 noise between separately trained runs](../../inference/submissions/2026-09-17-seed-control-submissions.md).
Under the shared rule it is not adopted; the user kept it in `configs/training.yaml` on 2026-09-22
as the reference for the [oblique campaign](../../../autoresearch/campaigns/oblique/INDEX.md),
which looks for what the model needs to use these views.

## Run

`outputs/training/oblique`: `k_05`'s recipe plus
[`oblique_tilt_deg`](../../history/oblique-groups.md#mechanism)
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

## Group embedding on the cleaned source (S51)

`oblique-group-embedding` from the [cleaned-source campaign](../../../autoresearch/campaigns/oblique-clean/INDEX.md)
adds one learned 256-dimensional vector to the tokens of oblique groups in split slots
(`model.oblique.enabled`). Its run (`outputs/autoresearch/oblique-clean/experiments/oblique-group-embedding-20260923T062345Z-3d06eddd`,
selected epoch 9) was compared manually on 2026-09-23 with the campaign's `compare_fold_runs`
settings (report-group bootstrap, 2,000 draws, seed 20260914). The campaign was stopped, so it has
not recorded its own comparison yet. All values are ×4, τ 1 predictions against `oblique-clean`.

| Metric | `oblique-clean` | Embedding | Δ | 80% interval | 95% interval |
|---|---:|---:|---:|---|---|
| Hard macro AUC | 0.90371 | 0.90514 | +0.0015 | [−0.0004, +0.0033] | [−0.0014, +0.0044] |
| Soft macro AUC | 0.79859 | 0.80051 | +0.0019 | [+0.0009, +0.0030] | [+0.0004, +0.0036] |

By label family (hard AUC): OA +0.0039, the other eight +0.0015, menisci −0.0011. The largest
per-label moves are MCL +0.013 and Fracture +0.014, against Contusion −0.007 and lateral meniscus
−0.005. The first oblique campaign had measured the same idea at +0.0018 (×1, older source,
80% [−0.0005, +0.0040]). The hard-AUC gate rejects it; soft AUC would accept it.

Submitted as **S51** (×4, τ 0.75, batch 4, bundle `339719b4`, exported from the experiment's own
source), it scored **0.942** publicly. That ties `k05-clean` (S48, 0.942), which both local metrics
put below it (×4 hard 0.90276, soft 0.79846). It is also below S46 (0.944, oblique groups on the
older source) and the best submissions (S41 and S50, 0.945). `oblique-clean` itself has not been
submitted, so the embedding's own public effect is unmeasured.

## Group embedding on image-blend labels (S52)

`image-blend-oblique` is the `image-blend` run (the S50 recipe with image-blend labels on the
cleaned source) plus oblique groups and the group embedding. Its code is byte-identical to the
embedding candidate. Selected epoch 9. At ×1 it led `image-blend`'s best epoch by +0.0012 hard
(0.90532 against 0.90413) and +0.0013 soft (0.80383 against 0.80249). At ×4 τ 1, with the
campaign's report-group bootstrap (2,000 draws, seed 20260914):

| Metric | `image-blend` | `image-blend-oblique` | Δ | 80% interval |
|---|---:|---:|---:|---|
| Hard macro AUC | 0.90629 | 0.90655 | +0.0003 | [−0.0013, +0.0018] |
| Soft macro AUC | 0.80499 | 0.80573 | +0.0008 | [−0.0002, +0.0017] |

The per-label hard deltas at ×4:
- gains: PF OA +0.005, medial meniscus +0.0025, ACL, MCL and medial OA about +0.0016;
- losses: Contusion −0.004, Synovitis −0.003, Baker's −0.002, lateral meniscus −0.0015.

Submitted as **S52** (×4, τ 0.75, batch 4, bundle `imageblend-oblique-20260923-x4-tau075`), it
scored **0.943**, against 0.945 for S50. That is the third pair with oblique groups at or below
its counterpart, after S46 against S41 (−0.001) and S51 against S48 (0.000). The user closed the
line on 2026-09-23.
