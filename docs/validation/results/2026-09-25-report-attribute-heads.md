# Report-attribute heads, v3 and v3.1 targets (2026-09-25)

**Result:** +0.00075 ×4 macro AUC against the current control, 95% paired interval
[−0.0015, +0.0031]. Positive but below the 0.001 adoption threshold; not adopted by the rule.
The attribute term trained throughout, and the gains sit in the labels whose attributes audited
best. **Public leaderboard: 0.938 (S61), a tie with the control's 0.938 (S60)**, both fold 1 at
×4, attention temperature 0.75, batch 4; bundle `attr-heads-f1-20260925-x4-tau075` (manifest
`f37d548b`).

## Setup

| | Control `oa-detector-control` | Candidate `report-attribute-heads` |
|---|---|---|
| Recipe | current ([current model](../../modeling/current-model.md)) | same plus [report-attribute heads](../../modeling/report-attribute-heads.md) |
| Attribute targets | none | v3 manifest `configs/labels/jevk5-report-attribute-targets-v3.json` (superseded; kept in the run's `source_snapshot.zip`), targets SHA-256 `dce1b2bd…` |
| `trainer.report_attribute_loss_weight` | – | 0.1 |
| Seed, fold | 1485198267, fold 1 of `all_5fold_scanner_v1` | same |
| Training and validation rows | SHA-256 `d7688619…`, `361d5d06…` | identical |

The control ran before commit 665af0e removed retired options (`gated_attention`,
`residual_readout`, `oa_readout.shared_detector`, `repeated_series_share`, `class_balanced_bce`).
Its settings used the code paths the current code runs unconditionally, so the source-hash
difference changes no computation. Comparison:
`outputs/analysis/report-attribute-heads-vs-oa-detector-control/comparison.json`
(`scripts.tools.compare_fold_runs`, 874 validation studies, 852 report groups, 2,000 draws).

## Disease labels (×4, best epoch 10 for both)

| Metric | Control | Candidate | Delta |
|---|---|---|---|
| Macro AUC | 0.900568 | 0.901319 | +0.00075 (95% [−0.0015, +0.0031]) |
| Soft macro AUC | 0.79805 | 0.79895 | +0.0009 |
| Soft-target BCE | 0.44682 | 0.44567 | −0.0012 |

| Label | Delta | | Label | Delta |
|---|---|---|---|---|
| MCL | +0.0137 | | Fracture | −0.0057 |
| Contusion | +0.0068 | | Synovitis | −0.0055 |
| ACL | +0.0034 | | PF OA | −0.0036 |
| Lateral Meniscus | +0.0033 | | Baker's | −0.0027 |
| Medial OA | +0.0021 | | Medial Meniscus | −0.0022 |
| Effusion | +0.0014 | | Lateral OA | −0.0021 |

Family means: meniscus +0.0005, OA +0.0000, other eight +0.0010. OA medial/lateral margin AUC
0.8593 → 0.8690. Predictions stayed close (per-label Spearman 0.96–0.99). Per-label deltas are one
seed on one fold; the ones listed are exploratory.

**Pattern, not proven:** the largest gains are MCL, Contusion and ACL, whose status questions
audited at 87–95% and were fixed further in v3. Baker's, whose v3 size answers invent cysts, lost.
Synovitis and Fracture also lost despite clean v2 audits.

## Attribute term

`report_attribute_loss` fell every epoch, 1.136 → 0.952 (start 1.60, target entropy about 0.63),
and the mean per-question KL from 0.49 to 0.31. The worst-fitted questions at the end were PF
cartilage depth (0.64), effusion size (0.57), medial cartilage depth (0.54) and the two medial
meniscus questions (0.49–0.53); the best were loose bodies, synovitis, medial extrusion,
lateral osteophytes and lateral ligaments (0.10–0.13). Hoffa barely moved in the first two epochs.
Disease training loss matched the control's once the weighted attribute term is subtracted.

## Second run: target set 3.1

`report-attribute-heads-v31`: same seed, fold, weight and extracted answers, with target set 3.1
(osteophytes and Hoffa dropped, same-meaning options merged, cartilage extent and PF surface
gated, reliability weights; [design](../../modeling/report-attribute-heads.md)). Best epoch 9.
Comparisons: `outputs/analysis/report-attribute-heads-v31-vs-oa-detector-control/` and
`outputs/analysis/report-attribute-heads-v31-vs-v3/`.

| ×4 | Macro AUC | Delta | 95% paired interval | Soft macro | Soft BCE |
|---|---|---|---|---|---|
| Control | 0.900568 | | | 0.7980 | 0.4468 |
| v3 heads | 0.901319 | +0.00075 | [−0.0015, +0.0031] | 0.7989 | 0.4457 |
| v3.1 heads | 0.901616 | +0.00105 | [−0.0012, +0.0035] | 0.7988 | 0.4452 |

v3.1 against v3 heads: +0.00030, 95% [−0.0020, +0.0027]. The point estimate reaches the 0.001
threshold, but the interval spans zero; a normal approximation of the 80% interval is about
[−0.0005, +0.0026].

| Label | v3 − control | v3.1 − control |
|---|---|---|
| ACL | +0.0034 | +0.0020 |
| MCL | +0.0137 | +0.0136 |
| Medial Meniscus | −0.0022 | +0.0047 |
| Lateral Meniscus | +0.0033 | −0.0014 |
| Medial OA | +0.0021 | +0.0015 |
| Lateral OA | −0.0021 | −0.0004 |
| PF OA | −0.0036 | +0.0008 |
| Effusion | +0.0014 | +0.0004 |
| Synovitis | −0.0055 | −0.0004 |
| Baker's | −0.0027 | −0.0064 |
| Contusion | +0.0068 | +0.0036 |
| Fracture | −0.0057 | −0.0054 |

**Public leaderboard: 0.940 (S62)** against 0.938 for both the control (S60) and the v3 heads
(S61), all fold 1 at ×4, attention temperature 0.75, batch 4; bundle
`attr-heads-v31-f1-20260925-x4-tau075` (manifest `a2291317`).

MCL and Contusion gain in both runs; Fracture and Baker's lose in both. Synovitis and PF OA no
longer lose. Attribute loss fell 1.006 → 0.806 (floor about 0.55), mean per-question KL 0.51 →
0.31.
