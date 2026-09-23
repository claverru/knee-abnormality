# 2026-09-22 stride-16 truncation: lower validation, faster and smaller

**Truncating ReXNet-200 at stride 16 loses 0.0042 fold-0 macro AUC against `k_05`** (0.898229
against 0.902469 at ×4, paired 95% interval −0.0083 to −0.0002). Meniscus labels gain and the
other eight labels lose, PF OA most of all. Training is 13% faster, peak memory 2.5 GB lower and
the model has 70% fewer parameters. Truncation does not replace the current model. No
submission was made. The [stride-16 investigation](../../history/stride16-feature-fusion.md) closed
without adoption.

## Run

`stride16` is the [truncation arm](../../history/stride16-feature-fusion.md):
`model.feature_stride: 16` keeps `backbone.features[0:11]` (the last stride-16 block outputs
14×14×257) and removes the stride-32 blocks and the terminal 2560-channel expansion. Each triplet
gives 196 tokens instead of 49; the 2D position encoding is computed from stride-16 anchors.

Its effective configuration equals `k_05`'s except `model.feature_stride` and the label-manifest
path, whose file is byte-identical. The `feature_stride` option was removed from the code after
this result, per [project guidance](../../../AGENTS.md) against keeping rejected alternatives;
its code remains in this run's `source_snapshot.zip`. The shared settings are
`all_5fold_scanner_v1` fold 0,
[combined labels](../../public-labels/combined-labels.md), class-balanced BCE with k=0.5, seed
2263738815 (training is not deterministic) and ten epochs. The input identities differ only in the
preprocessing block (feature stride, anchor stride, source hash). `stride16` was trained from the
uncommitted working tree on `ab49830`, captured in its `source_snapshot.zip`; `k_05` was trained on
`79d5ba8` and re-predicted at ×4 ([class-balance result](2026-09-21-class-balance-exponent.md)).
Both selected epoch 8 at ×1 and were scored at ×4, τ 1.

| Run | Selection score (×1) | Macro AUC (×4) | Training | Peak allocated | Parameters |
|---|---:|---:|---:|---:|---:|
| `k_05` | 0.899042 | 0.902469 | 1,584 s | 21.0 GB | 14.60M |
| `stride16` | 0.894765 | 0.898229 | 1,379 s | 18.5 GB | 4.38M |

Epoch macro AUC at ×1 trails from the start and the gap narrows: −0.042 at epoch 1, −0.007 at
epoch 4, −0.004 at epochs 8–10.

## Validation against `k_05`

Paired report-group bootstrap, 2,000 draws, from [`compare_fold_runs`](../fold-run-comparison.md);
`outputs/analysis/stride16-vs-k05/comparison.json`.

| Group | Δ | 95% interval |
|---|---:|---|
| Macro | −0.0042 | −0.0083 to −0.0002 |
| Meniscus | +0.0063 | −0.0009 to +0.0143 |
| OA | −0.0026 | −0.0096 to +0.0042 |
| Other eight | −0.0073 | −0.0125 to −0.0022 |

| Label | Positives | `k_05` | Δ |
|---|---:|---:|---:|
| ACL | 133 | 0.9463 | −0.0096 |
| MCL | 46 | 0.9249 | −0.0052 |
| Medial Meniscus | 354 | 0.9295 | +0.0026 |
| Lateral Meniscus | 138 | 0.8972 | +0.0101 |
| Medial OA | 328 | 0.9008 | +0.0002 |
| Lateral OA | 241 | 0.8756 | −0.0054 |
| PF OA | 279 | 0.8647 | −0.0319 |
| Effusion | 523 | 0.8660 | −0.0035 |
| Synovitis | 113 | 0.8877 | −0.0063 |
| Baker's | 216 | 0.9147 | +0.0085 |
| Contusion | 153 | 0.9113 | −0.0077 |
| Fracture | 56 | 0.9109 | −0.0027 |

Gold macro AUC is 0.9110 against 0.9075 on 13 studies, too few to weigh. The compartment margin
AUCs rise (meniscus +0.0026, OA +0.0123) with intervals spanning zero.

## Interpretation

**Hypothesis, moderate confidence.** The finer grid helps small localized findings (lateral
meniscus, Baker's cyst), while the removed stride-32 layers carry what ACL and PF OA need. The
fusion arms kept both and tied `k_05` ([fusion result](2026-09-22-stride16-fusion-result.md)). Fold-0 validation has misjudged loss weighting
against the public leaderboard ([class-balance result](2026-09-21-class-balance-exponent.md)); no
such disagreement is established for architecture changes.
