# Series filtering census: flag precision and inference effect

## Flag precision and recall

Precision is non-target (non-knee or other knee) among reviewed series (`precision.json`).

| Flag | Flagged | Reviewed | Non-target precision |
|---|---:|---:|---:|
| Proposed far rule (below) | 11 | 11 | 1.00 |
| Other-knee side (x spread > 150 mm) | 35 | 35 | 1.00 |
| Robust 20 mm drop | 159 | 159 | 0.16 (91 patellar, 37 target, 6 other session) |
| Centre projects outside slab or FOV | 127 | 127 | 0.32 |
| Slab-centre offset > 80 mm | 43 | 43 | 0.84 |

Recall: all 47 robust-rejected studies were reviewed; 3 non-knee series in `…63928925` cannot be
seen by any centre-based rule (the study has no accepted centre). A non-knee series in an accepted
study that escapes every geometric flag would have to lie within 20 mm in-plane and inside the
slab of the centre. The random sample of 40 unflagged studies (217 series) found no non-knee,
other-knee or plane problem and one missed T1 FS series (Philips `WATS`, now in the pattern);
by the rule of three, the per-study miss rate is below about 7.5% at 95% (hypothesis-level
bound, small sample).

## Inference effect (fold 0)

Measured. Re-scoring of the three 2026-09-17 control checkpoints (`00-control`,
`01-control-seed-43`, `02-control-seed-44`, [overnight results](../../validation/results/campaigns-and-queues/2026-09-17-overnight-results.md))
with fold-0 series removed from, or moved within, the series index **before preparation**, so the
study reference is re-estimated and same-slot siblings take over the budget; nothing reallocates
budget across slots. Only affected studies are re-scored, at batch size 1 for both the intact and
the modified study (batch-1 intact differs from the saved batch-8 predictions by up to 0.006 in
probability through bf16 batch-composition numerics). Paired report-group bootstrap (852 groups,
2,000 draws), delta of macro AUC averaged over the checkpoints (intact 0.8930 / 0.8898 / 0.8916).

| Variant (fold 0) | Series | Studies | Δ macro ×1000 | 95% CI ×1000 | Per checkpoint ×1000 |
|---|---:|---:|---:|---|---|
| Drop far + stray other-knee (proposed drops) | 2 | 2 | −0.004 | [−0.040, +0.028] | +0.004, −0.012, −0.005 |
| Drop other-knee minor side (incl. bilateral) | 11 | 3 | −0.016 | [−0.096, +0.052] | −0.025, +0.014, −0.038 |
| Drop pixel > 1 mm | 2 | 2 | +0.015 | [−0.001, +0.055] | −0.003, +0.031, +0.017 |
| Drop T1 FS in T1/PD slots | 26 | 26 | +0.088 | [−0.120, +0.369] | +0.009, +0.100, +0.155 |
| Drop zero-TE | 40 | 29 | −0.092 | [−0.207, +0.030] | −0.095, −0.134, −0.047 |
| Drop small-FOV patellar axials | 14 | 14 | −0.061 | [−0.235, +0.103] | −0.058, +0.046, −0.171 |
| Drop oblique > 30° | 113 | 101 | −0.264 | [−1.064, +0.562] | −0.028, −0.287, −0.477 |
| Relabel contrast (moved) | 4 | 4 | +0.009 | [−0.023, +0.042] | +0.000, +0.018, +0.007 |
| Relabel swapped planes (moved) | 7 | 1 | −0.207 | [−0.660, +0.000] | −0.110, −0.243, −0.267 |
| All drops and relabels | 91 | 64 | −0.195 | [−0.689, +0.276] | −0.194, −0.215, −0.176 |

No variant moves fold-0 macro AUC by more than 0.0003, and every interval includes zero (the plane
swap touches zero at its upper bound). The only consistent negative is the plane-swap relabel of
one study: these checkpoints were trained with that study's series in their geometric slots, so a
relabel is off-distribution for them; whether a model trained on relabelled slots gains is
untested. The proposed drops free 0 to 1 triplets on fold 0; the two contrast relabels that fill a
missing fluid slot add 20 triplets. Caveat: checkpoints were trained with every flagged series
present, so removal measures reliance, not the value of training without them.
