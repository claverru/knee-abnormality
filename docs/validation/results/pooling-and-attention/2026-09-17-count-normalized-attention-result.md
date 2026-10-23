# 2026-09-17 count-normalised attention

**Rejected.** Training with acquisition-count-normalised attention lost 0.0035 [−0.0069, −0.0001]
against the same recipe without it, and the loss falls on the targets the normalisation was always
expected to hurt. Together with the inference measurements it closes the idea, and the option was
removed from the code.

## Runs

Fold 0 of `all_5fold_v3`, ten epochs, selected epoch 10 in both cases.

| Run | Change | Macro AUC |
|---|---|---:|
| `count-normalized-attention` | `model.count_normalized_attention: true` **and** `data.allocation_concentration: 10.0`, seed 2264567078 | 0.88707 |
| `allocation-concentration` | `data.allocation_concentration: 10.0` only, seed 2634103341 ([result](../sampling/2026-09-17-allocation-concentration-result.md)) | 0.89054 |
| `ranked-filtered-seed{42,43,44}` | the recipe, no options ([result](../sampling/2026-09-17-ranked-filtered-result.md)) | 0.889573 / 0.889650 / 0.892795 (mean 0.890673) |

The pair was trained together because random per-type allocation is what makes a series' token
count vary between epochs, which is the condition under which count normalisation was expected to
help. `allocation-concentration` therefore isolates the normalisation.

- **Comparison:** `scripts/tools/compare_fold_runs.py`, 2,000 paired report-group bootstrap draws,
  artifacts in `outputs/analysis/c/count-norm-vs-allocation/`. Predictions differ by a mean 0.045.
- **Macro Δ against `allocation-concentration`:** −0.00346 [−0.00693, −0.00011].
- **Families:** meniscus −0.0091 [−0.0178, −0.0012], OA −0.0020 [−0.0085, +0.0043], the other eight
  −0.0024 [−0.0067, +0.0017].
- **Per target:** lateral meniscus −0.0139, contusion −0.0084, ACL −0.0082, fracture −0.0059,
  lateral OA −0.0054, medial meniscus −0.0043, against synovitis +0.0075, MCL +0.0031 and medial
  OA +0.0014.
- **Training cost:** 1,624 s, the same as the controls.

## Why it fails

- **It flattens exactly the weighting the model had learned.** Attention already favours
  fluid-sensitive series relative to their token share (enrichment 1.17–1.46, against 0.63–0.86 for
  T1/PD;
  [type importance](2026-09-17-plane-contrast-importance.md#attention-already-over-weights-fluid-types)),
  and the menisci depend on coronal and sagittal fluid
  ([contribution](2026-09-17-type-contribution.md)). Equalising mass per acquisition removes that
  preference, and the menisci lose the most.
- **It is not a convergence problem.** The gap was −0.0096 at epoch 1, −0.0023 at epoch 3, and
  stopped closing after that, ending at −0.0035. The sibling run without normalisation closed its
  own early gap to −0.0001 by epoch 10.
- **Inference agreed already.** On frozen checkpoints the same normalisation is neutral at today's
  allocation (−0.0009 on the three ranked-filtered checkpoints, +0.0001 on
  `allocation-concentration`) and loses to plain attention temperature at every count
  ([temperature](2026-09-17-attention-temperature.md)). Its only benefit was removing the drop when
  every candidate triplet is pooled, where per-series counts are most uneven.

## Status

The option was removed from `src/knee_abnormality/` after this result, so
`configs/` no longer carries `model.count_normalized_attention` and the working tree returns to the
unnormalised pooling. The run directory keeps its own archived sources and config, which still
name the field.

## Caveats

- Single runs on both sides, at different seeds; run-to-run noise has a standard deviation of
  0.0015–0.0018, and the interval only just excludes zero.
- The candidate carries two changes; the isolation rests on `allocation-concentration` being the
  same recipe with the normalisation off, which it is.
- Only the trained, whole-model variant was tested. A per-type rather than per-acquisition
  normalisation, or one applied to a subset of series, was not.
