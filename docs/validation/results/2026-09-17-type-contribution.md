# 2026-09-17 per-type logit contribution

**Fluid series carry about three quarters of the positive–negative logit separation, and every
fluid type contributes more per token than any T1/PD type.** Sagittal fluid is the most efficient
type and axial T1/PD the least. The per-label pattern follows clinical practice. This is
evidence for the [triplet budget allocation investigation](../../history/triplet-budget-allocation.md).

## Method

The attention pool is linear in its tokens, so for label k the logit splits exactly by series type
t: `C[k, t] = Σ_{i ∈ t} a[k, i] · (x[i] · w[k])`, where `x` are the layer-normed tokens, `a` the
label's attention and `w`, `b` the head, and `Σ_t C[k, t] = logit[k] − b[k]`.

- **Checkpoints:** `ranked-filtered-seed{42,43,44}`
  ([result](2026-09-17-ranked-filtered-result.md)), run with their byte-identical archived
  sources (`source_snapshot.zip` on `PYTHONPATH`), fold 0 of `all_5fold_v3`, ×2 validation
  images, bf16.
- **Checks:** the decomposition matches the model's logits to 2.2e-4, 3.6e-4 and 2.1e-4 (bf16
  rounding). The reproduced probabilities match `validation_predictions.csv` to 3e-8, with the
  recorded macro AUCs 0.889573, 0.889650 and 0.892795.
- **Groups:** shares are additive only on a fixed study set, so they are computed on (b) the 517
  studies with exactly the five types other than axial T1/PD and (a) the 112 studies with all six.
  Positives are targets ≥ 0.5, as in the metric.
- **Measures, averaged over the three runs:**
  - *Separation share:* `D[k, t]` = mean `C[k, t]` over positives minus over negatives, as a
    share of `Σ_t D[k, t]`.
  - *Variance share:* `cov(C[k, t], logit[k]) / var(logit[k])`.
  - *Efficiency:* separation share ÷ token share.
- **Scripts:** [snapshots](../../history/series-and-sampling-sources/README.md#triplet-allocation)
  (`infer.py`, `summarize.py`). Outputs: `outputs/analysis/t/type-contribution-20260917/`
  (`type_label_summary.csv`, `type_label_summary_macro.csv`, `label_group_counts.csv`,
  `raw/*.npz`, `raw/inference.json`). The `sep_share_min_pct`/`max_pct` columns of the macro file
  span labels, not runs; the run spread below was computed from `raw/`.

## Macro shares

| Measure (%) | Group | Ax T1/PD | Ax fluid | Cor T1/PD | Cor fluid | Sag T1/PD | Sag fluid |
|---|---|---:|---:|---:|---:|---:|---:|
| Separation | (b) | — | 24.5 | 12.8 | 22.7 | 13.6 | 26.4 |
| Separation | (a) | 7.4 | 21.3 | 11.6 | 20.4 | 13.5 | 25.8 |
| Variance | (b) | — | 25.6 | 12.0 | 21.8 | 14.1 | 26.6 |
| Variance | (a) | 9.2 | 22.5 | 11.3 | 19.0 | 13.6 | 24.4 |
| Tokens | (b) | — | 22.9 | 19.3 | 19.1 | 19.4 | 19.2 |
| Tokens | (a) | 18.7 | 18.8 | 15.7 | 15.5 | 15.7 | 15.7 |
| Efficiency (×) | (b) | — | 1.07 | 0.66 | 1.19 | 0.70 | 1.37 |
| Efficiency (×) | (a) | 0.39 | 1.14 | 0.74 | 1.32 | 0.86 | 1.64 |

The macro separation share moves at most 1.8 points across the three runs.

## Separation share per label, group (b)

| Label | Ax fluid | Cor T1/PD | Cor fluid | Sag T1/PD | Sag fluid |
|---|---:|---:|---:|---:|---:|
| ACL | 15.3 | 8.6 | 19.0 | 19.1 | **38.0** |
| MCL | 26.5 | 17.5 | **34.9** | 6.5 | 14.5 |
| Medial meniscus | 3.3 | 18.1 | **32.9** | 17.7 | 28.0 |
| Lateral meniscus | 6.3 | 22.2 | **31.3** | 17.3 | 22.9 |
| Medial OA | 13.9 | **25.8** | 23.8 | 17.4 | 19.0 |
| Lateral OA | 16.5 | 20.1 | 16.0 | **24.0** | 23.5 |
| PF OA | 29.4 | 7.8 | 8.6 | 22.7 | **31.6** |
| Effusion | **39.6** | 2.9 | 18.5 | 7.7 | 31.3 |
| Synovitis | **34.9** | 7.5 | 17.4 | 11.6 | 28.6 |
| Baker's | **54.2** | 2.0 | 10.9 | 6.8 | 26.0 |
| Contusion | 28.1 | 5.7 | **31.8** | 4.9 | 29.5 |
| Fracture | 26.4 | 15.2 | **27.3** | 7.6 | 23.5 |

In group (a), axial T1/PD takes 20.0% of effusion, 10.8% of synovitis, 11.1% of lateral OA,
8.8% of MCL and Baker's, 4.9% of ACL, and at most 1.4% of the menisci and contusion. The other
per-label tables are in `type_label_summary.csv`.

## Reading

- The pattern agrees with the [removal and attention evidence](2026-09-17-plane-contrast-importance.md):
  - *Sagittal fluid:* ACL.
  - *Axial fluid:* Baker's cyst, effusion and synovitis.
  - *Coronal fluid:* MCL, the menisci, contusion and fracture.
- T1/PD types still lead some labels: coronal T1/PD for medial OA and sagittal T1/PD for lateral
  OA. An allocation that follows only the macro shares could starve them.
- A split of group (b)'s 26 training triplets in proportion to the macro separation share would be
  about 6.4/3.3/5.9/3.5/6.9 (axial fluid, coronal T1/PD, coronal fluid, sagittal T1/PD, sagittal
  fluid) against today's 6/5/5/5/5. This is an illustration, not a proposal.

## Caveats

- These are average shares at today's budgets: they measure reliance, not the value of one more
  triplet of a type.
- Tokens of one type compete in a shared softmax, so a type's share partly follows its token
  count.
- **Noisy cells:** seven cells swing 10–16 points across runs: ACL and MCL on axial fluid in (b);
  ACL and effusion on axial fluid, MCL and effusion on axial T1/PD, and MCL on coronal T1/PD in
  (a). Group (a) has fewer than 20 positives for synovitis (17), Baker's cyst (13) and fracture
  (9).
- Axial T1/PD availability is not random
  ([availability shift](2026-09-17-plane-contrast-importance.md#availability-and-prevalence-shift)),
  so groups (a) and (b) differ in population as well as in types.
