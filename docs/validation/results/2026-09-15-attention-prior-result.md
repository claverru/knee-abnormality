# 2026-09-15 attention-prior result

The prior was [retired](../../history/model-lineage.md#retired-attention-prior) on
2026-09-16; `configs/attention-prior.yaml` and its implementation are recoverable from
`outputs/training/attention-prior/source_snapshot.zip`.

**A regression against its matched control.** The `attention-prior` run
(`configs/attention-prior.yaml`: graph off, no additive 3D embedding,
`model.attention_prior: gaussian_1d_2d_3d`) selected epoch 7 (index 6) at **0.890586**
fold-0 macro AUC. Its control, the overnight `00-baseline`, uses the same recipe without
the prior and scored 0.892763, so the prior is **−0.002177** worse. It is also 0.001774
below the reflection/sigma-40 champion's 0.892360. There is no public score, and the
[encoding diagnostic](../manual-encoding-usage-diagnosis.md) has not been run on it.

| Run | Best macro AUC (epoch) | vs `00-baseline` | Soft-target BCE | Gold (11) | Last epoch |
|---|---:|---:|---:|---:|---:|
| `00-baseline` (control) | 0.892763 (10) | — | 0.44767 | 0.9320 | 0.892763 |
| `attention-prior` | 0.890586 (7) | −0.002177 | 0.45260 | 0.9295 | 0.890516 |

The prior trailed the control at most epochs, for example 0.8906 against 0.8927 at epoch 7
and 0.8905 against 0.8928 at epoch 10, and plateaued from epoch 7. Training took 1,686 s
with peak allocation 20.02 GB.

The learned priors moved little from their `mu = 0`, `sigma = 2` (200 mm) initialization.
Across all terms, `mu` changed by at most 0.17 and `log sigma` by at most 0.16, in
100 mm units. The 3D sigmas stayed at 170–215 mm, so the bias varies by only about
0.06 logits within 70 mm of the study centroid. Directions were anatomically plausible
but weak:

- Baker's cyst shifted posterior (`mu_y` +15 mm).
- Effusion and synovitis shifted superior (`mu_z` +11 and +10 mm).
- PF OA shifted anterior (`mu_y` −10 mm).
- ACL moved toward the centre (`mu_|x|` −13 mm).

Such nearly flat biases are unlikely to explain the regression by their direct effect;
changed training dynamics or run-to-run variability may contribute. This is one seed on
fold 0. Artifacts are in `outputs/training/attention-prior/`.

## Encoding diagnostic: the priors are used, weakly

The [encoding diagnostic](../manual-encoding-usage-diagnosis.md) on the selected checkpoint
reproduced the saved probabilities (maximum difference 3e−8) with no preparation
conditions. Results are in `outputs/analysis/encoding-usage-attention-prior/`.

| Variant | Macro AUC change | Mean absolute probability change | Paired 95% interval |
|---|---:|---:|---|
| Without all priors | −0.000407 | 0.0038 | [−0.00062, −0.00020] |
| Without 2D prior | −0.000258 | 0.0023 | [−0.00041, −0.00011] |
| Without 3D prior | −0.000126 | 0.0013 | [−0.00023, −0.00003] |
| Without 1D prior | +0.000006 | 0.0004 | [−0.00006, +0.00006] |
| Without 1D embedding | −0.000672 | 0.0477 | [−0.00137, +0.00005] |
| Without all position (embeddings and priors) | −0.001864 | 0.0911 | [−0.00294, −0.00081] |

- **The priors help within this checkpoint.** Removing them lowers nine of twelve target
  AUCs, most on lateral meniscus (−0.0013), ACL (−0.0008) and contusion (−0.0007). The
  2D and 3D terms carry the effect; the 1D term is unused.
- **The effect is about five times smaller than the regression.** The priors add about
  0.0004, while the run scored 0.002177 below its control. The regression is therefore
  better explained by a different training trajectory, or by the priors changing what the
  rest of the head learned, than by the priors' direct inference-time effect.
- **The exclusive-positive OA margin is low** at 0.8389, against 0.878–0.879 for the
  champion and `graph-forced`, near the bottom of the earlier 0.837–0.897 fold-0 range.
  The meniscus margin is 0.9736.
