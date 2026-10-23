# Prior work

Where the seeds come from, the measured precedents outside autoresearch, and the closed
directions.

- **Selection review, 2026-09-25:** [MRI augmentation research](../../../docs/research/training-objectives/mri-augmentation.md) ranks the
  candidates, their evidence and the data facts that set each strength (acquired resolution,
  scanner mix, contrast heterogeneity, reconstruction filters). Its candidate 4 became three
  MixUp seeds that differ only in the mixing point (image, token, pooled output), by user decision
  on 2026-09-25. Seed order: GIN, resolution, sharpening, the three MixUp variants, late affine
  shutdown, bias field, channel reversal, bilateral smoothing, intra-study contrast mixing (added by the
  user on 2026-09-25), DSU, ReXNet stochastic depth (moved from the small-changes backlog by
  the user on 2026-09-25).
- **Seed noise:** re-seeding the adopted knee-geometry recipe (`adopted-new-seed` against
  `meniscus-half-knee-mil`) moved macro AUC by −0.0063; the largest per-label moves were
  Contusion −0.0205, PF OA −0.0154, Fracture −0.0137 and MCL −0.0129. Treat single-label moves of
  that size in those labels as noise.
- **Resolution sensitivity:** training and evaluating at 0.875 mm cost −0.0228 macro, lateral
  meniscus −0.106 and MCL −0.067
  ([learned zoom](../../../docs/validation/results/crops-and-views/2026-09-16-learned-zoom-result.md#per-target-auc)).
- **Bias field precondition:** bias-field augmentation was not selected in the 2026-09-21
  backlog review, for reasons the docs do not record, and residual coil shading was never
  measured. Seed 8 runs with a fixed ±20% amplitude without that measurement, as a user-ordered seed.

## Precedent table

Measured runs outside autoresearch on the existing families. All are fold 0, single runs,
against the control of their day; deltas are macro AUC.

| Mechanism | Setting | Δ | Source |
|---|---|---|---|
| Affine | off | −0.0041 [−0.0078, −0.0002] | [2026-09-17](../../../docs/validation/results/campaigns-and-queues/2026-09-17-overnight-results.md) |
| Affine | strong | −0.0024 | [2026-09-15](../../../docs/validation/results/campaigns-and-queues/2026-09-15-overnight-results.md) |
| Affine | mild (7°, 3%, ±3%) | +0.0015 | [2026-09-18](../../../docs/validation/results/campaigns-and-queues/2026-09-18-overnight-results.md) |
| Affine | probability 0.5 | −0.0019 | [2026-09-20](../../../docs/validation/results/campaigns-and-queues/2026-09-20-overnight-hyperparameter-results.md) |
| Affine | rotation 7° | −0.0005 | 2026-09-20 |
| Affine | scale 0.97–1.03 | +0.0026 / −0.0007 against two controls | 2026-09-20 |
| Gamma | 0.70–1.30 | −0.0007 | 2026-09-20 |
| Contrast | 0.80–1.20 | −0.0033 / −0.0067 against two controls | 2026-09-20 |
| Intensity | probability 0.25 | −0.0008 | [2026-09-19](../../../docs/validation/results/campaigns-and-queues/2026-09-19-untried-hyperparameter-queue.md) |
| Intensity | probability 0.75 | −0.0005 | 2026-09-20 |
| Intensity quantiles | 0.001/0.999; 0.01/0.99 | −0.0003; −0.0021 | 2026-09-17 |
| Gaussian noise | probability 0.3 | −0.0019 | 2026-09-17 |
| Gaussian noise | std 0.01–0.04 | +0.0011 | 2026-09-19 |
| Dropout | 0.1 | +0.0032 (09-18); −0.0005 (09-20) | 2026-09-18, 2026-09-20 |
| Series dropout | 0.15; 0.3 | +0.0030; −0.0005 | 2026-09-18, [2026-09-15](../../../docs/validation/results/campaigns-and-queues/2026-09-15-overnight-results.md) |
| Attention-token dropout | random, top-attention; 5% retest | adopted then lost at every τ | [near-misses](../../../docs/validation/results/campaigns-and-queues/2026-09-25-near-misses-campaign.md) |
| EMA, SWA | decay, last 3 epochs | ties; not adopted | 2026-09-17 |
| Image MixUp (public EfficientNet-B0) | — | 0.7914 → 0.7937, judged noise | [public research](../../../docs/research/competition-and-prior-art/public-research.md) |

## Closed directions

Broader gamma/contrast, Rician noise, weaker affine, slice-stride and slice-index jitter,
unaligned MixUp, CutMix/cutout, elastic distortion, RandConv, MixStyle, fat-suppression failure
simulation, local pixel shuffle, CLAHE, AugMix/RandAugment, two-view consistency losses,
percentile clip-bound jitter and channel-blending thickness simulation; reasons in
[MRI augmentation research](../../../docs/research/training-objectives/mri-augmentation.md#rejected-or-deferred).
