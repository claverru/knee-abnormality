# MRI augmentation candidates

**Selection research, 2026-09-25.** The evidence and data facts behind the
[augmentation campaign](../../autoresearch/campaigns/augmentation/INDEX.md); outcomes are in the
[campaign result](../validation/results/2026-09-26-augmentation-campaign.md). Adopted: bilateral
smoothing, DSU, intra-study contrast mixing, GIN and one-axis resolution degradation. Sources: an
external proposal review (`mri-augmentation-proposals.md`, 2026-09-25), a literature and Kaggle
review, and the repository's measured data profile. Rebuilt on 2026-09-26 from the working
document, which was lost before it was committed.

## Pipeline before the campaign

Study reflection (p 0.5), one affine per acquisition (p 0.7, ±10°, ±4%, scale 0.95–1.05),
gamma/contrast (p 0.5, 0.85–1.15 / 0.9–1.1) and Gaussian noise (p 0.15, std 0.005–0.02), on the GPU
after the uint8 crop is scaled to `[0, 1]` and before ImageNet normalization, with parameters
shared by every triplet of an acquisition
([schedule and augmentation](../modeling/schedule-and-augmentation.md#ten-epoch-schedule-and-augmentation)).
Series dropout (0.2) and epoch-dependent triplet sampling already varied slice selection and
protocol coverage.

## Data facts that set the candidates

- **Scanner mix:** 4,407 train studies from seven vendors and 46 machine models: Siemens 1,951,
  Philips 1,301, GE 905, Toshiba 181, Canon 45, Fujifilm 16, Hitachi 8; 1.5 T 2,543, 3 T 1,601
  ([vendor associations](../metadata/vendor-label-associations.md),
  [validation protocol](../validation/project-validation-protocol.md)).
- **Acquired in-plane resolution** against the 0.625 mm grid
  (`outputs/analysis/i/input-resolution-research-20260915/acquisition/distributions_train.csv`):
  finer axis p5/p50/p95 0.33/0.49/0.70 mm; coarser (phase) axis 0.45/0.625/0.885 mm, p75 0.70 mm,
  GE the coarsest. About a quarter of series are effectively coarser than the grid along one axis
  ([input resolution](input-resolution.md)).
- **Resolution matters for detail labels:** training and evaluating at 0.875 mm cost −0.0228
  macro, lateral meniscus −0.106 and MCL −0.067
  ([learned zoom](../validation/results/2026-09-16-learned-zoom-result.md#per-target-auc)), so
  resolution augmentation stays inside the measured range.
- **Through-plane:** slice thickness p5/p50/p95 2.5/3.0/4.0 mm, spacing 3.0/3.5/4.95 mm.
- **Contrast heterogeneity inside a slot:** 19% of T1/PD-slot series are T2 without fat
  suppression, 129 T1-FS series carry non-FS flags and 130 GE zero-TE sagittals have flat near-PD
  contrast ([series census](../metadata/series-filtering-census.md)). The fat-suppressed fluid look
  carries effusion, contusion and marrow signal.
- **Reconstruction heterogeneity:** GE deep-learning reconstructions and Philips reformats occur
  as duplicate pairs ([current preparation](../data-preparation/current-preparation.md)).
- **Motion:** at most 2 of 4,407 training reports describe motion or ghosting (researcher check,
  2026-09-26), so motion augmentation has no premise.

## Candidates and evidence

| Candidate | Evidence | Setting tested |
|---|---|---|
| GIN, random shallow conv intensity remap | Isolated, strongest found: on top of affine, elastic, brightness, contrast, gamma and noise, cross-centre prostate MRI Dice 56.6 → 65.1 and cardiac bSSFP → LGE 76.0 → 83.8; depth 2/4/8/16 gave 84.3/86.0/85.7/79.2 ([Ouyang et al.](https://arxiv.org/abs/2111.12525)); segmentation | p 0.3, blend ≤ 0.5, 4 layers, 2 hidden channels, kernel 1 or 3, one net per acquisition, 1-channel |
| One-axis k-space resolution degradation | nnU-Net recipe component, not isolated ([trainer](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py)); measured phase-axis mismatch | p 0.25, keep 70–100% of frequencies on one axis |
| Unsharp sharpening | Best single MRI transform in BigAug against a crop-only baseline ([Zhang et al.](https://arxiv.org/abs/1906.03347)); RSNA 2024 lumbar 9th-place recipe | p 0.25, σ 0.5–1.0 px, s 0.2–1.0 |
| MixUp (image, token, pooled output) | RSNA 2024 lumbar 2nd place (MixUp p 0.3, α 1, plus manifold MixUp), 3rd and 6th place, never isolated; knee cartilage MixUp across scanners ([Panfilov et al.](https://arxiv.org/abs/1908.04126)) | 30% of studies, λ ~ Beta(1, 1), same knee side and reflection draw, slot- and depth-matched, two-target loss including the auxiliary heads |
| Bias field | Mechanism only (TorchIO, MONAI) | p 0.25, cubic polynomial, ±20% |
| Channel reversal (axial/coronal) | Kaggle author claims only | p 0.5 |
| Bilateral smoothing | Guess; premise from deep-learning reconstructions | p 0.2, 5×5, spatial σ 1 px, range σ 0.05–0.1 |
| Intra-study contrast mixing | Guess; premise from within-slot contrast heterogeneity | p 0.3 per plane with both contrasts, target weight 0.5–1.0 |
| Late affine shutdown | Hypothesis: affine off (−0.0041) and strong affine (−0.0024) both lost | affine off for epochs 8–9 |
| DSU feature statistics | Natural-image domain generalization only ([Li et al.](https://arxiv.org/abs/2202.03958)) | p 0.5 at the stride-8 stage |
| Stochastic depth | Standard timm regularizer; never measured on ReXNet here | drop path 0.1 |

## Rejected or deferred

| Candidate | Reason |
|---|---|
| Broader gamma/contrast, separate gates | Measured: contrast 0.8–1.2 lost −0.0033 and −0.0067, gamma 0.7–1.3 −0.0007, intensity p 0.25/0.75 flat ([results](../validation/results/2026-09-20-overnight-hyperparameter-results.md)) |
| Series-dropout rate | A hyperparameter (0.15, 0.3 measured) |
| Coverage-aware series dropout | Only a brain-segmentation preprint |
| Rician noise | Replaces the Gaussian branch; noise changes measured flat |
| Motion, ghosting, spike, Gibbs as separate families | Library availability only; no motion premise; Gibbs ringing comes with k-space resolution degradation |
| Auxiliary Fourier Augmentation | Needs an extra forward view |
| Weaker affine | Measured: affine p 0.5 −0.0019, affine off −0.0041 |
| Slice-stride jitter | Stride 2 on 3 mm spacing exceeds the measured p95 of 4.95 mm |
| Slice-index jitter | Already covered by epoch-dependent triplet sampling |
| Unaligned image MixUp, CutMix, cutout | Mixing across knee sides or unmatched slots breaks the anatomical geometry; cutout can remove the only positive evidence |
| Elastic or grid distortion | Needs a dense coordinate update the geometry mapping does not support |
| RandConv | Dominated by GIN in the same comparisons |
| Monotone Bézier curves | Mixed ([SLAug](https://arxiv.org/abs/2211.14805)); a fallback if GIN's non-monotonic curves hurt |
| MixStyle | Roughly zero in the GIN and SLAug MRI tables |
| Anisotropic in-plane scale | A setting of the existing affine family |
| Two-view consistency loss | Needs a second forward pass |
| Percentile clip-bound jitter | Changes the preprocessing identity; bounds already vary between draws |
| Slice-thickness simulation by blending channels | Exceeds the measured thickness range |
| Fat-suppression failure simulation | No evidence, and bright marrow is confounded with contusion and edema |
| Local pixel shuffle, CLAHE, AugMix, RandAugment | No isolated MRI classification evidence |
| Dilated stride-16 backbone | 3× backbone time and 24 GiB ([measurement](multiscale-features-knee-mri.md#a-dilated-stride-16-backbone-costs-three-times-the-backbone)) |
