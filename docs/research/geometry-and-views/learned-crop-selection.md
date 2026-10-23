# Learned crop selection

Researched on 2026-09-16 with a multi-agent web review:

- **Search:** 5 search angles and 20 primary sources fetched.
- **Claims:** 100 extracted. The top 25 were checked by 3 adversarial voters each, with 14
  confirmed and 11 refuted.
- **Scope:** read against this repository's first two crop-selection runs and their crop
  inspection.
- **Sources:** 2019 to 2024.

The investigation was dropped later that day. The
[investigation record](../../history/crop-selection-investigation.md) covers the later runs and
their conclusion. References to "our design" below mean the dropped scout design.

## Answer

Learned crop selection is not the largest available gain for this competition, and the
implemented design is more elaborate than its results justify.
- **Literature:** learned patch selection adds at most about +0.02 AUC over strong
  whole-image baselines, even for tiny mammography lesions. Its main documented benefit is
  memory and compute. It is fragile to train with weak labels.
- **Kaggle:** winning RSNA MRI/CT solutions localized with supervision (segmentation masks
  or coordinates) and cut fixed anatomical windows. None learned placement from study
  labels.
- **Here:** the fundamentals are not refuted, since more detail on the joint helped detail
  targets. The direction signal is what failed.

## Local evidence

- **Results:** one crop per canvas scored 0.887151
  ([crop-selection result](../../validation/results/crops-and-views/2026-09-15-crop-selection-result.md)). Two crops per
  canvas scored 0.889409 with 1.78× training time
  ([two-crops result](../../validation/results/crops-and-views/2026-09-15-two-crops-result.md)). The champion scores
  0.892360.
- **Placement:** learned placement beats random and centered crops by about 0.005, but
  crop centers differ by at most 4 px (2 mm) between positive and negative studies for
  every target and plane.
- **The value map is flat across slices:** the top canvas holds 3.1% of a study's value,
  and crops per canvas already track canvas value (within-study Spearman 0.87). See the
  [crop inspection](../../validation/results/crops-and-views/2026-09-15-two-crops-result.md#crop-inspection).
- **Averaging:** two-crops averaged with the champion gives +0.003, but any same-recipe run
  averaged with the champion gives +0.003 to +0.004.

## Learned patch selection in the literature

- **Small gains over strong baselines** (high confidence).
  - GMIC beat a ResNet-34 with top-t% MIL pooling by +0.026 malignant AUC (0.915 vs 0.889;
    +0.020 with a matched backbone), within the box-supervised Faster R-CNN's spread
    (0.908 ± 0.014) ([Shen et al.](https://arxiv.org/abs/2002.07613)).
  - 3D-GMIC's local module added +0.020 on DBT (0.811 → 0.831, 53 malignant cases,
    overlapping CIs), and it averages global and local predictions, so part of that is
    ensembling ([3D-GMIC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11449265/)).
  - Attention sampling tied full-resolution Deep MIL on histopathology error (0.093 vs
    0.092/0.093) at about 10× speed ([Katharopoulos & Fleuret](https://arxiv.org/abs/1905.03711)).
- **Why these methods exist** (medium confidence): very large inputs with tiny objects,
  where a full-resolution network does not fit. 3D-GMIC's patches cover 0.26% of the volume
  and use 78–90% less memory. The champion already sees the full 140 mm view, and stored
  pixels are about 2× interpolated ([input resolution](input-resolution.md)).
- **Fragile training** (medium confidence):
  - The differentiable top-k authors report collapse to the majority class and a scorer
    confidently selecting wrong patches ([Cordonnier et al.](https://arxiv.org/html/2104.03059)).
  - Iterative patch selection still overfits at low signal-to-noise ratio
    ([Riffi-Aslett & Fell](https://arxiv.org/html/2412.11237)).
  - Our flat draft AUC and label-independent layouts fit these failure modes.
- **GMIC's working parts that this design lacks** (high confidence for the loss):
  - The low-capacity saliency branch has its own BCE through top-t% pooling.
  - An L1 sparsity penalty applies to the saliency maps.
  - Global and local predictions are fused rather than replaced
    (`L = Σ BCE_local + BCE_global + BCE_fusion + β·L1`).
  - Verifiers reported GMIC ablations of random patches 0.629 vs saliency-selected 0.898,
    global only 0.892, local only 0.897 and learned fusion 0.913. These figures were not
    independently re-checked, and a claim that most of the gain comes from fusion was
    refuted as worded.

## Global-local branch fusion

Added on 2026-09-16 from [GLNet](https://arxiv.org/abs/1905.06368) (Chen et al., CVPR 2019 oral)
and the 188 papers citing it, listed through the Semantic Scholar graph API on the same date. Only
the GLNet abstract was fetched; the citing papers are pointers, and their results are unchecked.

- **GLNet** pairs a global branch on the downsampled whole image with a local branch on cropped
  patches, for segmenting images up to 30 megapixels under 2 GB of GPU memory on one 1080Ti. Its
  contribution is not the two branches but that it **deeply fuses their feature maps in both
  directions at several levels**, plus a coarse-to-fine variant that uses the global branch to
  handle foreground/background imbalance. Datasets are aerial and medical segmentation.
- **The line continues in segmentation:** ISDNet (shallow and deep networks, 2022), MFVNet
  (multiple fields of view, 2023) and the Guided Patch-Grouping Wavelet Transformer
  ([2307.00711](https://arxiv.org/abs/2307.00711)) all keep a low-resolution context stream beside a
  high-resolution one. The nearest weakly supervised relative is DSNet, a dual-stream gigapixel
  pathology model ([2109.05788](https://arxiv.org/abs/2109.05788)).
- **What this says about our design.** Our scout and crop branches never exchange features: they
  meet only in union attention pooling, plus the coupling that adds a fine token's underlying
  coarse score. That is late fusion of scores. GMIC's fused loss and GLNet's fused features are
  both evidence that combining the two views inside the network, rather than at the end, is where
  the reported gains come from.
- **Concrete follow-up:** project the scout feature map, sample it at the crop's cells and add it
  to an early or middle ReXNet stage. Fine token anchors already land exactly on coarse cells, so
  the alignment costs no new geometry.
- **Transfer caveats:** these are dense per-pixel tasks with full supervision, so their fusion
  gradients are far better conditioned than twelve study-level labels; they tile patches to cover
  the image while we select a few; and their memory argument does not apply, since our bottleneck
  is loader CPU, not GPU memory.

## RSNA Kaggle winners

- **Supervised localization, then crops** (high confidence):
  - **Cervical spine 2022:** the top 3 used segmentation, a 2D CNN and an RNN
    ([RSNA paper](https://pubs.rsna.org/doi/10.1148/ryai.230256)).
  - **Abdominal trauma 2023:**
    - 1st place cropped studies with 3D segmentation masks
      ([1st place](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution)).
    - 2nd place used a supervised 3D ResNet18 U-Net organ segmenter feeding a 2D CNN and RNN
      on crops ([2nd place](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma)).
  - **Lumbar spine 2024:** 1st place regressed slice index and (x, y) from the provided
    coordinates; 2nd place used 2.5D coordinate regression
    ([1st place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution),
    [2nd place](https://github.com/brendanartley/RSNA-2024-Competition)).
  - All of these competitions supplied masks, boxes or coordinates. This competition does
    not.
- **Crop-only gains are unmeasured** (medium confidence). The abdominal 2nd place improved
  private log loss from 0.39 to 0.35 when adding crop models, but the backbone and RNN
  changed in the same step. No winner published a no-crop baseline.
- **Crops were fixed windows around predicted anchors** (medium confidence, single source).
  - Lumbar 1st place resized to 512 px, then cut fixed offsets (e.g. 128 × 80 px sagittal)
    over 5 slices centered on the predicted slice.
  - Training jitter of ±10 px and ±2 slices was called crucial.
  - The transferable recipe is anchor → fixed anatomical window → jitter → 2.5D stack.

## Ranked options

Expected gains are extrapolations from the evidence above, not verified measurements.

| Rank | Option | Expected macro AUC | Cost |
|---|---|---|---|
| 1 | Ensembles of diverse runs, folds and seeds | +0.003 to +0.008 | Low |
| 2 | Explicit localization (anatomical landmarks, trained localizers or pseudo-labels) with fixed per-structure crops, fused with the champion's global tokens | 0 to +0.01 | High |
| 3 | GMIC-style repair of the learned selector: BCE on the coarse-token draft, a sparsity or slice-budget prior, and global/local fusion inside the network ([branch fusion](#global-local-branch-fusion)) | 0 to +0.005 | Medium |
| 4 | Stop the two-crops variant | — | — |

**Overengineered for its return:**
- The draft-attention × `ŷ(1−ŷ)` × head-norm value map with greedy coverage, trained only
  by the final BCE.
- The two-crops variant.
- A MobileNetV3-small scout standing in for global context the champion already has.

## Caveats and open questions

- **Knee-specific evidence was not verified.** No claim on region crops versus full view in
  knee MRI, MRI-pretrained encoders, or localizers from public knee datasets (OAI-ZIB,
  SKM-TEA, fastMRI+) reached verification. Ten extracted claims were dropped by the
  25-claim budget.
- **Transfer is not established.** The evidence comes from mammography, histopathology,
  synthetic images and CT/spine competitions with provided localization.
- **Refuted as worded:**
  - "learned selection helps most when content is small";
  - the iterative patch selection 98.4% vs 84.1% traffic-sign comparison;
  - the lumbar 2nd-place localization accuracy and crop-size claims;
  - "an auxiliary Dice loss was the largest single gain" in abdominal 1st place;
  - GMIC's "most of the gain comes from fusion".
- **Open questions:**
  - Would a scout with its own loss and a sparsity prior make layouts differ between
    positive and negative studies?
  - Can public knee segmenters give per-structure crops (meniscal horns, ACL, MCL,
    posteromedial region) on this competition's series?
  - Are the per-target deltas versus the champion real across folds?
