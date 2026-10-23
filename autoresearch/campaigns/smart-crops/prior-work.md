# Smart-crops prior work

Crop, resolution and multi-view evidence the seeds build on. Sources: the
[crop-selection investigation](../../../docs/history/crop-selection-investigation.md),
[input resolution](../../../docs/research/geometry-and-views/input-resolution.md),
[learned crop selection](../../../docs/research/geometry-and-views/learned-crop-selection.md), the
[multi-scale review](../../../docs/research/geometry-and-views/multiscale-features-knee-mri.md), the
[mixed-field campaign and the 120 mm adoption](../../../docs/validation/results/crops-and-views/2026-10-01-mixed-field-campaign.md),
two probes of the 140 mm control and a literature review, both of 2026-10-01.

## Precedent table

Deltas are ×4 macro AUC against each run's own control and are not comparable across rows. Every
row is one run. The mixed-field rows are against `rn18-l2sp-no-confusion` (140 mm, 0.928062).

| Mechanism | Setting | Run or source | Macro delta | Notes |
|---|---|---|---:|---|
| One 0.5 mm crop per slice, scout-placed | 80 mm at 160 px replaces the full slice, ReXNet-200 | crop-selection, 160 px | −0.0056 | Old labels and split |
| 0.5 mm crops on 75% of slices, scout-placed | 112 mm at 224 px | per-label crops | −0.0011 | ACL +0.0111, synovitis +0.0077, medial OA +0.0057; lateral meniscus −0.0102, fracture −0.0085 |
| Coarse context on every slice plus fine glimpses | 0.875 mm context, 0.42 mm glimpses | learned zoom | −0.0228 | Glimpses took 2% of attention; removing them changed nothing |
| Scout-placed against fixed or random crops | 160 px crops | crop-selection placement ablation | about +0.005 | Crop centres differed by at most 2 mm between positive and negative studies; scout attention entropy 0.97–0.99 |
| Joint-line-anchored 140 mm window | one moved window | sampling campaign | −0.0010 | PF OA −0.011, contusion −0.017 |
| Single 120 mm field | 0.536 mm, every slice | `field-120-single` | +0.0011, 80% [−0.0004, +0.0026] | PF OA, ACL and the OA labels gain; synovitis, Baker's cyst and the medial meniscus fall |
| The same with middle-slice augmentation | | `field-120-middle-slice-augmentation-20261001` | +0.0012, 80% [−0.0003, +0.0028] | 0.947 public (S90) against 0.943 (S86); adopted |
| 120 mm with plane-contrast readouts and no anatomical bias | the campaign reference, the recipe in force | `rn18-field-120-control` (0.930666) | +0.0014 against the run above, 80% [+0.0006, +0.0022]; +0.0008 against the same recipe at 140 mm (`no-anatomical-bias`), 80% [−0.0006, +0.0022] | Against 140 mm: lateral meniscus +0.0080, ACL +0.0046, PF OA +0.0041; synovitis −0.0044, Baker's cyst −0.0030, medial meniscus −0.0018 |
| Single 160 mm field | 0.714 mm, every slice | `field-160-single-logged` | −0.0034, 80% [−0.0051, −0.0019] | 160 mm added nothing in five designs |
| T1/PD at 112 mm, fluid at 140 mm | 0.5 mm on T1/PD, all planes | `contrast-split-112-140` | +0.0015, 80% [+0.0004, +0.0026] | Ties 120 mm alone (−0.0003, 80% [−0.0017, +0.0011]); 0.945 public (S91); not adopted. Alone, its 112 mm tokens score 0.859 and its 140 mm ones 0.916 at ×1 |
| 120 and 160 mm alternating, shared backbone | field embedding | `mixed-field-120-160` | +0.0006, 80% [−0.0007, +0.0018] | |
| The same with per-field readout weights | | `mixed-field-readout-weights` | −0.0001, 80% [−0.0016, +0.0013] | Readout weights per view bought nothing |
| The same with one ResNet-18 per field | | `field-backbones-routed` | −0.0012; −0.0017 against the shared mix, 80% [−0.0028, −0.0007] | A second backbone loses |
| Every slice at 120 mm (ResNet-18) and 140 mm (ResNet-10t) | second stream behind a zero gate | `two-stream-paired` | +0.0023, 80% [+0.0007, +0.0039] | Second stream never trained: 0.506 alone, joint equal to narrow-only at every epoch; in effect 120 mm alone. 1,736 s against 1,222 s |
| Dense per-plane readouts with their own loss | shared ResNet-18 | `dense-plane-readouts` (revisits) | −0.0052 | The local case against per-view losses |
| Per-scale readouts with their own loss | stride-16 and stride-32 tokens, ReXNet-200 | `fusion-scaleheads` | +0.0006 | Tie |
| More slices per study | 36 against 26 | `rn18-more-triplets` | +0.0018 | Deferred to the endgame |

## Acquired resolution

Share of train series whose finer acquired axis is kept in full at each spacing
(`outputs/analysis/i/input-resolution-research-20260915/acquisition/series_acquisition.parquet`,
counted 2026-10-01):

| Series type | Series | 0.536 mm | 0.5 mm | 0.4167 mm | 0.375 mm |
|---|---:|---:|---:|---:|---:|
| Coronal T1/PD | 3,984 | 18% | 35% | 76% | 92% |
| Sagittal T1/PD | 4,959 | 23% | 37% | 73% | 90% |
| Coronal fluid | 4,384 | 37% | 60% | 91% | 94% |
| Sagittal fluid | 4,192 | 44% | 60% | 89% | 92% |

T1/PD series are acquired finer (finer-axis median 0.45 mm against 0.50 mm). 0.375 mm is the
spacing at which nine in ten of them are fully resolved; 84 mm at 224 px and 60 mm at 160 px are
that spacing. The wide view's 0.536 mm resolves about a fifth of them.

## What the 120 mm field already cuts

From [input resolution](../../../docs/research/geometry-and-views/input-resolution.md#what-a-narrow-field-cuts)
(292 series, middle slice):

- **Coronal:** more than 10 mm of tissue is cut on a side in 30–31% of series.
- **Sagittal:** more than 20 mm of popliteal tissue is cut in 34%, and more than 10 mm anteriorly
  in 38%.
- **Below the joint:** under 40 mm of tibia is left in 22–28% of series.
- **The joint itself stays inside** on every knee-centred contact-sheet tile: condyles, plateau,
  menisci and cruciates.
- **Medial meniscus:** it fell in every run that used 120 mm (−0.002 to −0.007); cause not
  established. A narrower view of coronal series cuts further into the same region.

## Where the model looks

Attention probe of `no-anatomical-bias`, the reference's recipe at 140 mm, not the reference
itself: fold 0, ×1, 912 validation studies, attention temperature 1
(`outputs/analysis/l/label-attention-20261001/x1`; [scripts](../../../scripts/adhoc/INDEX.md)).
Attention mass is not evidence, tokens there are 20 mm wide and their receptive fields cover most
of the image. Extents are knee-centred millimetres and carry over to the 120 mm field; nothing has
been measured on a 120 mm model.

- **Sharpness:** within-slice normalised entropy is 0.67–0.86 per label; the September scout's was
  0.97–0.99.
- **Planes and contrasts:**
  - ACL and the menisci keep 85–93% of their attention on coronal and sagittal slices.
  - Axial fluid carries 47% of Baker's cyst, 41% of effusion and 36% of synovitis.
  - Per slice, against a uniform share, the detail labels put 1.5 on sagittal fluid, 1.1 on
    coronal fluid, 1.0 on sagittal T1/PD and 0.9 on coronal T1/PD.
  - The periphery labels put 0.6–0.8 on coronal and sagittal T1/PD and 1.0–1.4 on fluid series.
- **Extent:** the detail labels' attention spans 100 mm medial to lateral in coronal slices and
  80 mm front to back in sagittal ones (5th to 95th percentile, knee-centred), about 80–90 and
  60–70 mm of structure once 10 mm of token blur is taken off each side.
- **Window and placement**, share of a label's total attention inside one window per coronal and
  sagittal slice:

  | Label | 60 mm, one position for all studies | 60 mm, per study and plane | 80 mm, one position | 80 mm, per study and plane |
  |---|---:|---:|---:|---:|
  | ACL | 0.55 | 0.59 | 0.71 | 0.74 |
  | Medial meniscus | 0.56 | 0.75 | 0.74 | 0.88 |
  | Lateral meniscus | 0.49 | 0.60 | 0.65 | 0.76 |
  | Medial OA | 0.43 | 0.51 | 0.60 | 0.65 |
  | Lateral OA | 0.46 | 0.50 | 0.62 | 0.64 |
  | PF OA | 0.41 | 0.43 | 0.52 | 0.54 |

  One window per study and plane keeps within 0.02–0.07 of one per slice. The meniscus attention
  sits 20–28 mm either side of the fitted midline.
- **Anatomy, not lesions:** the attention centre moves under 7 mm between positive and negative
  studies for ten labels; Baker's cyst moves 15–18 mm and effusion 5–9 mm.
- **Coverage:** 80% of a label's attention spreads over 6–16 of the 26 slices.
- **Rule position:**
  - The fitted knee centre is close to the image centre: a median 2.2 mm apart across a coronal
    slice (8.7 mm at the 95th percentile) and 2.7 mm front to back in a sagittal one (10.7 mm).
    The rule therefore moves most windows by a few millimetres and a twentieth of them by about
    10 mm.
  - Geometry is trusted on 96–97% of slices.
  - The detail labels' input sensitivity is centred 8–13 mm below the image centre and 5 mm
    anterior of the fitted centre in sagittal slices (input gradients, 300 studies,
    `outputs/analysis/l/label-attention-20261001/saliency`); the attention midpoint is 9 mm
    anterior. The rule takes 10 mm below and 7 mm anterior.
- **Contrast does not move the window:** within a plane the percentiles of that sensitivity agree
  within 1–4 mm between T1/PD and fluid series.
- **Smaller windows lose steadily.** Share of the detail labels' input sensitivity inside a window
  at the rule position, out of a 140 mm slice (a diffuse measure, read as relative):

  | Window (mm) | 93 × 93 | 93 × 67 | 80 × 67 | 67 × 67 | 60 × 60 |
  |---|---:|---:|---:|---:|---:|
  | Coronal | 0.62 | 0.52 | 0.47 | 0.40 | 0.33 |
  | Sagittal | 0.67 | 0.54 | 0.49 | 0.44 | 0.37 |

  There is no size below which the loss stops, so the 60 mm crop needs its placement to earn its
  size.

## Literature

Reviewed 2026-10-01; natural images and video unless marked. Earlier coverage (GMIC, attention
sampling, differentiable top-k, GLNet, RSNA winners) is in
[learned crop selection](../../../docs/research/geometry-and-views/learned-crop-selection.md).

- **Placement trained only by the final loss is about as good as random:** NTS-Net without its
  ranking loss scores below its baseline (83.3 against 84.5); regressed crops match random in
  [Learning to Zoom](https://arxiv.org/abs/1809.03355) and
  [AdaptiveNN](https://arxiv.org/abs/2509.15333).
- **What makes it work:**
  - a strong global branch as the locator;
  - staged training ([AG-CNN](https://arxiv.org/abs/1801.09927), chest X-ray: 0.868 staged against
    0.849 joint);
  - a stop-gradient from the placement into the global encoder, per-branch losses and a
    random-crop pass ([AdaFocusV2](https://arxiv.org/abs/2112.14238): 69.4 → 78.7 mAP);
  - crop jitter, called crucial by the RSNA lumbar winners.
- **A crop needs its own loss to be used:** [GFNet](https://arxiv.org/abs/2010.05300) +2.0,
  [TNet](https://arxiv.org/abs/2102.10212) +7.9, AdaFocusV2 +3.1 points.
- **Size of the gain:** AG-CNN gained 0.027 AUC, about 0.010 of it from ensembling two branches.
  In [TASN](https://arxiv.org/abs/1903.06150) placement added 1.2 points over a uniform sampler,
  which itself added 4.2 over the baseline.
- **Diffuse findings lose from cropping** (AG-CNN, pneumonia −0.067 in the local branch).
- **Knee MRI:** no paper measures a higher-resolution crop.
  [Tack 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8675251/) raised the medial anterior horn
  from 0.74 to 0.87 with a segmentation-derived crop at the same resolution (different encoders);
  [MRPyrNet](https://proceedings.mlr.press/v143/dunnhofer21a/dunnhofer21a.pdf) gained 0.02–0.05 on
  MRNet with label-free nested centre pooling.

## Code facts the seeds rely on

- **Preparation:** training and validation resample images on CPU in the loader
  (`collate_prepared`) and ship 224 px uint8 `images`; only Kaggle prediction resamples on the GPU.
  The model never sees native pixels in training.
- **Middle slice:** with `repeated_middle` the model selects the middle slice before reflection,
  contrast mixing and augmentation. The loader still reads and resamples all three slices of an
  input.
- **Window offset:** `_crop_geometry` centres every crop. An offset has to enter `resample_plan`'s
  key and `output_to_native`; `patient_maps` follow, and with them the medial and anterior axes,
  the meniscus own-side mask, the OA regions and the half-knee loss.
- **Native rows:** `decode_series` reads the native rows of the 120 mm window once; a window inside
  it needs only its own resample plan.
- **Slice selection** takes its in-plane radius from `image_size × spacing_mm / 2`, 60 mm.
- **Intensity:** series bounds are quantiles of the prepared pixels of that series.
- **Single input shape:** `images` is one square tensor, `encoded` is rectangular
  `[images, cells]` with five call sites expanding over it (`pool_encoded`, `half_knee_mil_loss`,
  the report-attribute and scanner-confusion steps, and the reshape in
  `PlaneContrastReadout.forward`), and `use_fixed_backbone_chunks` captures one inference graph
  at one size.
- **Positions:** `position_2d` is one embedding per grid cell from the anchor normalised in the
  input, shared by all images. ResNet anchors sit at pixels 0, 32, …, 192, 17.1 mm apart in the
  wide view.
- **Contrast mixing** registers the two series of a plane through their patient maps, so it holds
  between views of different spacing.
- **Readout:** `PlaneContrastReadout` scores each token with the row of its series type, so a
  view's tokens need no readout of their own.
- **Cost of the reference:** 1,149 s of training (about 115 s per epoch), 3.4 GiB peak.
- **Time follows backbone compute, not the image count:** `two-stream-paired` read about 51 images
  per study against 26.8 (1.9×), half of them through a ResNet-10t at roughly half the cost, so
  about 1.4–1.5× the backbone compute, and took 1.42× the time with each slice read once. A crop
  seed reads 1.35× the images at 1.18× the backbone pixels.

## Expected size

- **Ranking by expected gain** (judgement):
  - `crop-rule`: September's ACL gain with the wide view kept; no direct evidence on this recipe.
  - `crop-compartments`: where the probe leaves placement the most room.
  - `t1pd-detail-84`: mixed evidence. Narrowing 140 to 120 mm gained three times, but the next
    step on T1/PD alone (the contrast split, 112 mm) tied 120 mm, and the medial meniscus has
    fallen with every narrowing.
  - `crop-own-loss`, then `crop-attention`.
- **Learned placement over the rule (guess, moderate confidence):** under 0.001 macro. What there
  is to learn is where the anatomy is, which the rule already gives; scale diversity was priced
  at about +0.001 in the multi-scale review.
- **Noise yardstick:** paired Δ training noise about 0.0006–0.0010 on top of a bootstrap standard
  error of 0.0015 ([calibration](../../literature-mechanisms.md#3-calibration-against-repeat-pairs-keep-the-rule));
  a fresh seed moves absolute ×4 macro by up to about 0.006. One run per seed is weak evidence.
- **Untested risks:**
  - Attention sharpness was measured on the 140 mm version of the recipe at the end of training,
    not on this reference and not at epoch 3, where `crop-attention` starts using it.
  - A crop shows structures 1.43× larger than the wide view to the same backbone; the
    shared-backbone mixes covered 1.25× and 1.33×.

## Directions outside the seeds

Open to the researcher inside the contract's three limits (user, 2026-10-01); listed with what is
known about each.

- **Measured against, in mixed-field:** a second backbone, a wide view beyond 120 mm at 224 px, a
  second stream behind a zero gate, readout weights per view. Proposing one again needs a named
  reason it should differ here.
- **Left out of the seeds as low prior:** random placement, a dedicated locator head, a regressed
  differentiable crop, removing the wide tokens under a crop.
- **Follow-ups the measurements point to:** a 93 mm detail view at 0.4167 mm beside the wide view
  on sagittal fluid series, the series the detail labels use most; a 224 × 160 px window;
  train-time crops distilled into the one-pass model (TASN), which removes their inference cost.
- **Related backlog:** [two views per slice](../../../docs/backlog/untried-ideas.md#views-and-fields)
  (every slice at 120 and 140 mm) is the wide-context counterpart of these crops and shares their
  design constraint: both views feed the pool from the first step, with no gate.
- **Untried:** a ranking-trained locator (several crops per slice in training); crops that follow
  Baker's cyst and effusion (large findings that held at coarse spacing).
- **Closed by the limits or standing rules:** more slices per study, inputs above 224 px,
  ensembling and test-time augmentation.
