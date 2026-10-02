# Smart-crops campaign contract

## Objective

Give the model more resolution on the joint and let it decide where to look. The current input
reads every slice at 0.536 mm over 120 mm, which keeps the full acquired detail of 18–23% of T1/PD
coronal and sagittal series; at 0.375 mm that is 90–92%
([prior work](prior-work.md#acquired-resolution)). The five forced seeds first measure what that
resolution is worth with no crop, then add a 60 mm crop at 0.375 mm beside the wide view, placed by
rule, by the model's own attention, and per compartment, and last give the crop its own loss. Two
more seeds give the fluid-sensitive slices a second, wider view through a light backbone, beside
a main view of 120 mm and of 112 mm.
After them the researcher is free to try any smart-crop idea inside the three
[limits](#limits), until the user stops the campaign (`rotate_after: 1000`, user, 2026-10-01).

The reference and starting incumbent is `rn18-field-120-control`
(`outputs/training/rn18-field-120-control`, seed 1533697213, fold 0 of `all_5fold_scanner_v2`,
×4 0.930666, 0.947 public as S92, 1,149 s of training), the `configs/training.yaml` recipe in
force since 2026-10-01: the 120 mm field with middle-slice augmentation, plane-contrast readouts
and no anatomical bias.

This reopens learned crops, dropped on 2026-09-16
([history](../../../docs/history/crop-selection-investigation.md)) and excluded from the
[mixed-field campaign](../mixed-field/contract.md), and lifts the endgame deferral of finer input
inside the limits below (user, 2026-10-01).

## Evaluation

The [shared evaluation and adoption rule](../../protocol.md#shared-evaluation-and-adoption)
applies unchanged against the campaign incumbent.

## Limits

The only limits on what an idea may do (user, 2026-10-01):

1. **Time:** training time inside the [budget](#budget) range.
2. **Slices:** no more selected slices per study than the reference's 26
   (`data.study_triplet_total`, the images per series type and the slice selection stay as they
   are). Further views or crops of those slices are allowed.
3. **Image size:** no backbone input above 224 px on either side.

Everything else about views, crops, their placement and how the model uses them is open,
including what the seeds leave out: other window sizes, spacings and shapes, crops on
fluid-sensitive and axial series, several crops per slice, a locator head, a second backbone or a
scout network, regressed or random placement, and removing the wide tokens under a crop.

Standing project rules still apply and are listed under [Excluded](#excluded).

## View policy

A view policy states, for training, ×1 validation and ×4 prediction, which views each selected
slice is read at. Every seed and every later idea states its complete view policy and **replaces**
the incumbent's; view policies do not stack. Non-view mechanisms the incumbent has adopted are
kept.

The views of the seeds:

- **Wide view:** the reference's input, 120 mm at 224 px (0.536 mm), image-centred.
- **Detail series:** the T1/PD coronal and sagittal series (`fluid_sensitive` 0,
  `fat_suppression` 0). In the seeds, axial series and fluid-sensitive series have the wide view
  alone.
- **Detail view:** 84 mm at 224 px (0.375 mm), of a detail series, in place of its wide view.
- **Crop:** 60 mm at 160 px (0.375 mm, 5×5 tokens), of a slice of a detail series, beside its wide
  view.
- **Rule position**, for the detail view and for rule-placed crops, one per series:
  - *Coronal:* centred horizontally on the fitted knee centre.
  - *Sagittal:* centred horizontally 7 mm anterior of the fitted knee centre.
  - *Vertical, both planes:* centred 10 mm below the image centre.
  - *Fallback, per series:* a series without trusted geometry is centred horizontally on the
    image centre.
  - *Clipping, per series:* the window is shifted, never shrunk, to stay inside the wide view's
    120 mm window and the stored image.
- **Model position**, for model-placed crops, one per series: the centre of the best 3×3 block of
  the wide view's 7×7 token grid by the summed attention the idea names, over the series' slices,
  mapped to the prepared image and clipped as the rule position is. The choice carries no
  gradient. A series without valid wide tokens takes the rule position.
- **Compartment positions**, for the two crops per coronal slice of `crop-compartments`: while
  the rule places them, 22 mm medial and 22 mm lateral of the fitted knee centre, 10 mm below the
  image centre; a series without trusted geometry has one crop at the rule position.
- **Jitter:** in training every crop position moves by a uniform draw of up to 12 mm (one crop
  token) on each axis, per series and step. Validation and prediction use none.
- **Context view**, for the two-views seeds: 140 mm at 224 px (0.625 mm), image-centred, of a
  slice of a fluid-sensitive series in any plane, beside its main view and through a
  Refined-RadImageNet ResNet-10t.
- **Narrow view**, for `two-views-112-fluid-140`: 112 mm at 224 px (0.5 mm), image-centred, on
  every slice in place of the wide view.

Rules for every view policy, the seeds' and the researcher's:

- **Spatial settings come from measurement.** A window size, spacing, offset or share of slices
  is derived from measured extent, acquired resolution or the logged window statistics of earlier
  runs, and stated with its basis; it is not swept.
- **Slice selection is not a view mechanism.** Pool building, source ranking and triplet selection
  keep the incumbent's 60 mm radius, so a candidate reads the reference's slices.
- **Validation and prediction are deterministic.** The views and positions of every validation
  slice are a fixed function of the study and the model; the validation dataset receives no
  training seed.
- **Real pixels.** Every view is resampled from the native image at its own spacing, never from
  another view. A second view of a slice reuses the native rows already read for it.
- **Every stream trains from the first step.** No view or stream sits behind a zero-initialized
  gate or projection; the mixed-field two-stream run did, and its second stream never trained.
- **Fallbacks are per series** and stated in the idea.

## Seeds

Forced, in this order (simple to complex; user, 2026-10-01). It is not the order of expected gain,
which is in [prior work](prior-work.md#expected-size).

1. `t1pd-detail-84`: detail series at the detail view instead of the wide view. No crop.
2. `crop-rule`: every slice at the wide view, and detail-series slices also at a rule-placed crop.
3. `crop-attention`: as 2, the crop placed by the model's attention after three epochs at the rule
   position.
4. `crop-compartments`: as 3, with two crops per coronal detail slice, one placed by the medial
   labels' attention and one by the lateral labels'.
5. `crop-own-loss`: as 2, and the crop tokens also pooled alone under their own loss.
6. `two-views-fluid-140` (user seed, 2026-10-01): every slice at the wide view, and
   fluid-sensitive slices also at 140 mm through a ResNet-10t that feeds the joint pool from the
   first step. It retakes mixed-field's two-stream run, whose second stream sat behind a zero gate
   and never trained. No crop.
7. `two-views-112-fluid-140` (user seed, 2026-10-01): as 6, with the main view at 112 mm (0.5 mm)
   on every slice in place of 120 mm.

Seed 1 shows what resolution alone gives, so that a crop result can be told apart from it. Seed 2
is the control for learned placement: seeds 3 and 4 differ from it only in where the window sits.
Seeds 6 and 7 are the wide-context counterpart of the crops and are independent of them; 7
differs from 6 by the main view alone.

## Budget

Wall-clock range: each candidate's training time must lie between 50% and 150% of the reference's
1,149 s, that is 575–1,724 s. It is the only budget rule. Ten epochs, batch size 8 and 26 slices
per study stay; a candidate that would pass the upper bound removes duplicate reads and decodes
before anything else and is otherwise blocked, never trained for fewer epochs.

## Fixed

Split, fold, evaluation labels (`rebuild_state_v2`), class-balanced BCE and its exponent 0.5, the
half-knee, report-attribute and L2-SP terms and their weights, slice selection, series types and
images per series type, slices per study, epochs, batch size, the augmentation settings and their
middle-slice path, and the evaluation and prediction multipliers.

## Excluded

- Anything past the three [limits](#limits): training time outside the range, more than 26 slices
  per study, a backbone input above 224 px.
- Fewer epochs to buy time.
- Sweeps of a spatial setting.
- A trained model inside the loader; a locator has to be part of the model.
- Ensembling and test-time augmentation (endgame).
- Weight averaging (SWA, EMA), reflection-probability changes, the class-balance exponent,
  label-source changes, gold-study weighting, multi-seed or confirmation runs.

## Evidence requirements

- Every candidate logs per epoch, for training and validation, the share of slices at each view
  (`view_share_wide`, `view_share_detail`, `view_share_crop`), next to its own mechanism scalar.
- Every candidate with two kinds of tokens in the pool logs each label's attention mass on the
  non-wide tokens (`attention_mass_view_<label>`, and their mean `attention_mass_view_mean`) and
  the ×1 macro AUC obtained by pooling only the wide tokens and only the other tokens of the same
  validation input (`val_macro_auc_view_wide`, `val_macro_auc_view_detail`, per label as well).
  They are diagnostics and select nothing.
- **Activation signatures prove the mechanism acts, not that a value is non-zero.** A crop seed's
  signature is `attention_mass_view_mean` of at least 0.05 at epoch 0; the mixed-field two-stream
  run passed an above-zero check while its second stream never trained.
- Every candidate that places a window logs, per plane, the mean and spread of the window centre in
  millimetres from the image centre, its mean distance from the rule position, and, per label, the
  distance between the mean centres of positive and negative validation studies.
- Every result is reported against the incumbent, against the reference and against every earlier
  run it differs from by one mechanism (`crop-attention`, `crop-compartments` and `crop-own-loss`
  against `crop-rule`; `crop-rule` against `t1pd-detail-84`), with the paired 80% interval.
- One run per idea does not separate differences under about 0.002 macro AUC
  ([prior work](prior-work.md#expected-size)). Placement ideas are therefore also read on the
  logged diagnostics and on the two meniscus labels, where placement moves the most attention.
- Results are read per label against the detail group (ACL, both menisci, the three OA labels) and
  the periphery group (MCL, effusion, synovitis, Baker's cyst, contusion, fracture). The medial
  meniscus fell in every run that narrowed the field to 120 mm and is read on its own.
- The researcher starts each refill from the per-epoch curves and these view columns of every
  decided run and the reference, and names the behaviour each proposal targets.
