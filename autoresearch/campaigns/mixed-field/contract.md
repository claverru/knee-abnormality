# Mixed-field campaign contract

## Objective

Find a way to feed the model two fields of view at the fixed 224 px input: a wide 160 mm view
(0.714 mm) for the periphery the current 140 mm window clips, and a narrow view for detail in the
central joint. The narrow field is 120 mm (0.536 mm), the incumbent's is 140 mm (0.625 mm). The ten
forced seeds go from configuration-only single-field runs to two backbones reading the same slice
at both fields ([seeds](#seeds)); after them the researcher extends whichever mechanisms helped.

The reference and starting incumbent is `rn18-l2sp-no-confusion`
(`outputs/training/rn18-l2sp-no-confusion`, seed 1533697213, fold 0 of `all_5fold_scanner_v2`,
×4 0.928062, 0.943 public as S86), the `configs/training.yaml` recipe since 2026-09-30.

This reopens the 2026-09-21 decision against a field-of-view change
([input resolution](../../../docs/research/geometry-and-views/input-resolution.md#field-of-view-and-joint-position);
user, 2026-10-01). The fields were chosen from measured extent, not from a score sweep
([prior work](prior-work.md#geometry)).

## Evaluation

The [shared evaluation and adoption rule](../../protocol.md#shared-evaluation-and-adoption)
applies unchanged against the campaign incumbent.

## Field policy

A field policy states, for training, ×1 validation and ×4 prediction, which field each selected
slice is decoded at. Every seed and every later idea states its complete field policy and
**replaces** the incumbent's; field policies do not stack. Non-field mechanisms the incumbent has
adopted are kept.

- **Fields:** 120, 140 and 160 mm only, each at 224 px, image-centred as the loader crops today.
  The one substitution is 130 mm for 120 mm in a design that shows each slice at a single field,
  and only when its per-field AUC shows the periphery labels losing
  ([prior work](prior-work.md#geometry)).
- **Slice selection is not a field mechanism.** Pool building, source ranking and triplet
  selection keep the incumbent's 140 mm radius, so a candidate reads the same slices as the
  reference and only their decoded field differs. The two configuration-only yardsticks (seeds 1
  and 2) are the exception: there the radius used for source ranking follows `data.spacing_mm`.
- **Validation and prediction are deterministic.** The field of every validation slice is a fixed
  function of the study, series type and depth rank; the validation dataset receives no training
  seed.
- **Real pixels.** A field is decoded from the native image at its own spacing, never produced by
  rescaling the 140 mm crop.

## Seeds

Forced, in this order (simple to complex; user, 2026-10-01):

1. Single field 160 mm (configuration only).
2. Single field 120 mm (configuration only).
3. Field jitter: a random field from 120, 140 and 160 mm per training slice, validation at 140 mm.
4. Mixed 140/160: slices alternate between the two fields by depth, with a field embedding.
5. Mixed 120/160: the same mechanism with the 120 mm narrow field.
6. Mixed 120/160 with per-field readout weights in the joint pool.
7. Extent-ruled field: 140 mm, or 160 mm for a series whose tissue overhangs the 140 mm window.
8. Field backbones, routed: mixed 120/160 with one ResNet-18 per field.
9. Two streams, paired: every slice at 160 mm through a ResNet-10t and at 120 mm through the
   ResNet-18.
10. Co-registered context: as 9, with the 160 mm features added to each 120 mm token at its
    position.

Seeds 1 and 2 are yardsticks: they show per label what width and what detail are worth alone, so a
later mixed result can be told apart from one field simply being better.

## User seeds added during the run

- **Plane-weighted fields** ([seed](plane-weighted-ideas.yaml); user, 2026-10-01, after the two
  single-field results showed 120 mm ahead of 160 mm by 0.0046 and 160 mm ahead only on
  synovitis): 160 mm on every second axial and every fourth sagittal slice, 120 mm elsewhere,
  coronal all at 120 mm. The shares are one fixed choice from that evidence, not a sweep; it runs
  after `mixed-field-readout-weights`.
- **Contrast split at 112 and 140 mm** ([seed](contrast-split-ideas.yaml); user, 2026-10-01):
  T1/PD series at 112 mm (0.5 mm), fluid-sensitive series at 140 mm, no 160 mm. It adds **112 mm**
  to the permitted fields (user decision, 2026-10-01): 0.5 mm is the most common acquired
  resolution, and half of all series are fully resolved at 112 mm against 30% at 120 mm
  ([prior work](prior-work.md#geometry)). The value comes from that measurement, not from a sweep.
- **Paired seeds changed** (user, 2026-10-01, after eight results in which 160 mm added nothing
  in any shared-backbone design): seed 9, `two-stream-paired`, reads every slice at 120 mm through
  the ResNet-18 and at **140 mm**, not 160 mm, through the ResNet-10t. Seed 10,
  `coregistered-context`, is withdrawn and conditional: its 140 mm version
  ([seed](coregistered-context-140-ideas.yaml)) is forced only if `two-stream-paired`'s ×4 macro
  delta exceeds `field-120-single`'s +0.001135.
- **160 mm yardstick retake** ([seed](yardstick-retake-ideas.yaml)): the keeper's replacement for
  seed 1 after its wording conflict ([index](INDEX.md#keeper-repairs)).

## Budget

Wall-clock range: each candidate's training time must lie between 50% and 150% of the reference's
1,222 s, that is 611–1,833 s (user, 2026-10-01). It is the only budget rule. Seeds 1–8 read each
slice once and keep 26 slices per study and ten epochs. Seeds 9 and 10 read each slice twice and
may lower `data.study_triplet_total` or the epoch count to stay inside the range.

## Fixed

Split, fold, evaluation labels (`rebuild_state_v2`), class-balanced BCE and its exponent 0.5, the
half-knee, report-attribute and L2-SP terms and their weights, image size 224, slice selection,
series types and images per series type, the augmentation settings, and the evaluation and
prediction multipliers.

## Permitted

- **Field policy:** which of the three fields each slice is decoded at, by depth rank, series type,
  a rule on measured tissue extent, or at random in training.
- **Field awareness:** a field embedding on the tokens, millimetre token positions, per-field
  readout weights or a per-label field bias inside the joint attention pool.
- **Backbones:** one ResNet-18 per field, or a Refined-RadImageNet ResNet-10t for the wide view
  beside the ResNet-18; fusion of the two views of one slice by position.
- **Budget within the range:** epochs, batch size, gradient accumulation and slices per study.
- **Configuration-only ideas are allowed**, but every researcher refill must also carry at least one
  idea that needs a source change.

## Excluded

- Fields other than 120, 140 and 160 mm (apart from the 130 mm substitution above), and sweeps of
  the field size or of the share of slices per field (spatial settings come from measured extent;
  user).
- Image sizes above 224, more slices or images per study than the reference, ensembling and
  test-time augmentation (endgame), including predicting one slice at two fields without a
  mechanism that trains on both.
- Non-uniform (foveated) resampling, rule-anchored or learned crops and scout networks that choose
  where or at which field to look: a separate investigation with the user.
- Dense per-field readouts with their own loss (the per-plane version lost 0.0052 on ResNet-18,
  [prior work](prior-work.md#precedent-table)).
- A trained model inside the loader: field rules use header geometry or a tissue threshold only.
- Weight averaging (SWA, EMA), reflection-probability changes, the class-balance exponent,
  label-source changes, gold-study weighting, multi-seed or confirmation runs.

## Evidence requirements

- Every candidate with a source change logs per epoch the share of training and validation slices
  at each field
  (`field_share_120`, `field_share_140`, `field_share_160`), which is also the basis of its
  activation signature, next to its own mechanism scalar.
- Every candidate whose validation uses two fields logs `val_macro_auc_field_narrow` and
  `val_macro_auc_field_wide`: the ×1 macro AUC obtained by pooling only that field's tokens of the
  same validation input, per label as well. They are diagnostics and select nothing.
- Results are read per label against the periphery and detail groups of
  [prior work](prior-work.md#geometry): Baker's cyst, MCL and fracture depend on the wide view;
  ACL, the menisci and the cartilage labels on the narrow one.
- The researcher starts each refill from the per-epoch curves and these per-field columns of
  every decided run and the reference, and names the behaviour each proposal targets.
