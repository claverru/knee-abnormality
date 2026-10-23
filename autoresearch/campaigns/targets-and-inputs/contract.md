# Targets-and-inputs campaign contract

Written 2026-10-12 with the user. Started that evening and stopped by the user on 2026-10-13; the
seeds added while it ran are listed in the [index](INDEX.md).

## Objective

Raise the macro AUC of a Refined-RadImageNet ResNet-50 on the recipe of main by changing how the
report-derived targets enter the loss (class weights, hard or soft targets, the weight of
uncertain cells, which cells are kept) and what the intensity pipeline feeds the model (clipping
bounds, intensity and noise augmentation), plus three seeds the user's ranking kept: a mask
decoder beside the labels, a final phase on hard pairs, and the plane-contrast readouts' removal.
The development backbone is the ResNet-50 itself, because the ResNet-18's structure gains did not
reach it ([prior work](prior-work.md)).

The reference and first incumbent is fold 2 of `outputs/training/rn50-aligned-mixing-folds-0-2`
(trained 2026-10-12 from main at `0bd598f`: `configs/refined-resnet50.yaml`, seed 1110484336, ten
epochs, aligned contrast mixing, usual labels, teacher share 0). Its scores and training time are
in [prior work](prior-work.md).

Ten forced seeds in the order of the user's ranking by potential, then the researcher without end:

1. `soft-prevalence-class-weights`: the class weights use the mean soft target of a label, not
   its thresholded fraction.
2. `stable-intensity-bounds`: the clipping bounds of a series no longer depend on the slices
   drawn.
3. `uncertain-report-targets-weigh-less`: a report-derived cell weighs by how far its target is
   from 0.5.
4. `hard-report-targets`: report-derived training targets become 0 or 1.
5. `mask-decoder-beside-the-labels`: a mask decoder trained on the prepared masks, dropped at
   prediction.
6. `intensity-augmentation-off`.
7. `noise-augmentation-off`.
8. `balanced-target-columns`: per label, every positive cell and an equal expected number of
   negative cells in the loss at each step, all at weight one.
9. `hard-pair-final-phase`: in the last three epochs the worst-ordered pairs share a batch and a
   pair term is added; every study is still seen once per epoch.
10. `plane-contrast-readouts-off`.

## Reference exceptions

The reference was trained from main as one run over folds 0, 1 and 2, so three recorded config
values differ from this worktree's recipe and are declared as `reference_config_exceptions`.
**Authorized by the user on 2026-10-12.**

- `paths.label_manifest` and `paths.report_attribute_manifest`: the run records main's absolute
  paths; the two files have the same SHA-256 in both checkouts (`08339ec5…` and `cbd0baa5…`), and
  the label manifest's hash is part of the input identity the comparison checks.
- `validation.folds`: the run records `[0, 1, 2]` and the recipe `[2]`. Training runs one process
  per fold with the config's seed, so fold 2 of that run is the training a single-fold run would
  have done; no single-fold rerun has checked it.

- `preprocessing_identity.sources_sha256` (a `reference_identity_exception`, added by the keeper
  on 2026-10-12 when the calibration paused on it, authorized by the user the same day): the
  reference was trained at `0bd598f` and this worktree holds the loader changes of `c1d973b`,
  which edit hashed preprocessing sources. Their recorded parity is zero differing fields over
  every training, validation and ×4 prediction batch of a fold
  ([profile](../../../docs/inference/profiling/2026-10-12-middle-slice-preparation-and-loader-speed.md#parity)).

## Settled

- **The backbone is `resnet50.a1_in1k` with the Refined-RadImageNet weights.** No other backbone.
  A seed may add modules inside its blocks that keep every pretrained weight and start as the
  identity, as the squeeze-and-excitation, global-context and coordinate-attention seeds of
  2026-10-13 state, or change a fixed downsampling operation on the same weights, as the
  average-pool-shortcut and blur-pool seeds of that day state (user decisions).
- **The label source is `rebuild_state_v2` and the teacher share is 0.** A seed may change how
  the training targets of that table enter the loss; it does not change the table, the gold
  labels of the training studies or the evaluation targets.
- **Prediction settings are not subjects:** ×1 validation and ×4 prediction at τ 1 as the recipe
  has them; no test-time augmentation, no ensembling, no weight averaging.
- **Everything is computable for an unseen knee** from its own images through the model. Masks
  and report targets are used in training only.
- **Ten epochs.** Never fewer; the epoch count is not a subject.

## Principles

1. **One mechanism per run**, applied to the incumbent it is assigned.
2. **A number set by hand comes from a cited measurement or from the mechanism's definition**,
   never from a sweep on the validation labels. A seed fixes its numbers before the run.
3. **A seed can be proved active** by metrics it names, at most six.
4. **A superseded behaviour is removed**, not kept behind an option.
5. **A rule is shared by the twelve labels** unless the seed states a per-label rule fixed before
   the run.
6. **Bundles are allowed** when a seed names the runs it combines.

## How a result is read

One run is weak evidence. On the ResNet-18, three trainings of one recipe gave ×4 within ±0.0003
and ×1 within 0.0015, and labels with fewer than 100 positives moved by 0.003 to 0.007; four
ResNet-50 recipes on fold 1 landed within 0.0001 of each other at ×4. A per-label effect is read
on the ×1 AUC averaged over epochs 6 to 10 together with ×4. The seeds on the targets aim at
Synovitis, Effusion, PF OA and Medial OA, the labels with the most uncertain training cells
([prior work](prior-work.md#the-training-targets)); their effect is read on those labels first,
then on the macro. A change of the loss weights moves the validation loss by construction: read
AUC, never the loss.

**Class weighting has disagreed with the public board before** (exponent 0.5 lost on fold 0 and
won publicly; a ramped exponent tied locally and scored 0.941 against 0.942). An adoption of
seeds 1, 3, 4 or 8 is provisional until the user checks it publicly.

## Out of bounds

Another backbone; ensembling, weight averaging, test-time augmentation; fewer epochs; another
label table, changes to gold labels or to the evaluation targets, the teacher share; a
segmenter or a prepared target at validation or prediction; trained models in the loader; new
routing, attention or readout structure on the same features beyond what seed 5, seed 10 and
the two OA-detector seeds and the anatomical-bias seed of 2026-10-12 state (user decision); image sizes other than 224 px, other fields, slices or views beyond the removal of the 64 mm
sagittal fluid view in its seed of 2026-10-12 (user decision); reflection probabilities;
gold-study weighting; fixed measured offsets on the joint centre or pose; multi-seed,
confirmation or control runs not asked for by the user; choosing anything on the campaign fold's
validation labels; Kaggle submissions.

## Cost

**The budget is 0.75 to 1.25 of the reference's training time** (2,691 s). A run adds no time its
mechanism does not need: no new diagnostics in the training step, no host synchronisation. A
ResNet-50 run on this recipe peaks at 16.5 GB of GPU memory and leaves about 7 GB of the
machine's 39 GB of RAM free at its tightest; a seed that raises memory states its estimate, and
the developer runs nothing heavy beside a training.

## Duration and adoption

Started with `start`; no end. After the forced seeds the researcher keeps proposing inside this
contract until the user stops it. Adoption is by the runtime under the
[shared rule](../../protocol.md#adoption-policy): strictly above the incumbent at both ×1 and
×4. The keeper repairs and requeues; it does not adopt.

## Evaluation

Every run is read against the incumbent at ×4 and ×1, per label from largest gain to largest
loss, with the training and prediction seconds, the peak GPU memory and the lowest available RAM.
