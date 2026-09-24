# Near-misses campaign contract

## Objective

Run a fixed, user-ordered list of 15 ideas against the current recipe (`knee-geometry-merged-2`,
seed 638120755). The list comes from a review of past experiments on 2026-09-24:

- runs that beat their reference but missed the adoption rule;
- ideas that never ran or never ran properly;
- new OA and patellofemoral ideas.

The seeds run first, in order (`seed_order: forced`). Research starts only after all of them are
completed or blocked.

## Evaluation

The [shared evaluation and adoption rule](../../protocol.md#shared-evaluation-and-adoption)
applies unchanged. After an adoption, the supervisor retrains the adopted source at a fresh seed
and uses that run as the reference
([seed policy](../../protocol.md#training-seed-policy)). Soft AUC is recorded as a diagnostic.
Judge loss-shaped ideas (#5, #12, #14) on soft AUC as well as hard AUC when interpreting them;
adoption still follows the shared rule.

## Fixed recipe

- The recipe is `configs/training.yaml` as sealed at start.
- Fixed: labels, split, folds, epochs, optimizer schedule, class-balance exponent, image budget,
  sampler, crop and backbone.
- A candidate changes only its own mechanism.
- New settings go under that idea's config prefix (see `campaign.yaml`), switched on in
  `configs/training.yaml`.
- `trainer.half_knee_loss_weight` stays 0.075.

## Authorized scope

- **Family-scoped rules** (user decision, 2026-09-24) are allowed for the meniscus pair and the
  OA family, including PF OA: one rule per family, as each idea specifies. Every other label
  keeps its path.
- **Geometry:** knee side, knee centre, midline and patient coordinates come from DICOM headers
  and existing rule-based within-study checks only. Never from labels, predictions or a trained
  model.
- **No study-level fallback requirement.** The current model applies its side mask and
  half-knee loss per token, with unknown side or invalid geometry handled as it is today. Keep
  that behavior. Do not add the knee-geometry campaign's study-level incomplete-geometry
  fallback; it confounded earlier runs.
- **Follow-ups** (#3, #4, #10, #13) build on the code of the runs they follow. They run even if
  those runs were rejected: implement the parents' mechanisms together on the current incumbent,
  as the idea states.
- **Reflection consistency:** reflection augmentation and reflection TTA must keep every new
  side- or position-dependent quantity consistent. Side-normalized quantities stay unchanged
  under reflection.

## Evidence requirements

- **Activity logging:** log every mechanism's activity per epoch: studies or tokens affected,
  learned scales, auxiliary loss values, and dropped or masked counts.
- **Idle mechanisms:** a candidate whose mechanism never activates is invalid evidence, not a
  result. Report it as blocked instead.
- **Batch keys:** every new batch key the model reads must be added to `MODEL_BATCH_KEYS` in
  `inference/engine.py`.

## Exclusions

- Per-label tuning outside the two authorized families.
- Label rewriting and gold-study weighting.
- Sampler or crop changes.
- Bigger images, ensembling and TTA changes.
- Trained models in preprocessing.
- Score sweeps of any spatial setting.
- Report-derived supervision: the JevK5 attribute line runs separately.
