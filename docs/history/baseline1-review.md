# Baseline1 review

Reviewed on **2026-09-10**. The saved fold-0 run reaches **0.843997 reference
macro AUC**. No confirmed implementation defect explains its score. The audit
does identify broken latest-checkpoint retention, an ignored pretraining flag,
and insufficient checks when loading a checkpoint. These issues are now
[corrected](#implemented-corrections), with versioned physical positions and
legacy prediction compatibility. This historical result did not justify
scaling the encoder before improving sampling, optimization, augmentation and
supervision.

The initial review inspected source, local dependency implementations,
checkpoint contents, label tables and saved predictions. Subsequent corrections
were exercised with bounded real-data training/resumption and legacy inference,
as recorded below. Proposed modeling gains remain hypotheses unless a
measurement is explicitly identified below.

## Run identity and measured results

Artifacts are under `outputs/training/baseline1/` (deleted).

| Item | Recorded value |
|---|---|
| Source identity | No Git commit; `source_snapshot.zip` SHA-256 `85b506403ba90fb118ec2245e5a6ee06029ac469cffd65035ed886795cd934dd` |
| Configuration | `effective_config.json`; seed 42, `resnext26ts.ra2_in1k`, pretrained, 256-wide projection, one CLS transformer, masked max across series |
| Model size | 10,093,524 parameters, including 8,248,520 in the backbone; checkpoint weight tensors occupy about 38.62 MiB |
| Images | One channel, 224 px, 0.8 mm spacing, 179.2 mm field of view; six type limits `[17, 16, 15, 15, 15, 15]` |
| Training input count | Mean 73.90, median 76, maximum 93 selected slices/study before decoding failures and series dropout, calculated from shortest-series metadata |
| Labels | Steven v4, dataset version 6; table SHA-256 `eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4` |
| Split | `gold_5fold_v1`, fold 0; membership SHA-256 `f32934f4e13d55fa53aa4245cc7ad26b6747ababe35b7c54e52ef20f07eef176` |
| Membership | 4,391 training studies: 46 gold and 4,345 silver; 12 gold validation studies |
| Runtime settings | Batch 8, accumulation 1, eight workers, BF16, encoder activation checkpointing, channels-last, series dropout 0.1 |
| Completed budget | Epochs 0–9, 5,490 optimizer steps; selected epoch 4 at step 2,745 |
| Runtime | 3,142.09 s training, about 52.37 min or 13.97 training studies/s including validation overhead; saved prediction step 2.37 s |
| Recorded peak allocation | 12,686,625,792 bytes, about 11.82 GiB; sampled after prediction as well as training, so not an isolated training peak |

At the initial review, the snapshot and numerical training implementation
agreed; the differences were header-failure logging and the pipeline catalog.
The subsequent corrections below change training/checkpoint handling and add
an explicitly selected physical-position mode. Baseline1 artifacts retain the
original configuration and source snapshot.

| Epoch, zero-based | Training BCE | Validation macro AUC |
|---:|---:|---:|
| 0 | 0.579633 | 0.722939 |
| 1 | 0.543490 | 0.796467 |
| 2 | 0.521750 | 0.808501 |
| 3 | 0.503129 | 0.767303 |
| 4 | 0.484763 | 0.843997 |
| 5 | 0.470909 | 0.802213 |
| 6 | 0.452056 | 0.841559 |
| 7 | 0.436403 | 0.829214 |
| 8 | 0.423140 | 0.834389 |
| 9 | 0.408734 | 0.841747 |

The optimizer's cosine schedule assumes 30 epochs. Its learning-rate multiplier
is still **0.95425 at the end of epoch 4** and **0.78279 at the end of epoch 9**.
Early stopping therefore ended the run before the low-learning-rate part of
training. Decreasing training loss and noisy validation do not establish either
underfitting or overfitting; the interaction motivates a short complete schedule.

| Target | Selected AUC | Validation positives / 12 |
|---|---:|---:|
| ACL | 0.916667 | 6 |
| MCL | 0.600000 | 2 |
| Medial Meniscus | 0.828571 | 7 |
| Lateral Meniscus | 0.843750 | 4 |
| Medial OA | 0.888889 | 3 |
| Lateral OA | 0.750000 | 2 |
| PF OA | 0.828125 | 4 |
| Effusion | 1.000000 | 7 |
| Synovitis | 0.657143 | 5 |
| Baker's | 0.814815 | 3 |
| Contusion | 1.000000 | 3 |
| Fracture | 1.000000 | 5 |

For MCL and Lateral OA, changing one positive-negative pair's ordering changes
that target's AUC by 0.05 and macro AUC by approximately 0.00417. Three perfect
target scores here do not establish that those targets are solved.

The saved run contains no Kaggle submission score. Its `oof_predictions.csv`
has only 12 rows; it is partial reference OOF, not the complete 58-study endpoint.
The [public evidence](../research/competition-and-prior-art/public-research.md#validation-and-ensembling) includes a
reported 0.936 public score for a small single-fold ResNet at 224 px, but this
does not imply that baseline1 is 0.092 worse on the same endpoint. The official
requirements rechecked later are recorded in the
[inference notes](../inference/submission-requirements-and-output-contract.md#verified-submission-requirements).
Historical public-score claims here remain unverified.

## Confirmed defects

These findings describe baseline1's original implementation. Line references
in this section refer to its source snapshot. The current implementation has
the [corrections below](#implemented-corrections).

### Latest checkpoint actually contains the best epoch

[`scripts/pipeline/train.py`](../../scripts/pipeline/train.py), lines 200–207,
combines `monitor="val_macro_auc"`, `save_top_k=1`, and `save_last=True`.
In installed Lightning 2.6.5, `ModelCheckpoint.on_train_epoch_end` and
`on_validation_end` refresh `last.ckpt` only when a checkpoint was saved that
step; `on_train_end` does not refresh it when an earlier checkpoint exists.

Both `best-epoch=04-val_macro_auc=0.84400.ckpt` and `last.ckpt` contain epoch 4,
step 2,745, and early-stopping wait count zero. The logs and timing record
completion through epoch 9 and step 5,490. Resuming `last.ckpt` loses the final
five epochs and their optimizer/callback state. The selected best score remains
valid, but later weights cannot be recovered from these artifacts.

### Fresh training overrides `pretrained: false`

[`scripts/pipeline/train.py`](../../scripts/pipeline/train.py), line 173, passes
`load_pretrained=resume is None`. The module forwards it to the model, where
`model.py`, lines 66–69, treats any explicit
boolean as an override of `model.pretrained`. Consequently every fresh training
run loads pretrained weights, even if YAML says false; the identity JSON can
then report a false flag for a pretrained model.

### Prediction accepts incompatible checkpoint semantics

`inference.py`, lines 23–35, reads tensors
and checks their shapes, but ignores the checkpoint's saved config, targets
and fold. A different spacing, normalization, slice policy, series-type order,
or twelve-target order can load successfully and produce semantically incorrect
predictions. An incorrect producing fold can also be recorded by the CLI.

## Implemented corrections

Implemented on **2026-09-10** in
`checkpoints.py`,
[`train.py`](../../scripts/pipeline/train.py),
`training.py`,
`inference.py` (package modules since moved into subpackages), and the shared
configuration/preparation/dataset path.

- Best and latest checkpoints use separate callbacks. The unmonitored callback
  overwrites `last.ckpt` every completed epoch, even when validation declines.
  Early-stopping state is updated before checkpoint callbacks run.
- Fresh training uses `model.pretrained`; resumed training constructs the model
  without a pretrained download before restoring the saved weights.
- Prediction requires saved configuration and ordered targets, checks all data
  settings and model semantics, and verifies the producing fold and split for
  validation, including membership/protocol hashes when recorded.
  Paths and runtime settings can differ. Backbone pretraining, channels-last
  and activation-checkpointing switches do not alter weight/input identity at
  inference. Raw state dictionaries without metadata are rejected.
- New training checkpoints embed an `input_identity`, also written to
  `fold_<k>/input_identity.json`. It fingerprints the label manifest, selected
  label source, split protocol, membership, actual training/validation rows,
  series metadata and optional image-count table. Resume requires these to
  match, along with training settings. Storage paths, run name, selected fold
  list and loader/device placement can change; the actual producing fold is
  checked separately. DICOM pixels are not rehashed by this guard.
- `data.position_encoding: physical_mm` is explicit in `configs/baseline.yaml` (removed).
  With `position_reference: center`, it encodes float32 distances from the
  physical midpoint of the full indexed acquisition, keeping the reference
  fixed across sampled subsets.
  Positions use DICOM orientation/location, with `SliceLocation` as a physical
  fallback. Index-only metadata is not silently treated as millimeters; a
  candidate without the required positions is rejected through the existing
  acquisition fallback path. Padding also uses float32 positions.
- Configurations that omit `position_encoding` retain `index` semantics for
  existing checkpoints. Baseline1 must use its saved `effective_config.json`,
  not today's physical-position baseline YAML. Legacy checkpoints without
  recorded input identities remain usable for prediction but cannot satisfy
  the stricter resume guard. Their missing provenance and unsaved later weights
  cannot be recovered by changing code.

Verification artifacts are in
[`outputs/archive/maintenance/baseline-fixes-20260910T175603Z/`](../../outputs/archive/maintenance/baseline-fixes-20260910T175603Z):

- `development-physical`: seed 42, eight training studies, all twelve fold-0
  validation studies, three epochs, batch four, two workers, 224 px and unchanged
  slice limits, BF16 and activation checkpointing, `pretrained: false`, physical
  positions. Best remained epoch 0/step 2 while `last.ckpt` reached epoch 2/step 6.
  BatchNorm counters were 2 and 6 respectively, consistent with fresh random
  initialization. Training took 11.29 seconds. This is execution evidence, not
  a competitive model or a claim of improved AUC.
- `development-resumed`: restored that epoch-0 checkpoint with the same recorded
  inputs/settings and completed epoch 2/step 6. The resulting checkpoints retain
  matching input identities. This establishes working resumption, not bitwise
  equivalence to uninterrupted stochastic training.
- `baseline1-reproduced.csv`: a fresh prediction CLI execution reproduced all
  144 original probabilities exactly over the same twelve studies using the
  legacy config. `verification.json` records these artifact comparisons and the
  unchanged fold-manifest hash.
- Ruff checks and the script catalog consistency check cover the changed files;
  these are static checks.

Baseline1's checkpoints were deleted on 2026-09-16, so this prediction can no longer be
reproduced; its saved `effective_config.json` and epoch records remain under
`outputs/training/baseline1/`.

## Correctness checks and modeling limitations

The following findings describe the original baseline1 run.

- Labels are joined by study ID; eligible training gold overrides silver.
  The training and validation IDs and normalized report groups have zero overlap.
  Labels are soft probabilities and observed entries use BCE-with-logits.
- AUC uses the complete validation epoch and an unweighted mean of twelve
  per-target ROC-AUCs. Recomputing from `validation_predictions.csv` gives
  **0.8439966380070546**. All 144 saved probabilities exactly match
  `epochs/validation_epoch_004.csv` after ID alignment. This verifies the saved
  artifacts agree; the subsequent correction verification also reproduced them
  with fresh checkpoint inference.
- Slices are ordered by DICOM position projected onto the orientation normal,
  with fallback metadata. Valid slices alone enter the CNN; the transformer
  masks padding and pooling omits absent or dropped series. Series dropout
  always retains an input. The checkpointed ByobNet stages match its normal
  forward path, and recomputation preserves BatchNorm buffers. All 31 saved
  BatchNorm counters advanced exactly 2,745 from the cached pretrained weights,
  consistent with one update per training batch, rather than two.
- Positional encoding uses **ordinal indices in physical slice order**, not
  millimeter coordinates: `prepare_series` appends `record.original_index`
  and returns int64 positions at lines 178–201 of the historical
  `preprocessing.py`. Physical
  distances are discarded after sorting. This is a documentation/design
  mismatch, not shuffled slices. Physical float positions are now implemented
  as a separately recorded mode; no full-training score comparison establishes
  their effect on AUC.
- Normalization is `(x / 255 - 0.5) / 0.5`; it differs from the pretrained
  checkpoint's RGB ImageNet transform. Fine-tuning can adapt, so this is a
  transfer-learning choice rather than a proven bug. A three-channel experiment
  can use the checkpoint's documented channel normalization directly. The
  current baseline now packs three adjacent slices into three channels with
  [ImageNet mean/std](../modeling/triplet-inputs-and-sampling.md#three-channel-inputs); baseline1's saved input
  semantics remain reproducible.
- Training has random slice subsets and series dropout, but no spatial or
  intensity augmentation. The 30-epoch schedule/5-epoch patience combination,
  single-slice inputs and shared CLS representation are plausible limitations
  requiring controlled comparisons.

## Follow-up experiments

Current proposals in the [backlog](../backlog/INDEX.md)
use the all-study validation protocol. Baseline1's 0.843997 score uses a
different endpoint and cannot serve as their paired baseline. Label-source
semantics and comparisons belong in the [public-label audit](../public-labels/INDEX.md).

As an exploratory calculation, the arithmetic mean of saved epoch 7–9
probabilities scores
**0.848264**, only **+0.004267** over the selected epoch. This is reused
12-study evidence, not a validated ensemble gain. These later weights were not
retained, so it cannot currently become a deployable three-checkpoint ensemble.

Broader deferrals and public evidence now live in the [backlog](../backlog/INDEX.md) and the
[public modeling review](../research/competition-and-prior-art/public-research.md#geometry-sampling-encoders-and-losses).
