# Auxiliary signals campaign contract

## Objective

Find the training-time auxiliary heads and inference-available inputs that add healthy signal
to the live incumbent, using only data that already exists: the JevK5 v3 report-attribute
answers and the DICOM headers. No new extraction. The campaign starts from
`half-stack-control` (seed 2617320043) on fold 3. The six seeds come from the user's review of
2026-09-26 ([prior work](prior-work.md)) and run first, in order (`seed_order: forced`). Research
starts only after all of them are completed or blocked.

## Evaluation

The [shared evaluation and adoption rule](../../protocol.md#shared-evaluation-and-adoption)
applies unchanged. Per-label AUC, per-question attribute divergence and scanner-group subgroups
are diagnostics only; nothing is adopted per label.

A gain that could come from a site prior (scanner or protocol information) is judged with the
public leaderboard before the user adopts it: hidden-test labels are image-derived, local labels
mostly report-derived ([prior work](prior-work.md#site-priors)). For the same reason, a mechanism
that removes site information (seed 6) is expected to score low locally; the user may send it to
the public board whatever its local verdict.

## Fixed recipe

- The recipe is `configs/training.yaml` as sealed at start, including the half-stack
  augmentations and the target set 3.1 manifest.
- Fixed: disease labels, gold overrides, the disease loss and class balance, the half-knee loss,
  split, fold, epochs, optimizer schedule, image budget, sampler, crop, image size, backbone and
  augmentation.
- The target manifest `paths.report_attribute_manifest` stays unchanged. Target variants are
  built at load time from the 3.1 table, or from the raw answers the manifest records
  (`attributes.path` with its SHA-256), under the candidate's config prefix; the candidate
  records its transform's settings in the checkpoint hyperparameters.

## Authorized scope

Every rule below applies per study unless stated otherwise.

- **Attribute targets:** fine-grained operations on the existing v3 answers: merging options
  (including coarsening grades), removing options with renormalization, un-merging 3.1 merges
  from the raw answers, combining questions (joint or collapsed), deleting questions, restoring
  dropped questions, changing gates, soft or hard targets and reliability weighting. Each run
  applies one operation family through a rule fixed before the run and stated over option keys,
  question groups or measured answer statistics, never chosen from label results.
- **Attribute loss:** its coefficient and how questions are averaged.
- **Head design:** pooling, depth, what the heads read, and heads kept at inference (for example
  feeding predicted attributes into the disease readout).
- **Metadata heads:** training-only targets from DICOM headers or the scanner-group table
  (`outputs/analysis/sharing-disagreement-20260914/study_headers.parquet`, SHA-256 `8714fa8b…`,
  groups defined in `data/folds/all_5fold_scanner_v1/protocol.json`), including adversarial
  heads.
- **Acquisition inputs:** header fields read from the raw DICOMs (field strength, vendor, TR/TE,
  pixel spacing, slice thickness) as model inputs, with an explicit missing-value path per study
  (238 training studies have stripped headers). Inference-loader support follows adoption as
  separate work.

## Evidence requirements

- **Activity logging:** log every new term per epoch (loss, studies with targets, accuracy or
  divergence per head) and every input's coverage (studies with each field present).
- **Idle mechanisms:** a head whose loss never moves or an input that is always missing is
  invalid evidence. Report it as blocked instead.
- **Disabled parity:** with the new option off, the run reproduces the incumbent. New modules
  initialize under a forked RNG, like `ReportAttributeHeads`.

## Exclusions

- New JevK5 or other model extraction; any report text at inference.
- Changes to the twelve disease targets, gold overrides or the disease loss (label repair is a
  separate line).
- Per-label or per-question tuning chosen from label results: rules that pick questions or
  weights must be label-agnostic and fixed before the run.
- Site identity as an input (protocol signature, model name): measured as a post-hoc stacker at
  +0.0009 local with a negative gold sign; it waits for the endgame.
- Sweeps of the attribute coefficient beyond seed 1.
- Gold-study weighting and trained models in preprocessing.
- Bigger images, ensembling and test-time augmentation.
