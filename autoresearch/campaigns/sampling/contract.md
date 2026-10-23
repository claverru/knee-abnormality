# Sampling campaign

Defined 2026-09-21 and run overnight into 2026-09-22 against `k_05`. **Retired by the user on
2026-09-22**, a historical record like its predecessor. The supervisor stopped at revision 145
after an orchestrator job failed on a transient provider error (API 529). Six experiments ran and
all were rejected; four queued ideas were never trained
([findings](findings.md), [ideas](ideas.yaml), [backlog](backlog.yaml)). Its successor for the
oblique-series topic is the [oblique campaign](../oblique/INDEX.md). It succeeded the stopped
[triplet-sampling campaign](../triplet-sampling/INDEX.md) with a wider scope, the
combined labels and the scanner-stratified split.

## Objective

Improve what the model sees during training at fixed compute: which slices are
drawn and at which depths, how each triplet's channels are composed, which series
supply a slot, how studies are ordered into batches, where the in-plane window
sits, and how training-time model feedback steers selection. Keep coverage and
exploitation explicit. Creative and combined mechanisms are welcome; scalar
sweeps are not.

Out of scope: backbone, losses and per-cell loss weights, optimizer and schedule,
the 224-pixel input, its 0.625 mm spacing and 140 mm field of view, augmentation, series dropout, study visit frequency,
labels, fold membership and the evaluation itself.

**No submissions or publication. Only local training and local validation are
authorized.** Never submit to Kaggle, upload models or predictions, publish
datasets or notebooks, or launch remote evaluation jobs. Reading public research
and documentation is allowed.

## Evaluation and adoption

The metric, split, labels, validation multiplier, bootstrap and adoption rule are
the [shared evaluation](../../protocol.md#shared-evaluation-and-adoption), identical
for every campaign. This contract does not restate or override them. Every
candidate uses the incumbent's recorded training seed
([seed policy](../../protocol.md#training-seed-policy)).

A matched seed does not keep a sampling candidate paired with its incumbent: any
change to the drawn inputs diverges training from the first batch, and a
same-seed sampling perturbation moved fold-0 macro AUC by up to 0.0031
([evidence](../../../docs/validation/results/sampling/2026-09-17-ranked-filtered-result.md)).
Interpret single-run gains with that noise in mind; the adoption rule is unchanged.

## Reference and fixed constraints

- Reference: `outputs/training/k_05`, the user-chosen champion recipe with
  `trainer.class_balance_exponent: 0.5`, fold 0, seed 2263738815, selected epoch 7,
  macro AUC 0.899042. The recipe is `configs/training.yaml` in the current source,
  identical to the reference apart from run name and output path.
- Ten epochs, 4,380 optimizer steps and a 1,847 s runtime ceiling (115% of the
  reference's recorded 1,605.7 s, covering preparation, training with validation,
  and comparison). An adopted candidate never raises them.
- Training visits per epoch and total training triplets over the run equal the
  baseline CPU replay's counts exactly. Count real selections, excluding padding.
  Redistribution between studies, types and epochs is allowed.
- Every training study is visited exactly once per epoch; only the order and batch
  composition of those visits may change.
- Training sampling stays stochastic. Validation is deterministic, unaugmented,
  independent of the training seed and keeps its x1 triplet count; record any
  changed deterministic selection identity.
- Learned feedback uses training batches only; no validation labels or predictions.
  Its computation runs inside the measured training process.
- Crop placement comes from measured anatomy (landmarks, tissue extent), not from
  score sweeps. Any landmark estimate runs inside the measured
  process, uses images only (no labels), falls back to the image centre when it
  fails, and must also run at inference.
- Preserve labels, fold membership, grouping, scoring and checkpoint selection.

## Sealing

At start the supervisor seals the current source, resolved recipe, data identities,
reference predictions, the baseline replay's realized counts at zero tolerance, the
runtime ceiling and the evaluation settings. It verifies the reference against the
sealed recipe and input identity before any candidate. Candidate agents cannot
change these limits. See [runtime operation](../../runtime.md).
