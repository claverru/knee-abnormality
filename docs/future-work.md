# Future work

Ideas to explore. Follow the [work-tracking rules](../AGENTS.md#work-tracking)
when moving an idea into active work.

## Fast baseline experiments

These are the remaining low-cost experiments for the
[current 224 px baseline](../configs/baseline.yaml). All are unvalidated
hypotheses grounded in the measured [baseline review](baseline-review.md),
public evidence and broader competition research.

### Evaluation rule

Use **fold 0** of the [current validation protocol](validation.md#project-validation-protocol)
for iterations until near the end of the competition, as selected on 2026-09-11.
Keep its 3,504 training and 903 held-out studies fixed. Compare full-fold macro
AUC and per-target results against frozen evaluation labels thresholded at
`>= 0.5`, retaining the separate gold diagnostic. Preserve seed, model size,
training budget and validation density unless that setting is the experimental
factor. Training targets remain soft unless explicitly testing supervision.
The optional export parity utility is not required for each run or submission.

### Deferred report-label supervision

Gaussian sampling and shared 0.72 mm spacing are now planned in
[root next.md](../next.md) and tracked in [ongoing work](ongoing.md).
Pooling and augmentation ablations were dropped from the proposed experiments.

Keep **soft versus hard report-derived training targets** as a future experiment.
Retain the original gold labels and frozen validation targets. Investigate
uncertainty weighting separately; report-derived probabilities need not be
calibrated confidence. This would require a small loss or label-preparation
change with similar model compute and memory. No gain is established and no
implementation is scheduled. See the [public label audit](public-label-audit.md)
and [supervision evidence](public-research.md#report-derived-supervision).

## Deferred preparation experiments

- **Learned per-series intensity bounds:** revisit after the fixed-clipping
  baseline. A shared predictor could estimate bounds for each acquisition,
  including unseen series. This needs unclipped floating-point input before
  the learned operation; uint8 produced by fixed clipping has already lost
  information. Parameters indexed by training series UID would not generalize
  to unseen acquisitions.
- **Prepared pixel or feature caches:** consider pixel caching only if measured
  loading throughput limits training after worker/prefetch tuning. Record
  source, preprocessing and code identities, preserve enough slice coverage,
  and budget disk space. Feature caching is useful only while the encoder is
  frozen and also requires checkpoint and fold-lineage identities.

## Public-evidence experiment candidates

Recorded on 2026-09-09 after the
[discussion/notebook investigation](public-research.md). These remain candidates
after the two changes in [next.md](../next.md); none is a demonstrated project
improvement.

- **Label handling:** using the [audited public sources](public-label-audit.md),
  compare hard versus soft targets and explicit uncertainty weights. Isolate
  targeted Synovitis repair
  from all-target imputation. Preserve an evaluation endpoint not used in
  label generation or repair.
  [Evidence](public-research.md#report-derived-supervision).
- **Geometry and slice coverage:** vary physical crop, output resolution and
  sampled-window coverage one at a time, with fixed split and labels. Check
  whether more sampled windows add information or only runtime. Evaluate the
  current triplet placement through the planned Gaussian comparison.
  [Evidence](public-research.md#geometry-sampling-encoders-and-losses).
  [Cross-competition comparison design](research/experiment-priorities.md#pooling-and-coverage).
- **Complementary model families:** compare an understandable DINO baseline
  with CoAtNet or a medical-pretrained branch and target-specific pooling.
  Measure independent validation gains and incremental ensemble cost before
  scaling the encoder or importing a large public ensemble.
  [Source implementations](public-research.md#public-code-worth-inspecting).
- **Cheap inference diversity:** evaluate a few complementary predictions
  against top-epoch weight averaging or a small amount of TTA, recording total
  hidden-test-like runtime. Avoid extended public-LB blend-weight sweeps when
  local evaluation cannot resolve the differences.
  [Evidence](public-research.md#validation-and-ensembling).

## Additional MRI research candidates

Recorded on 2026-09-10 after the [broader research](research/README.md).
These extend the candidates above and remain unvalidated.

- **Efficient encoder adaptation:** compare a frozen head, lightweight adapters,
  and full fine-tuning with the same encoder, labels, windows, and seeds before
  increasing backbone size. Measure actual forward-pass cost as well as
  trainable parameters.
  [Paper evidence and cache limitations](research/pretraining-and-external-data.md#adapting-2d-foundation-models-to-volumes).
- **Anatomy-guided pooling and external knee pretraining:** investigate only
  after the baseline reveals a representation or localization limitation.
  Retain global image context,
  evaluate predicted crops without reference correction, and establish
  eligibility and label mappings before acquiring external data.
  [Resource comparison](research/pretraining-and-external-data.md#external-resources-and-label-compatibility)
  and [implementation references](research/code-and-notebooks.md).

## Near the end of the competition

Keep additional-fold training and ensembling deferred while improving the
single-fold recipe. Once that recipe is sufficiently settled, consider the
remaining fixed folds, complete 4,407-study OOF evaluation with the separate
58-study gold diagnostic, and any final ensemble within the inference budget.
This is not a prerequisite for the current experiments.
