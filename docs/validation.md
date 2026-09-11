# Validation and experiments

On **2026-09-10**, the user selected option A: five stratified folds across
**all 4,407 studies**, with roughly 20% held out for validation in each fold.
Gold status does not control assignment or force studies into validation.
The default split is **`all_5fold_v1`**, protocol version 2. The old
reference-only split remains available for reproducing baseline1.

Use full held-out-fold macro AUC as the development selection metric, with
gold-only and report-derived diagnostics kept separate. This mixed-label
endpoint is not a public leaderboard score or an independently annotated
4,407-study reference endpoint. Repeated tuning remains development selection.

The strategy selected on **2026-09-11** is to iterate on **fold 0** until near
the end of the competition. Use its complete 903-study validation predictions
for routine comparisons. Additional-fold training, full OOF scoring and
ensembling are deferred to that later phase; they are not prerequisites for
each experiment. Keep the existing five-fold membership unchanged.

## Project validation protocol

The split uses `MultilabelStratifiedKFold` from
[iterative-stratification 0.1.9](https://github.com/trent-b/iterative-stratification),
with five folds and seed **42**. Effective labels use the official binary
targets on the 58 gold studies and pinned **Steven v4** probabilities on the
other 4,349 studies. The target probability is thresholded at **`>= 0.5`** only
to form stratification indicators and calculate AUC. Training retains the
original soft probabilities for BCE.

Studies are grouped by SHA-256 of report text after collapsing whitespace and
case-folding. Each group receives positive and negative presence indicators
for each target; a group with discordant member labels can have both indicators.
The splitter sees those target indicators, not `is_gold`. Sorting groups before
splitting makes assignments independent of input row order. No seed search or
model score was used to select the split. Keeping groups intact and balancing
multiple targets produces approximate, rather than exact, equal study counts.

| Fold | Training studies | Validation studies | Validation share | Gold in validation | Report-derived in validation |
|---:|---:|---:|---:|---:|---:|
| 0 | 3,504 | 903 | 20.490% | 15 | 888 |
| 1 | 3,535 | 872 | 19.787% | 10 | 862 |
| 2 | 3,524 | 883 | 20.036% | 10 | 873 |
| 3 | 3,538 | 869 | 19.719% | 7 | 862 |
| 4 | 3,527 | 880 | 19.968% | 16 | 864 |

All twelve targets have both classes in each full validation fold. Gold status
is not separately balanced: fold 3's seven gold studies have no MCL positives,
so its gold MCL AUC and complete twelve-target gold macro AUC are undefined.
The full-fold metric remains defined. Preserve this uncertainty rather than
searching for a seed with a more convenient gold distribution.

Report grouping prevents identical normalized reports crossing a fold boundary;
it does not establish patient or scanner independence. Revisit grouping if
image metadata or duplicate-image evidence supplies stronger identities.

### Membership and leakage controls

`data/folds/all_5fold_v1/studies.csv` contains every training study with
`StudyInstanceUID`, `report_group`, `is_gold`, and `fold`:

- For model `k`, training membership is `fold != k`; validation membership is
  `fold == k`. Join by study ID so every series/window follows its study.
  Every study belongs to exactly one fold; there is no shared training-only
  pool and no `fold == -1` in the new protocol.
- A held-out report group is entirely validation data, including any gold
  members. No gold study is forced into every model's validation set.
- Preserve held-out reference labels. Any label extractor, prompt, threshold,
  window selector, or preprocessing parameter fitted using reference targets
  must use only the training portion for that model. Record external label
  provenance; prior tuning against all 58 labels compromises independent CV.
- Fix the epoch budget and checkpoint rule for fold-0 comparisons and later
  multi-fold evaluation. Repeated checkpoint/hyperparameter selection on these
  folds is development tuning, not a fresh assessment. There is no additional
  permanent reference holdout in this protocol.

Completing all five folds produces OOF predictions for all 4,407 studies,
including all 58 gold studies. Predictions from baseline1's old shared training
pool cannot be relabeled as new-protocol OOF predictions. Retrain a recipe on
the new membership before comparing it with the new endpoint.

### Frozen evaluation targets and metrics

`targets.csv` freezes the effective gold/v4 probability table used to generate
the split. Training can select a different pinned label source for a controlled
label experiment, but validation continues to use these frozen targets. This
keeps the endpoint unchanged when testing label weights or label sources.

The Lightning module gathers every held-out study before computing metrics:

- `val_macro_auc`: mean of twelve ROC-AUCs against `targets.csv >= 0.5` over
  the complete validation fold. Predictions themselves remain continuous.
  An exact target probability of 0.5 maps to positive by this fixed rule.
- `val_soft_target_bce`: BCE against the original frozen probabilities.
- Epoch JSON files contain `gold` and `report_derived` subgroup metrics,
  study counts and per-target AUCs. An absent class produces `null` for that
  diagnostic target, lists it in `undefined_targets`, and leaves the subgroup
  macro AUC `null` rather than silently averaging fewer targets. A complete
  gold macro is also logged as `val_gold_macro_auc` when defined.

The full-fold metric intentionally combines gold and report-derived labels
for option A. Most rows are inferred labels; improvements may reflect better
agreement with that label source. Gold diagnostics retain the distinct
reference-label view. Do not average the two subgroup macro AUCs and call that
the full-fold metric.

### Reproduce and score

```bash
rtk proxy uv run --locked python -m scripts.tools.validation prepare
rtk proxy uv run --locked python -m scripts.tools.validation score --predictions outputs/oof.csv
```

[`scripts/tools/validation.py`](../scripts/tools/validation.py) saves `studies.csv`,
`label_counts.csv`, `targets.csv`, and `protocol.json` under the configured
split directory, currently `data/folds/all_5fold_v1/`. The explicit
`pydantic-settings` CLI accepts `--config`, `--data-dir`, `--split-dir`,
`--n-splits` and `--seed`; without a seed override it uses the config seed.
The protocol records official input hashes, label source/version/hash,
label-manifest hash, target-table hash, generator source hashes, dependency
versions and fold counts. Identical regeneration is allowed; conflicting artifacts
require a new `--split-dir`. Never change folds midway through a comparison.
The current study-manifest SHA-256 is:

```text
59232770bad9d5d3b7f141180aea75fc830eca67b27aaa006445169af7130a62
```

The OOF CSV must contain exactly all 4,407 study IDs, their producing
integer `fold`, and the twelve named target probabilities. The scorer aligns
by ID, verifies membership and source hashes, rejects invalid probabilities,
and reports pooled OOF macro AUC, per-target AUC, per-fold scores, their mean,
soft-target BCE and gold/report-derived diagnostics. A single-class full-fold
target fails explicitly; diagnostic subgroups use the null semantics above.
Fold IDs alone cannot prove a model excluded its validation data; retain
training membership, label versions, and checkpoint provenance with each run.

Near the competition end, when the other folds are trained, use pooled OOF
macro AUC across all 4,407 studies alongside every fold and target. Until then,
use full fold-0 macro AUC and per-target diagnostics. The OOF scorer requires
all folds; training already writes complete fold-0 metrics.
Pooled AUC compares scores from
different models; it need not equal the mean fold AUC. Retain continuous
probabilities with consistent output semantics. Avoid treating tiny changes
as reliable gains or a fold-score spread as a confidence interval. Preserve
the pooled 58-study gold diagnostic alongside the primary mixed-label score.

The membership audit in
`outputs/maintenance/all-study-folds-20260910T183229Z/membership-audit.json`
confirms that every fold's train/validation union covers all 4,407 studies,
with zero study or normalized-report overlap. The original split hash is
unchanged.

## Why baseline1 had twelve validation studies

The historical **`gold_5fold_v1`**, protocol version 1, stratified only the 58
reference studies, producing gold validation counts **12, 12, 11, 12, 11**
and gold training counts **46, 46, 47, 46, 47**. Their MCL positive counts were
**2, 1, 2, 2, 2**. All targets had both classes in those reference folds; the
58 studies had 54 distinct target combinations and 58 distinct report groups.

Its report-only pool was shared across folds (`fold == -1`), except four report
matches assigned to fold 0 and excluded from that model's training. Fold 0 used
4,345 report-labeled training studies; the other folds used 4,349. That design
was reference-only CV rather than an 80/20 split of the entire dataset, and is
superseded by option A. Its manifest hash remains:

```text
f32934f4e13d55fa53aa4245cc7ad26b6747ababe35b7c54e52ef20f07eef176
```

The loader and scorer retain version-1 support. To score a complete historical
58-study OOF table, pass `--split-dir data/folds/gold_5fold_v1`; its membership
and artifacts are not regenerated or changed. Its 12-row baseline1 output is
only partial OOF and cannot satisfy either complete OOF endpoint.

The Lightning pipeline gathers study-level predictions over the full validation
epoch before scoring. Averaging batch AUCs is not equivalent to the official
metric.

The completed `outputs/baseline1` run on 2026-09-10 selected epoch 4 at
**0.8439966380070546** over the twelve fold-0 reference studies. Its saved
checkpoint predictions exactly match the selected epoch's predictions. The
[baseline review](baseline-review.md) records full run identity, per-target
metrics and limitations. Its root OOF CSV contains only twelve studies;
baseline3 has since completed the 903-study all-study fold 0, while the complete
five-fold OOF baseline remains pending. Its `last.ckpt` also contains epoch 4,
although logs extend through epoch 9, so it is not the final training state.
This historical artifact remains unchanged; current training saves a separate
latest checkpoint every epoch. The [correction verification](baseline-review.md#implemented-corrections)
also reproduced baseline1's saved predictions with fresh inference.

## Official metric and public evidence

The [official evaluation page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)
specifies the unweighted mean of twelve per-target ROC-AUC values. Submit one
row per `StudyInstanceUID` with confidence scores for the twelve target columns
in `sample_submission.csv`; retain continuous scores rather than thresholding
for F1.

Public experiments demonstrate why the evaluation label source matters:
hard-label training can win against a hard-label surrogate while soft-label
training has a better point estimate on independent radiologist labels. See
[public supervision evidence](public-research.md#report-derived-supervision)
and [validation evidence](public-research.md#validation-and-ensembling) for the
paired results, uncertainty, duplicate-report grouping, scanner stress tests,
and public-LB tuning failures. These observations informed the
[project protocol](#project-validation-protocol); their reported scores have
not been reproduced locally.

## Local-to-public score relationship

| Stage | Reported observation |
|---|---|
| 2026-08-28 | Label refinement using out-of-fold image predictions improved agreement with gold labels |
| 2026-09-02 | Attempts to obtain more strong labels repeatedly ran into circular reasoning |
| 2026-09-03 | Local improvements appeared to transfer to public scores, even with roughly 11 gold studies per validation fold |
| 2026-09-09 | Repeated fold-0 training and label iterations produced local AUC 0.977, which was later considered no longer meaningful |
| 2026-09-09 | Five-fold gold-label performance was still described as correlated with Kaggle results |

Holding out all 58 gold studies and training on the rest was proposed as an
alternative; the available evidence does not establish that this split was
implemented or tested. The small gold set and prior use of its labels also
remain relevant.

Implication for this project: preserve independent evaluation when refining
labels or selecting windows, and treat repeatedly tuned validation/public
scores as selection evidence rather than a fresh estimate of generalization.
The available notes do not provide split manifests, patient grouping rules, or
enough detail to diagnose a specific leakage mechanism.

## Target-level diagnostics

- Earlier notes used macro ROC-AUC, now confirmed by the
  [official metric](#official-metric-and-public-evidence). Single-class-column
  behavior was unanswered in the retrieved
  [scoring edge-case thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740094).
  Explicitly report undefined local target AUCs rather than silently claiming
  an implementation matches an unverified scorer edge case.
- One unnamed target had F1 around 0.2 at threshold 0.5 despite high aggregate
  local AUC. Under focal loss its F1 reportedly rose to about 0.7.
- That target-level F1 change is not a reported improvement from 0.2 to 0.7 in
  competition AUC. Inspecting F1 at other thresholds was suggested.
- Errors were said to concentrate in four targets. Reviewing the worst
  40 cases led to a suspicion that about half the errors were label mismatches;
  this was model-assisted inspection, not independently adjudicated ground truth.
- Greedily removing errors to estimate their contribution was mentioned, but
  the resulting values and target names are unavailable.

## Ensemble interpretation

Earlier notes identified prediction-scale differences between losses as a
problem for combining logits or probabilities. Rank ensembling was mentioned
as an approach already tried, but its implementation and result are absent.
Hard voting was also considered without a resolved method; that speculation
should not be used as its definition.

Track per-target behavior and complementary errors alongside aggregate scores.
The [supervision notes](supervision.md#downstream-benefit-changed-over-time)
also record an observation that improved labels could make
models more correlated and reduce their ensemble contribution.

## Experiment workflow

A dashboard workflow linked ideas, experiments, local metrics, and Kaggle
submissions. Models described as `tabfpn3` and simpler
alternatives were fit to local/public results to predict promising combinations.
No training data, model definition, or predictive-accuracy result for this
meta-modeling workflow was recorded.

Interactive notebooks were useful for inspecting image/label pairs and
preprocessing. A reported failure mode of extended automated experimentation
was producing more benchmark runs without explaining why models with similar
metrics behaved differently. The actionable lesson is to compare their
predictions and failure cases, not only collect aggregate scores.
