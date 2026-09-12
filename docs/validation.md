# Validation and experiments

The default split is **`all_5fold_v3`**, protocol version 2: five stratified folds
across **all 4,407 studies**, with roughly 20% held out for validation in each
fold. Prepared on **2026-09-12**, it stratifies gold positive/negative targets
alongside effective labels and `is_gold`, and requires both gold classes for
every target in every fold. No study is forced into every fold's validation set.
Earlier split artifacts remain available for reproducing historical runs.

Use full held-out-fold macro AUC as the development selection metric, with
gold-only and report-derived diagnostics kept separate. This mixed-label
endpoint is not a public leaderboard score or an independently annotated
4,407-study reference endpoint. Repeated tuning remains development selection.

The strategy selected on **2026-09-11** is to iterate on **fold 0** until near
the end of the competition. Use its complete 904-study validation predictions
for routine comparisons. Additional-fold training, full OOF scoring and
ensembling are deferred to that later phase; they are not prerequisites for
each experiment. Keep the revised five-fold membership fixed within comparisons;
rerun the baseline on this split before comparing new experiments.

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
Each target also contributes separate gold-positive and gold-negative presence
indicators, computed only from gold members of the group. The splitter retains
one binary `is_gold` indicator, true when any member has gold labels. These
indicators are used only for stratification; training and evaluation retain
the same twelve targets. Gold and report-derived members of one report group
always share a fold.
Sorting groups before splitting makes assignments independent of input row
order. No seed search or model score was used to select the split. Keeping
groups intact and balancing multiple targets produces approximate, rather
than exact, equal study counts.

| Fold | Training studies | Validation studies | Validation share | Gold in validation | Report-derived in validation |
|---:|---:|---:|---:|---:|---:|
| 0 | 3,503 | 904 | 20.513% | 11 | 893 |
| 1 | 3,534 | 873 | 19.809% | 10 | 863 |
| 2 | 3,537 | 870 | 19.741% | 14 | 856 |
| 3 | 3,521 | 886 | 20.104% | 12 | 874 |
| 4 | 3,533 | 874 | 19.832% | 11 | 863 |

Preparation rejects indicators with fewer supporting report groups than folds.
After assignment, it requires both classes for every target in each full and
gold validation fold before saving any artifacts. Stratification encourages
balance; these final checks enforce coverage and reject unsuitable assignments.

The saved [membership audit](../outputs/maintenance/all-study-folds-v3/membership-audit.json)
confirms all 4,407 study IDs occur exactly once, every fold has zero study or
report-group overlap between training and validation, and all twelve targets
have both classes in the full, gold and report-derived validation subsets.
Every fold has three gold Medial OA positives; fold 0 also has eight negatives.
Its gold diagnostics remain based on only eleven studies.

Report grouping prevents identical normalized reports crossing a fold boundary;
it does not establish patient or scanner independence. Revisit grouping if
image metadata or duplicate-image evidence supplies stronger identities.

### Membership and leakage controls

`data/folds/all_5fold_v3/studies.csv` contains every training study with
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
split directory, currently `data/folds/all_5fold_v3/`. `label_counts.csv` records
full-fold `positive`/`negative` and `gold_positive`/`gold_negative` counts for
every target and fold. The explicit
`pydantic-settings` CLI accepts `--config`, `--data-dir`, `--split-dir`,
`--n-splits` and `--seed`; without a seed override it uses the config seed.
The protocol records official input hashes, label source/version/hash,
label-manifest hash, target-table hash, generator source hashes, dependency
versions and fold counts. Identical regeneration is allowed; conflicting artifacts
require a new `--split-dir`. Never change folds midway through a comparison.
The current study-manifest SHA-256 is:

```text
c709cd27ff54dd10d833abfa966622f9a5ecadef5d37670da658d81c26f7b3ed
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

### Previous all-study split

The **`all_5fold_v2`** split, prepared on 2026-09-12 with protocol version 2,
added `is_gold` to the effective-label indicators but did not stratify diagnoses
within the gold subset. Its validation study counts were **874, 921, 864, 865,
883**, with gold counts **12, 12, 11, 11, 12**. Fold 0 had zero gold Medial OA
positives and twelve negatives, leaving gold macro AUC undefined throughout
`baseline-gold-balanced`. Its study-manifest SHA-256 remains:

```text
c71a71243c690e0ee076845c427f942dee0085249013ee3584b0d0bd48ee2853
```

The **`all_5fold_v1`** split, prepared on 2026-09-10 with protocol version 2,
used the same target indicators, report grouping and seed without `is_gold`.
Its validation study counts were **903, 872, 883, 869, 880**, with gold counts
**15, 10, 10, 7, 16**. Fold 3 had no gold MCL positives, leaving its gold MCL
and complete gold macro AUC undefined. Its study-manifest SHA-256 remains:

```text
59232770bad9d5d3b7f141180aea75fc830eca67b27aaa006445169af7130a62
```

The historical membership audit in
`outputs/maintenance/all-study-folds-20260910T183229Z/membership-audit.json`
confirmed that every fold's train/validation union covered all 4,407 studies,
with zero study or normalized-report overlap. The original split hash is
unchanged. The revised split retains the same frozen target table, but changes
fold membership. Scores from different splits are separate development results;
start a fresh run on v3 rather than resuming a checkpoint trained on v1 or v2.

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

### Overnight hyperparameter queue

Run the seven planned experiments sequentially with:

```bash
rtk proxy uv run --locked python -m scripts.adhoc.train_overnight
```

To keep the queue running after closing the terminal, launch from the repository
root with:

```bash
rtk proxy mkdir -p outputs
rtk proxy bash -c 'nohup uv run --locked python -u -m scripts.adhoc.train_overnight > outputs/overnight-launch.log 2>&1 < /dev/null &'
```

Use `--prepare-only` to write the seven configurations without starting
training, or `--config PATH` to select the baseline YAML. Every invocation
creates a new queue; preparation does not reserve a queue for a later command.

The script trains and plots results without invoking Codex or submitting to
Kaggle. Before training, it raises the process soft open-file limit to the
inherited hard limit and saves the before/after limits in `runtime.json`.
`--prepare-only` leaves process limits unchanged.

The first background attempt on 2026-09-12 stopped at batch 44 of epoch 0 in
all seven runs and produced no validation results or checkpoints. Three logs
reported file-descriptor exhaustion; the service had a soft limit of 1,024,
compared with 1,048,576 in the shell. This is the likely shared cause. The
corrected queue below completed all seven runs with the raised limit. The
subsequent Codex resume was interrupted. Failed outputs were deleted at the
user's request, and the resume hook was removed in favor of manual launching.

The launcher snapshots the current baseline into seven validated YAML files
under a unique `outputs/overnight-<UTC timestamp>-<suffix>/configs/` directory.
It fixes `all_5fold_v3`, fold 0, seed 42, batch size eight with no accumulation,
series dropout 0.2 and validation density 2×. Other model and data settings
come from the baseline at launch. Each run starts fresh, with one change from
the control:

| Run | Backbone LR | Head LR | Weight decay | Epochs / patience |
|---|---:|---:|---:|---:|
| `01-control` | 0.0001 | 0.0003 | 0.01 | 10 / 10 |
| `02-longer` | 0.0001 | 0.0003 | 0.01 | 20 / 20 |
| `03-lower-lr` | 0.00005 | 0.00015 | 0.01 | 10 / 10 |
| `04-higher-lr` | 0.0002 | 0.0006 | 0.01 | 10 / 10 |
| `05-faster-head` | 0.0001 | 0.001 | 0.01 | 10 / 10 |
| `06-lower-decay` | 0.0001 | 0.0003 | 0.003 | 10 / 10 |
| `07-higher-decay` | 0.0001 | 0.0003 | 0.03 | 10 / 10 |

Each queue has `runs/<name>/` for normal training artifacts, `logs/<name>.log`
for training output and `status.tsv` for completed run exit codes. The launcher
prints each experiment's start, log path and completion status. A failed
experiment is recorded and the queue continues; its final exit code is nonzero
if any training or plotting command failed. SIGINT or SIGTERM
stops the queue and terminates its active subprocess group. Successful runs
are plotted together in `validation_metrics.png` at the
end. Every invocation creates fresh directories and never resumes an earlier
queue. The completed queue below took approximately five hours.

### 2026-09-12 overnight results

**`04-higher-lr` is the preferred recipe for subsequent controlled comparisons.**
Its learning rates were adopted in `configs/baseline.yaml` on 2026-09-12.
It scored **0.918** publicly, versus **0.913** for `02-longer` and **0.903**
for `05-faster-head`. All three competition submissions completed successfully;
the Kaggle API scores and saved submission records were checked on 2026-09-12.
This is development-selection evidence from three submissions, not a private
leaderboard result or a statistically established ranking.

The queue is
[`outputs/overnight-20260911T234416Z-8du9a4nh`](../outputs/overnight-20260911T234416Z-8du9a4nh).
All seven runs used ReXNet-200 with attention MIL, adjacent triplets, 224-pixel
inputs at 0.67 mm, center-weighted sampling, series dropout 0.2, Steven v4
labels, seed 42 and the fixed `all_5fold_v3` fold 0: 3,503 training and 904
validation studies, including eleven gold and 893 report-derived validation
studies. Training batch size was eight; validation density was 2×.
The [control configuration](../outputs/overnight-20260911T234416Z-8du9a4nh/runs/01-control/effective_config.json)
and the hyperparameter table above define the comparison.

Run provenance records commit `14940b06344345e333386a3b2aad3645d0d5d5f9`
with a dirty working tree; each run's `source_snapshot.zip`,
`effective_config.json`, `provenance.json` and `fold_0/input_identity.json`
preserve the actual source/configuration and input identities. Input identities
match across all seven runs, including membership hash
`c709cd27ff54dd10d833abfa966622f9a5ecadef5d37670da658d81c26f7b3ed`.
Every saved prediction table contains the expected 904 unique fold-0 IDs,
matches its selected epoch's probabilities exactly, and reproduces the saved
full-fold AUC. This comparison used saved predictions, without fresh inference.

Metrics below are evaluated at the checkpoint selected by full-fold AUC.
Epochs are numbered from one; checkpoint filenames and the
[plot](../outputs/overnight-20260911T234416Z-8du9a4nh/validation_metrics.png)
number epochs from zero. Public scores are reported to the API's three decimals.

| Run | Selected epoch / budget | Full-fold AUC | Gold AUC | Soft-target BCE | Training minutes | Public AUC |
|---|---:|---:|---:|---:|---:|---:|
| `01-control` | 10 / 10 | 0.87848 | 0.91961 | 0.45870 | 35.8 | Not submitted |
| `02-longer` | 12 / 20 | 0.88354 | 0.94183 | 0.46144 | 79.2 | 0.913 |
| `03-lower-lr` | 10 / 10 | 0.86097 | 0.91634 | 0.47215 | 35.0 | Not submitted |
| `04-higher-lr` | 10 / 10 | 0.88294 | 0.91756 | 0.45404 | 37.4 | 0.918 |
| `05-faster-head` | 10 / 10 | 0.87724 | 0.93995 | 0.46091 | 36.0 | 0.903 |
| `06-lower-decay` | 10 / 10 | 0.87866 | 0.92814 | 0.45850 | 35.7 | Not submitted |
| `07-higher-decay` | 8 / 10 | 0.87944 | 0.92794 | 0.45947 | 34.7 | Not submitted |

All submissions used 2× images and inference batch size two. Exact checkpoint,
bundle, notebook and submission identities are linked in the
[submission artifact record](inference-efficiency.md#overnight-fold-0-submissions).

- Doubling both learning rates to backbone `2e-4` and head `6e-4` improved
  local full-fold AUC by 0.00446 over control and improved nine targets. Largest
  gains were Lateral Meniscus +0.02486, PF OA +0.01328 and Synovitis +0.01066.
  It also had the lowest BCE and the highest public score among submitted runs.
  Prefer its ten-epoch recipe with weight decay 0.01 for the next comparisons.
- `02-longer`'s local advantage over higher LR was only 0.00060 and reversed
  publicly by 0.005. It peaked at epoch 12, ending at 0.87990 local AUC with
  worsening late BCE despite falling training loss, consistent with overfitting.
  The twenty-epoch budget also stretches the cosine schedule; a fresh
  twelve-epoch schedule would be a different experiment. Longer training did
  not deliver the best public checkpoint here, but this does not establish
  that every longer schedule is worse.
- The eleven-study gold diagnostic ranked longer and faster-head above
  higher-LR, whereas higher-LR won publicly. Faster-head's gold AUC of 0.93995
  did not transfer to a strong public result. Keep gold AUC diagnostic rather
  than using it to override the full-fold endpoint. BCE is useful supporting
  evidence; three selected submissions do not justify replacing AUC with BCE.
- Lower LR underperformed at this epoch budget. Weight-decay changes produced
  small local gains without public evidence; they do not establish a reason
  to change weight decay from 0.01.

The previous best public submission, baseline4 at 2×, scored 0.913, so
higher-LR improved the reported best by 0.005. That historical comparison also
changes the split from `all_5fold_v1` to `all_5fold_v3` and series dropout from
0.1 to 0.2. The overnight control was not submitted, so the public gain over
baseline4 cannot be attributed solely to learning rate. Tiny local differences,
an eleven-study gold subset and repeated public selection should not be treated
as precise estimates of hidden-data generalization.

### 2026-09-12 140-mm crop result

**`crop140mm` scored 0.925 publicly**, improving on the 150.08-mm
`04-higher-lr` model by 0.007. Submission `56182948` completed successfully,
confirmed through the Kaggle API on 2026-09-12. Prefer its 140-mm geometry
for subsequent comparisons; the current baseline YAML already uses 0.625 mm
spacing at 224 pixels.

| Selected checkpoint | Full-fold AUC | Gold AUC | Soft-target BCE | Public AUC |
|---|---:|---:|---:|---:|
| Higher LR, 150.08 mm, epoch 10 | 0.88294 | 0.91756 | 0.45404 | 0.918 |
| Higher LR, 140 mm, epoch 8 | 0.88612 | 0.92698 | 0.45296 | 0.925 |

The [saved configuration](../outputs/crop140mm/effective_config.json) differs
from `04-higher-lr` only in spacing, apart from run name/output path:
0.625 versus 0.67 mm. Both used seed 42, `all_5fold_v3` fold 0, the same
input identities, ReXNet-200, ten training epochs, backbone/head LR
`2e-4`/`6e-4`, weight decay 0.01 and series dropout 0.2. Training source
snapshots have identical pipeline/package Python files; only an unrelated
ad hoc benchmark differs. The crop run records commit
`e313b5c92a12e20ecf74f8f3ca61e488ea46fa56` with a dirty working tree in
[`provenance.json`](../outputs/crop140mm/provenance.json); its
`source_snapshot.zip` preserves the executed source.

Local AUC increased by 0.00318, with nine of twelve targets improving.
Largest gains were Contusion +0.01088, MCL +0.00962 and Lateral Meniscus
+0.00885; Synovitis fell by 0.00835. Gold AUC and BCE also improved, although
the gold diagnostic still contains only eleven studies. These are selected
checkpoint comparisons on the same 904-study development fold; they do not
establish statistical significance or private leaderboard performance.

Training took 35.5 minutes. Both submissions used 2× images and batch size two.
The input dimensions and model are unchanged, so this improvement adds no CNN
pixel workload; official hidden-run runtime is unavailable. Reducing spacing
both tightens coverage and increases spatial sampling density, so the result
does not identify which effect caused the gain. A separate 336-pixel comparison
at fixed 140-mm coverage would use `140 / 336` mm spacing.
See the [submission artifact](inference-efficiency.md#140-mm-crop-submission).

### Plot validation metrics

Plot epoch metrics from completed run folders with:

```bash
rtk proxy uv run --locked python -m scripts.tools.plot_validation_metrics baseline0 baseline1 baseline4
```

The command writes `outputs/validation_metrics.png` by default and shows the
full-validation, gold and report-derived macro AUCs plus soft-target BCE. Use
`--output` to select another PNG path. Undefined gold or report-derived macro
AUCs (`null` in epoch JSON) appear as gaps; a subgroup with no defined epochs
has no visible curve. Full-validation AUC and BCE must remain numeric.

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
