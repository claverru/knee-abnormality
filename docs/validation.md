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

### 2026-09-12 accurate-centers result

Retain midpoint sampling and the submitted `crop140mm` model as the frozen
fusion control. The geometric-center candidate did not improve the selected
fold-0 checkpoint; its sampling path was removed from production while the
shared geometry index and preprocessing-identity enforcement were retained.
There was no Kaggle submission for this experiment.

| Selected epoch-8 checkpoint | Macro AUC | Soft-target BCE | Gold AUC | Ten-epoch training |
|---|---:|---:|---:|---:|
| `crop140mm`, midpoint control | 0.88612165 | 0.45296129 | 0.92698413 | 2,130.23 s |
| `accurate-centers`, geometric candidate | 0.88513698 | 0.45403857 | 0.92791005 | 1,864.39 s |

The paired predictions cover the same 904 study UIDs. All recorded input,
label, membership and split identities match apart from the newly added
preprocessing identity. Both runs used seed 42, 3,503 training studies, ten
epochs, 4,380 optimizer steps, the saved `crop140mm` configuration and fixed
2× validation density. Training budgets per contrast group were axial 18,
coronal 15 and sagittal 15 raw slices. At implementation start, the mutable
YAML had a different allocation; the saved control config remained authoritative
throughout. At handoff the YAML also has 18/15/15. No model, optimizer, crop
or image budget was changed for the comparison.

Macro AUC fell by 0.00098466 and BCE increased by 0.00107728. Six targets
improved and six declined. Largest declines were Fracture (−0.01242), Lateral
Meniscus (−0.00717) and ACL (−0.00489); largest gains were Medial OA (+0.00634),
Baker's (+0.00571) and Synovitis (+0.00565). The tiny gold increase concerns
only eleven studies. This single-seed result does not establish a significant
regression; it provides no gain supporting adoption under the plan's rule.

Peak allocated CUDA memory was identical at 20,017,470,464 bytes. Full training
was 12.5% faster, and post-training validation prediction took 36.47 versus
37.37 seconds. The control's archived source predates the separately verified
native-reader optimization, so the runtime difference is cumulative and cannot
be attributed to centers alone. See the bounded loader measurements and
128-study geometry/coverage audit in [data preparation](data-preparation.md#accurate-sampling-centers).
A training worker snapshot recorded 280–340 MB unique resident memory per
worker, without a matched control or a peak claim.

The [control manifest](../outputs/analysis/accurate-centers-20260912/control.json)
pins checkpoint, config, executed source, input identities, timing and prediction
hashes. The selected control is
`outputs/crop140mm/fold_0/best-epoch=07-val_macro_auc=0.88612.ckpt`, SHA-256
`f3fd7b72bed1315297d54ee92742965ebb9c892ae0683c717bf20ad0947d3a92`,
with frozen bundle `outputs/exports/knee-crop140mm-20260912`.
The rejected candidate is
`outputs/accurate-centers/fold_0/best-epoch=07-val_macro_auc=0.88514.ckpt`, SHA-256
`bf23a0e92e860062c0e52dbebf58428f0b56cd21052ce3e1a338de3cd1274c50`.
Its complete source/config/checkpoints and epoch predictions remain under
`outputs/accurate-centers`; a runnable local bundle is preserved at
`outputs/exports/accurate-centers-candidate-20260912`.

The candidate passed normal post-training validation loading, export, and
resume identity validation on its actual completed-run checkpoint. Its bundle
recomputed the same identity without training inputs and predicted the three
example-test studies successfully. Historical checkpoints with missing identity
were rejected by prediction/export. After selecting midpoint sampling, the
candidate's recorded identity is also rejected by current code. Use the
matching archived implementation for either historical model; do not rewrite
their metadata. Fresh midpoint runs use the distinct
`midpoint_triangular_shared_geometry_v1` identity.

After removing the rejected sampler, a two-pass, 16-study production profile
matched the archived control's study order, input counts and every prepared
image hash. Static lint, formatting and catalog checks passed; no tests were
written or run. These checks do not establish unexercised metadata cases or
future local-alignment validity.

### 2026-09-12 spatial-fusion engineering result

Historical subsequent decision: restore [27-query, width-64 eager fusion](modeling.md#historical-regional-fusion)
for training, later replaced by spatial-token MIL. The rejection below records the earlier
engineering comparison and is superseded as an implementation-selection
decision. Its measurements and untrained status remain historical facts.

**Reject this candidate and retain the midpoint `crop140mm` control.**
The implementation reached the plan's engineering stop condition: warm
end-to-end inference exceeded the 5% overhead limit at the submitted batch
size two. No full fold-0 training, accuracy comparison, geometry-free ablation
or Kaggle submission was started. This is a runtime rejection, not an accuracy
result or evidence against spatial fusion generally.

The initial implementation used one ReXNet feature pass, unchanged global
256-dimensional tokens, 27 patient-space queries, width 64, dense local
cross-attention with finite geometric bias, and the existing target-specific
MIL. Profiling led to compiled local tensor operations, fixed triplet packing,
factored geometry projection, nine queries (eight ±35-mm corners plus the
center), width 32, and removal of redundant indexing and host synchronization.
Every valid local observation remained available; no spatial shortlist was used.
The rejected implementation and audit utility were archived together and removed
from production. The pre-existing shared geometry foundation remains unchanged.

Final paired profile on the RTX 3090, BF16, float32 matmul policy `high`, seed 42,
32 fold-0 training studies, 2× images, batch two and 16 persistent workers:

| Measurement | Submitted bundle control | Final fusion candidate | Change |
|---|---:|---:|---:|
| Fixed-input mean batch inference, 16 warm repetitions | 62.53 ms | 65.30 ms | +4.44% |
| Complete 32-study warm loader pass, mean of three passes | 1.366 s | 1.468 s | **+7.47%** |
| Warm loader-pass range | 1.353–1.383 s | 1.448–1.482 s | Non-overlapping in this sample |
| Peak allocated inference memory | 1,633,286,656 bytes | 1,609,677,824 bytes | −1.45% |

The final input manifests match every selected triplet, prepared pixel hash,
position and input count in the submitted bundle. Training-mode manifests also
matched the archived training control on the earlier 16-study profile. Labels
and split provenance are saved with each measurement. The original 27-query
candidate passed the preliminary batch-eight training checks: 350.08 versus
361.73 ms per step (+3.33%) and 19.496 versus 19.618 GB peak allocation (+0.63%).
Those training measurements do not describe the final compiled candidate;
its failed inference gate stopped further training work.

The [decision artifact](../outputs/analysis/spatial-fusion-20260912/decision.json)
records the limits, ratios and exact profile directories. Each profile retains
source hashes, selected-input manifests, all timing samples and an operator
trace. The final candidate is `candidate-inference-b2-fused`; its reference is
`submitted-inference-b2-high`, executing the actual submitted bundle. Earlier
batch-eight profiles used the training snapshot and are explicitly preliminary.
The [source archive](../outputs/analysis/spatial-fusion-20260912/source_snapshot.zip)
contains the complete candidate, audit utility and executed plan; SHA-256
`8c5b2c04f841246ebc04fa6402cc70940db7df77027f4d2a9f561135b811a1d7`.
It is also extracted under that directory's `candidate-source/`. Reproduction
uses the archived `src` on `PYTHONPATH`, the saved `candidate.json`, and the
[archived profiler source](history/regional-sources/profile_spatial_fusion.py.txt).

The [image audit](data-preparation.md#spatial-fusion-mapping-audit) supports
coordinate bookkeeping on a bounded real sample, not complete registration
validation. Initial finite forward/backward and deterministic-validation checks
passed. Final trained-checkpoint/export behavior and attention attribution were
not evaluated because no training run was warranted. No tests were written or run.

### 2026-09-12 spatial-fusion validation comparison

The user-trained eager 27-query, width-64 model did not improve the selected
`crop140mm` checkpoint on fold 0. Saved predictions cover the same 904 studies;
label, split, membership, training/validation row and series-input hashes match.
Both runs use seed 42 and the same saved data, optimizer and trainer settings.
The selected epochs are 8 for `crop140mm` and 10 for `spatial-fusion`.

| Saved predictions | Macro AUC | Soft-target BCE |
|---|---:|---:|
| `crop140mm` | 0.886122 | 0.452961 |
| `spatial-fusion` | 0.884169 | 0.453061 |
| Equal probability average | 0.888049 | 0.449818 |

Fusion improves ACL, medial meniscus and synovitis individually; nine targets
decline, with the largest drops in fracture and lateral meniscus. A fixed
50/50 average improves 11 of 12 targets over `crop140mm`, with fracture the
exception. This supports complementary predictions on this fold, without
establishing that spatial geometry causes the benefit or that it generalizes.
No blend weights were tuned and no uncertainty estimate was computed.

The [comparison record](../outputs/analysis/spatial-fusion-vs-crop140mm-20260912.json)
pins prediction/label hashes, alignment, checkpoint epochs and metrics.
This analysis only read saved CSVs with Polars/NumPy/scikit-learn on CPU;
both individual metrics reproduced their saved results within `1e-6`.
No model execution, tests or ensemble submission were performed.

### 2026-09-13 max-pooling validation comparison

The completed `spatial-fusion-max` run selected epoch 10 at **0.87907846 macro
AUC**, below the average-pooling `spatial-fusion` reference by 0.00509042 and
`crop140mm` by 0.00704319. All runs use the same 3,503 training and 904 validation
studies, fold 0 of `all_5fold_v3`, Steven v4 labels and seed 42. Average and max
fusion have matching label, split, membership, training/validation row and
series-input identities. Their saved configurations differ in run name,
pooling and strict determinism; the historical model omitted the pooling field
and used timm's average-pooling default.

| Run | Selected epoch | Macro AUC | Soft-target BCE | Training seconds |
|---|---:|---:|---:|---:|---:|
| `crop140mm` | 8 | 0.88612165 | 0.45296129 | 2130.23 |
| `spatial-fusion`, average | 10 | 0.88416888 | 0.45306086 | 1855.21 |
| `spatial-fusion-max` | 10 | 0.87907846 | 0.45606172 | 1665.07 |

Max pooling improves four targets over average pooling: ACL, lateral OA,
effusion and contusion. Eight decline; the largest AUC drops are lateral
meniscus (-0.027538), MCL (-0.017717) and fracture (-0.011758).

Both fusion runs completed 4,380 optimizer steps. Max pooling's recorded
training duration is 10.25% shorter, with peak CUDA allocation of
20,031,506,944 bytes versus 20,077,255,168 bytes for average pooling. Max used
`trainer.deterministic: false`; average used `true`. These separate runs do
not isolate the accuracy or timing effect of pooling, determinism or external
runtime conditions. There is no measured matched nondeterministic average
control, repeated-seed estimate or max-pooling public score in this comparison.

This result gives no reason to promote max pooling for the fusion plan;
average pooling remains the stronger development reference. A parent/control
with matching execution settings is still needed for a controlled architecture
comparison. The active YAML remains set to max until the next pooling decision.

The [comparison record](../outputs/analysis/spatial-fusion-max-vs-average-20260913.json)
pins selected checkpoint, configuration and prediction hashes, source identities,
input-identity comparisons, timings and per-target deltas. The max checkpoint is
`outputs/spatial-fusion-max/fold_0/best-epoch=09-val_macro_auc=0.87908.ckpt`,
SHA-256 `2d4a934157ed9eb00688aa177813fe301bd104f45e090540bd82b295cd21983c`;
its run contains `source_snapshot.zip` for the executing source.
This review read saved JSON metrics and fingerprinted artifacts on CPU. It did
not recompute predictions, execute a model, run tests or launch a submission.

### 2026-09-13 per-plane fusion usage diagnosis

The trained `spatial-fusion-radius` checkpoint uses its regional branch, but
final MIL gives it only **4.94% of attention on average**. Omitting regional
tokens slightly reduces validation AUC and worsens BCE. This explains how
fusion can be active while the aggregate curves closely resemble `crop140mm`:
the global descriptors carry most of the final attention, and the regional
branch makes comparatively small adjustments.

The run selected epoch 8 with 0.88635162 AUC and 0.45363703 BCE on the same
904 fold-0 studies as `crop140mm`. The control selected epoch 8 with 0.88612165
AUC and 0.45296129 BCE. The AUC difference is only +0.00022998 and does not
establish a repeatable improvement. Both runs use seed 42, the same pretrained
backbone initialization, saved data and optimizer settings, and matching
label/split/row/series-input hashes. Strict determinism is disabled for the
per-plane run and enabled for the control.

The diagnosis evaluates the complete saved validation fold with the selected
checkpoint, 2× validation images, batch eight, BF16 and eight loader workers.
One unchanged CNN/fusion forward supplies both variants. The intervention
excludes regional tokens from final MIL and renormalizes attention over the
same global tokens. It does not zero features while leaving their attention
weights in the denominator. Original source identities are checked before
loading; the temporary pooling wrapper lives only in the diagnostic script.

| Same trained checkpoint | Macro AUC | Soft-target BCE |
|---|---:|---:|
| Intact fusion | 0.88635162 | 0.45363703 |
| Regional tokens omitted | 0.88565495 | 0.45451753 |
| Intact minus omitted | +0.00069668 | -0.00088050 |

Intact predictions reproduce the saved CSV to a maximum absolute probability
difference of `2.98e-8`, with identical AUC. Removing regional tokens changes
probabilities by 0.006576 on average, or **0.66 percentage points**; the median
change is 0.29 points and the largest is 11.69 points. These effects use the
same checkpoint and image features, without retraining.

Coverage is usually available: **872 of 904 studies retain all 27 regional
queries**, one retains 25, and 31 have no accepted center and no valid queries.
There are 873 accepted centers and no preparation conditions. Thus the
coverage gate is not broadly disabling the branch on validation. Regional
tokens would receive 36.58% of attention on average under uniform weighting of
valid tokens; learned attention assigns 4.94%, or 5.11% among eligible studies.
The median study/target regional share is 3.68%, with p90 10.33% and maximum
40.25%. The low average reflects final MIL weighting rather than a widespread
lack of eligible regional tokens.

Fusion improves AUC for eight targets and lowers it slightly for four within
this checkpoint. The largest gains are lateral OA (+0.002115), synovitis
(+0.001719) and medial OA (+0.001337). The largest reductions are effusion
(-0.000342) and ACL (-0.000309). Mean regional attention ranges from 1.53% for
medial meniscus to 10.17% for medial OA; synovitis receives 9.65%.
The [per-target plot](../outputs/analysis/fusion-usage-20260913/fusion_usage.png)
shows both attention and the omission effect.

These are diagnostics of one jointly trained checkpoint. Attention mass is
not a causal attribution score, and inference-time omission does not recreate
a model trained without fusion or isolate any training influence on the shared
backbone. No repeated-seed estimate or uncertainty interval was computed.
The evidence supports a functioning, lightly weighted regional branch with a
small observed contribution on this fold; it does not establish an advantage
over the independently trained `crop140mm` control or a public-score gain.

The [result record](../outputs/analysis/fusion-usage-20260913/results.json)
pins source/config/checkpoint and input identities, dependency/runtime versions,
saved-prediction reproduction, per-target metrics and summary distributions.
Sibling CSVs preserve intact, regional-omitted and original saved predictions;
`per_study.parquet` and `query_attention.npz` retain study/target attention,
validity and query offsets. The selected checkpoint SHA-256 is
`0481ee1228cb687d32f9a703671d9799089435d3b7220c1a65a28e14a2a43c14`;
the executing preprocessing source identity is
`b0f7b06268098dede3866aac113e15fd0544bfbee802684f78c4f70c8ba5762b`.

Reproduce using the matching source snapshot and a fresh output directory:

```bash
rtk proxy env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --locked python -m scripts.adhoc.diagnose_fusion_usage outputs/spatial-fusion-radius --output-dir outputs/analysis/fusion-usage-new --workers 8
```

This was an inference diagnosis, with no training, production model changes,
tests or submission. The 36.09-second diagnostic duration includes extra
pooling, analysis and artifact writing and is not an inference-speed comparison.

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

### 2026-09-13 spatial-token validation comparison

The completed `spatial-tokens-five-embeddings` run selected epoch 10 (checkpoint
index 9), with **0.88950565 macro AUC** and **0.45062525 soft-target BCE** on
the same 904 validation studies as `crop140mm`. Both metrics are the run's best
at the final epoch. This is a promising improvement on the development fold;
the paired uncertainty interval does not establish a repeatable advantage.

Kaggle submission **56206027** completed with **0.929 public AUC**, improving
on `crop140mm` at 0.925 by 0.004 and establishing the best recorded public result.
The public improvement agrees with the local direction; it does not isolate
individual architectural contributions. Exact artifact and API identities live
in the [submission record](inference-efficiency.md#spatial-token-submission).

| Selected checkpoint | Epoch | Macro AUC | Soft-target BCE | Training seconds |
|---|---:|---:|---:|---:|
| `crop140mm` | 8 | 0.886122 | 0.452961 | 2,130.23 |
| `spatial-fusion` | 10 | 0.884169 | 0.453061 | 1,855.21 |
| `spatial-fusion-max` | 10 | 0.879078 | 0.456062 | 1,665.07 |
| `spatial-tokens-five-embeddings` | 10 | **0.889506** | **0.450625** | 1,628.20 |

Relative to `crop140mm`, macro AUC increases by 0.00338400 and BCE decreases
by 0.00233603. Eight targets improve and four decline. Largest gains are
Baker's (+0.01357), MCL (+0.00915), Contusion (+0.00831) and ACL (+0.00760).
Declines are Effusion (−0.00502), Medial OA (−0.00275), Fracture (−0.00262)
and Synovitis (−0.00149). Gold AUC is 0.93112, but this subset has only eleven
studies and does not resolve small differences.

The saved-prediction analysis uses 2,000 paired study-level bootstrap replicates,
seed 20260913, keeping all twelve targets together. Candidate-minus-control
percentile 95% intervals are [−0.000249, +0.006887] for macro AUC and
[−0.005022, +0.000551] for BCE; all 2,000 replicates are defined. Both intervals
include zero. They condition on the selected checkpoints and this held-out
fold, excluding training-seed variability and checkpoint-selection uncertainty.

Validation AUC reaches 0.888696 at epoch 7 and 0.889506 at epoch 10, a late gain
of 0.000810. The last epochs improve slightly without an obvious late validation
collapse; this is not evidence that a longer training budget would improve it.
Peak allocated CUDA memory is 20,089,207,808 bytes, close to the control's
20,017,470,464 bytes. Full training is 23.6% faster, but strict determinism differs
(`false` here, `true` in the control) and the older control predates native-reader
changes. These full-run timings do not isolate architecture or inference cost.

Seed 42, the 3,503/904 training/validation membership, `all_5fold_v3`, Steven v4
labels, ReXNet-200 initialization, 224-pixel/140-mm inputs, triplet budgets,
2× validation density, optimizer and ten-epoch budget are unchanged. Saved
label/split/membership/series/training/validation identities match. Architecture
and execution changes prevent attributing the difference to 3D coordinates or
any one embedding. The manual encoding diagnostic below has not been run.

The [comparison artifact](../outputs/analysis/spatial-tokens-vs-crop140mm-20260913/comparison.json)
pins prediction, label and run-artifact identities, all per-target results,
bootstrap settings and analysis-source identity. The
[learning curves and target deltas](../outputs/analysis/spatial-tokens-vs-crop140mm-20260913/validation-comparison.png)
are also available as an
[SVG](../outputs/analysis/spatial-tokens-vs-crop140mm-20260913/validation-comparison.svg).
This analysis reads saved CSVs and metrics on CPU; no model execution, tests,
new training, encoding diagnostic or blend submission was performed.

### 2026-09-13 local and public agreement after the split change

Local full-fold AUC tracks the public leaderboard strongly across the six
distinct submitted checkpoints trained on `all_5fold_v3`. All use fold 0, seed 42,
the same 904 validation studies, matching label/split/membership/input identities,
and 2× image limits for both validation and submission. Earlier baseline3 and
baseline4 submissions use `all_5fold_v1` and are excluded; repeated submissions
of one checkpoint at different densities are not independent model comparisons.

| Submitted model | Local macro AUC | Public AUC | Submission |
|---|---:|---:|---:|
| Overnight faster head | 0.877244 | 0.903 | 56181919 |
| Overnight higher LR | 0.882938 | 0.918 | 56181821 |
| Overnight longer training | 0.883536 | 0.913 | 56181804 |
| Regional spatial fusion | 0.884169 | 0.920 | 56190442 |
| `crop140mm` | 0.886122 | 0.925 | 56182948 |
| Spatial tokens, five embeddings | **0.889506** | **0.929** | **56206027** |

Descriptive Pearson correlation is **0.9634** and Spearman rank correlation is
**0.9429**. Fourteen of fifteen pairwise rankings agree. The exception is the
longer versus higher-LR run: a local advantage of 0.000598 reverses into a public
deficit of 0.005. This supports continuing to use full-fold AUC as the primary
development signal, with BCE as supporting evidence and gold-only AUC diagnostic.

The v3 protocol stratifies gold-only positive/negative indicators and gold
membership alongside the effective labels, with both classes required for every
target in each gold validation fold. Gold counts are 11, 10, 14, 12 and 11 across
folds; fold 0 still contains 893 report-derived and only eleven gold studies.
The observed agreement follows the split change, but these six selected runs
cannot establish that balancing gold labels caused the correlation. Scores are
rounded to three decimals, model submissions were selected during development,
and all runs use one fold and seed. No local-to-public score conversion or
private-leaderboard guarantee is inferred.

The [comparison artifact](../outputs/analysis/local-public-all-5fold-v3-20260913/comparison.json)
records every checkpoint, bundle and submission identity, saved local metrics,
completed API scores, matching input identities and exclusions. The
[scatter plot](../outputs/analysis/local-public-all-5fold-v3-20260913/local-public.png)
is also available as an
[SVG](../outputs/analysis/local-public-all-5fold-v3-20260913/local-public.svg).
This analysis only reads saved metrics and submission records; no training,
inference or tests were run.

## Manual encoding usage diagnosis

After training, run the separate command manually against the selected checkpoint:

```bash
rtk proxy uv run --locked python -m scripts.adhoc.diagnose_encoding_usage outputs/spatial-tokens-five-embeddings --output-dir outputs/analysis/encoding-usage-01 --workers 8
```

`--checkpoint` selects an explicit checkpoint; otherwise the run's fold selection
is used. The output directory must be fresh. Configuration, source/checkpoint and
input identities are checked and recorded. No training weights are changed and
no callback or training recipe launches this command.

Every variant reuses one batch's projected backbone features and deterministic
held-out images, then recomputes the embedding sum, LayerNorm and MIL with the
same visual token membership. Variants are intact; without 1D; without 2D;
3D replaced by `unknown_3d`; without plane; without contrast; and without all
position information (omit 1D/2D and use unknown 3D). No production removal flags
are added. The complete held-out fold supplies per-study predictions, per-target
and macro AUC, soft-target BCE and absolute probability changes. Geometry counts
identify the valid-3D study subset, which also receives the 3D comparison.
Undefined target/subset AUCs are omitted as null values.

Paired study-level bootstrap resampling retains all twelve targets together.
The default is 2,000 replicates with seed 20260913 and percentile 95% intervals
on variant-minus-intact differences, recording valid replicate counts. These
intervals describe this checkpoint and held-out fold, not training-seed variability.
The intact output is compared with saved validation probabilities; investigate
any disagreement before interpreting small intervention effects.

These interventions measure reliance. Embeddings overlap and interact, effects
need not add, and they cannot be assigned unique contribution percentages.
Omitting an embedding never omitted in training can create unfamiliar inputs;
the 3D unknown intervention has explicit training support. Attention weights or
embedding magnitudes alone are not contribution scores. Separately trained
models would be needed to establish learning benefits; no such training is
part of this implementation. The implemented decisions and original method
references are preserved in the [completed plan snapshot](history/spatial-token-plan.md.txt).
