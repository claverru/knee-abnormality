# Project validation protocol

## all_5fold_scanner_v1

The default split since 2026-09-21. It is produced by `scripts.tools.validation prepare
--label-source combined_v1` with the recipe config, so the stratification seed is **2263738815**. It
follows the `all_5fold_v3` design below with three differences:

- **Effective labels:** gold on the 58 gold studies, [`combined_v1`](../public-labels/combined-labels.md)
  on the rest. These are also the frozen evaluation targets.
- **Scanner indicators:** each report group also receives one presence indicator per scanner group.
  Siemens, Philips and GE are split by `MagneticFieldStrength` into 3T (≥ 2), 1.5T (< 2) and
  unknown. Toshiba, Canon, Fujifilm and Hitachi are pooled as `minor`, one English-reporting site
  ([vendor associations](../metadata/vendor-label-associations.md)). The groups come from
  `outputs/analysis/sharing-disagreement-20260914/study_headers.parquet`, whose hash
  `protocol.json` records.
- **Fold sizes** differ from `all_5fold_v3`:

| Fold | Training | Validation | Gold in validation | Siemens 1.5T / 3T | Philips 1.5T / 3T | GE 1.5T / 3T / unknown | Minor |
|---:|---:|---:|---:|---|---|---|---:|
| 0 | 3,502 | 905 | 13 | 245 / 159 | 126 / 133 | 145 / 27 / 21 | 49 |
| 1 | 3,533 | 874 | 11 | 229 / 159 | 121 / 133 | 144 / 16 / 22 | 50 |
| 2 | 3,520 | 887 | 11 | 231 / 158 | 128 / 136 | 145 / 15 / 22 | 52 |
| 3 | 3,535 | 872 | 11 | 228 / 158 | 123 / 139 | 137 / 15 / 22 | 50 |
| 4 | 3,538 | 869 | 12 | 226 / 158 | 127 / 135 | 137 / 15 / 22 | 49 |

Stratification balances the large groups closely. The smallest group, GE 3T (88 studies), lands
unevenly: 27 in fold 0 against 15–16 elsewhere. No seed search was used.

## all_5fold_v3

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

The saved [membership audit](../../outputs/archive/maintenance/all-study-folds-v3/membership-audit.json)
confirms all 4,407 study IDs occur exactly once, every fold has zero study or
report-group overlap between training and validation, and all twelve targets
have both classes in the full, gold and report-derived validation subsets.
Every fold has three gold Medial OA positives; fold 0 also has eight negatives.
Its gold diagnostics remain based on only eleven studies.

Report grouping prevents identical normalized reports crossing a fold boundary;
it does not establish patient or scanner independence. Revisit grouping if
image metadata or duplicate-image evidence supplies stronger identities.

## Membership and leakage controls

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

## Frozen evaluation targets and metrics

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

## Reproduce and score

```bash
rtk proxy uv run --locked python -m scripts.tools.validation prepare --label-source combined_v1
rtk proxy uv run --locked python -m scripts.tools.validation score --predictions outputs/training/run/oof_predictions.csv
```

[`scripts/tools/validation.py`](../../scripts/tools/validation.py) saves `studies.csv`,
`label_counts.csv`, `targets.csv`, and `protocol.json` under the configured
split directory, currently `data/folds/all_5fold_scanner_v1/`. `label_counts.csv` records
full-fold `positive`/`negative` and `gold_positive`/`gold_negative` counts for
every target and fold. The explicit
`pydantic-settings` CLI accepts `--config`, `--data-dir`, `--split-dir`,
`--n-splits`, `--seed`, `--scanner-table` and `--label-source`; without a seed override it uses
the config seed, and without a label-source override the config's training source.
The protocol records official input hashes, label source/version/hash,
label-manifest hash, target-table hash, generator source hashes, dependency
versions, scanner-group table hash and rule, and per-fold study and scanner counts. Identical regeneration is allowed; conflicting artifacts
require a new `--split-dir`. Never change folds midway through a comparison.
The current study-manifest SHA-256 is `2ed3ad16b5d34d2547b5b0d202543d24d386c6f96275ee2f44d488dcee4fdf74`;
`all_5fold_v3`'s was `c709cd27ff54dd10d833abfa966622f9a5ecadef5d37670da658d81c26f7b3ed`.

The OOF CSV must contain exactly all 4,407 study IDs, their producing
integer `fold`, and the twelve named target probabilities. The shared scorer,
`knee_abnormality.evaluation.scoring`, serves both `validation.py score` and training runs that
cover every fold. It accepts only protocol version 2 and applies the same protocol checks
as the training label loader (input hashes, membership hash, 0.5 target threshold and
complete evaluation-target coverage). It aligns
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

## Previous all-study split

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
`outputs/archive/maintenance/all-study-folds-20260910T183229Z/membership-audit.json`
confirmed that every fold's train/validation union covered all 4,407 studies,
with zero study or normalized-report overlap. The original split hash is
unchanged. The revised split retains the same frozen target table, but changes
fold membership. Scores from different splits are separate development results;
start a fresh run on v3 rather than resuming a checkpoint trained on v1 or v2.
