# Validation splits

Every split used in the project, with the dates it was the default. The rules that apply to all
of them are in the [validation protocol](validation-protocol.md).

A split lives in `data/folds/<split>/` (`studies.csv`, `label_counts.csv`, `targets.csv`,
`protocol.json`) and is produced by `scripts.tools.validation`. Fold membership and evaluation
targets both change between splits, so **scores on different splits are never comparable**, and a
run is never resumed across splits.

| Split | Dates | Stratification seed | Labels and evaluation targets | Role |
|---|---|---|---|---|
| `gold_5fold_v1` (protocol 1) | to 2026-09-10 | n/a | 58 gold studies only | baseline1; superseded |
| `all_5fold_v1` | 2026-09-10 | 42 | Steven v4 | superseded |
| `all_5fold_v2` | 2026-09-12 | 42 | Steven v4 | superseded |
| `all_5fold_v3` | 2026-09-12 to 2026-09-21 | 42 | Steven v4 | superseded |
| `all_5fold_scanner_v1` | 2026-09-21 to 2026-09-28 | 2263738815 | `combined_v1` | superseded |
| `all_5fold_scanner_v2` | 2026-09-28 on | 1533697213 | `rebuild_state_v2` | the recipe's split |
| `all_5fold_scanner_v3` | 2026-10-19 on | 1533697213 | `rebuild_state_v3`, folds of v2 | v3 chain |
| `gold_holdout_5fold_v1` | 2026-10-10 | 1110484336 | `rebuild_state_v2` | student comparisons |

## all_5fold_scanner_v2

The default split since 2026-09-28 (user decision, with the adoption of
[`rebuild_state_v2`](../public-labels/label-rebuild/training-and-a2.md#a2-common-state-scale-and-the-zhen-lineage-fix-2026-09-28)).
`scripts.tools.validation prepare` with the recipe config produced it, so the label source is
`rebuild_state_v2` and the stratification seed is the recipe seed **1533697213** (the seed of the
recipe at that time). Effective labels: gold on the 58 gold studies, `rebuild_state_v2` on the
rest; these are also the frozen evaluation targets (`evaluation_targets_sha256` `0a11adce324571643d25aaeddf465136cf50c0ba803c4e04b790b0df154b0ae3`, float32).
Fold 0 draws 214 / 179 / 184 / 167 / 168 studies from the v1 folds 0-4, so no score on
`all_5fold_scanner_v1` is comparable with a run on this split.

| Fold | Training | Validation | Gold in validation | Siemens 1.5T / 3T | Philips 1.5T / 3T | GE 1.5T / 3T / unknown | Minor |
|---:|---:|---:|---:|---|---|---|---:|
| 0 | 3,495 | 912 | 12 | 237 / 159 | 133 / 134 | 149 / 28 / 21 | 51 |
| 1 | 3,533 | 874 | 12 | 231 / 159 | 129 / 132 | 137 / 14 / 22 | 50 |
| 2 | 3,532 | 875 | 12 | 229 / 158 | 119 / 139 | 144 / 15 / 22 | 49 |
| 3 | 3,525 | 882 | 11 | 234 / 158 | 124 / 138 | 140 / 16 / 22 | 50 |
| 4 | 3,543 | 864 | 11 | 228 / 158 | 120 / 133 | 138 / 15 / 22 | 50 |

Stratification uses `MultilabelStratifiedKFold`
([iterative-stratification 0.1.9](https://github.com/trent-b/iterative-stratification)) over
report groups, with positive and negative indicators for each target from the effective labels,
separate gold-positive and gold-negative indicators for each target (from gold members only), one
`is_gold` indicator and one presence indicator per scanner group. Siemens, Philips and GE are split
by `MagneticFieldStrength` into 3T (>= 2), 1.5T (< 2) and unknown; Toshiba, Canon, Fujifilm and
Hitachi are pooled as `minor`, one English-reporting site
([vendor associations](../metadata/vendor-label-associations.md)). The groups come from
`outputs/analysis/s/sharing-disagreement-20260914/study_headers.parquet`, whose hash
`protocol.json` records. Studies are grouped by SHA-256 of the report text with whitespace
collapsed and case folded; sorting the groups first makes assignments independent of row order.
Gold and report-derived members of one group always share a fold. No seed search or model score
selected the split. Groups stay intact and several targets are balanced, so fold sizes are
approximate. Preparation rejects indicators with fewer supporting groups than folds and requires
both classes of every target in each full and each gold validation fold before saving anything.
Report grouping stops identical normalized reports crossing a fold; it does not establish patient
or scanner independence. The smallest scanner group, GE 3T (88 studies), lands unevenly (28 in fold
0, 14-16 elsewhere).

## Rerolled, retargeted and special splits

- **`all_5fold_scanner_v3`** (2026-10-19): `validation retarget` copies the folds of
  `all_5fold_scanner_v2` (same `studies.csv`, manifest `4b4563c3…`) and rebuilds the frozen targets
  from `rebuild_state_v3` (`rebuild_state_v3` table sha256 `d3d8cccc…`; evaluation targets
  `eca41f6b7482f3fac8bd56a6762e965bfc3364c090e24ab06521a6a095be7731`). Gold rows are unchanged; where
  the new table has a null cell the source split's frozen value is kept. `all_5fold_scanner_v3_1`
  (targets `afbc5554…`, `rebuild_state_v3_1`) was built the same way and never used for a recipe.
  v3 was chosen over v3.1 and v3.2 by its agreement with the board over a fixed 26-submission panel
  (Spearman 0.429769 against v2's 0.196407; v3.2 0.410608, interval of the difference
  includes zero) and on gold ([v3.2 labels](../public-labels/audits/2026-10-19-v32-labels.md)).
- **Per-member rerolls:** every family and generation of the rebuild chain got its own split, so
  that members leave out different studies. `all_5fold_scanner_v2_r1`-`_r5` (2026-10-18) are the
  per-family splits of the first mixed-table (generation-3) fine-tunes (`_r1` ResNet-50 seed 1533777639, `_r2`
  MambaVision 3082927044, `_r3` EfficientViT 2128871193, `_r4` FocalNet 2597761050, `_r5` ConvNeXt
  2423726674; the largest overlap of a fold with a fold of another split is 0.23-0.26 against 0.20
  by chance), `_r6` is NextViT-S (seed 1624582548), and `_r7`-`_r12` (generation 2) and
  `_r13`-`_r18` (generation 3) are the twelve splits of the next rebuild cycle (one per family and
  generation, 2026-10-18). `all_5fold_scanner_v3_r1`-`_r6` (2026-10-19; ConvNeXt, ResNet-50, NextViT
  generation 2 as `_r1`-`_r3`, generation 3 as `_r4`-`_r6`) carry the same design on
  `rebuild_state_v3`: seeds 4128928322, 399542449, 1938279604, 1167487895, 1086511012, 3081832232,
  picked from 300 candidates to minimise the worst fold-overlap chi-squared against each other and all
  19 existing splits (worst 29.6 against 45-66 for a random pick). All are prepared like v2 with
  another stratification seed; `protocol.json` has the seed and manifest hash of each. Assignments
  of splits to members: the [teacher-student chain](../modeling/external-data-and-distillation/teacher-student-chain.md).
- **`gold_holdout_5fold_v1`:** a split for comparing ways of training students from teachers, not
  for the recipe's control. Built on 2026-10-10 for the
  [student-distillation campaign](../../autoresearch/campaigns/student-distillation/contract.md#the-split)
  with `validation prepare --gold-holdout` (seed 1110484336, `rebuild_state_v2`): every report group
  that holds a gold study goes to fold 0 and the others are stratified over the five folds.

  | Fold | Studies | Gold | Use |
  |---|---:|---:|---|
  | 0 | 932 | 58 | held out: scoring only, on the whole fold and on its gold studies |
  | 1 | 880 | 0 | teacher fold |
  | 2 | 861 | 0 | teacher fold |
  | 3 | 857 | 0 | teacher fold |
  | 4 | 877 | 0 | teacher fold |

  Fold 0 holds the 62 studies of the gold report groups. Every label has both classes in every fold
  (fewest positives: MCL 42, fracture 56) and in fold 0's gold studies (fewest: MCL 9 positives).
  Without `--gold-holdout` the tool still reproduces `all_5fold_scanner_v2` byte for byte.
- **Unseen-site splits:** `all_5fold_scanner_v2_sites_out` and `_sites_out_b` (2026-09-29 and
  2026-09-30; [sites-out diagnostic](results/validation-and-ensembles/2026-09-29-sites-out-diagnostic.md))
  and `all_5fold_scanner_v3_hrout` (2026-10-21, seed 4128928322: fold 0 holds all Croatian-report
  studies by report group plus about 400 seen-site studies; other folds from `all_5fold_scanner_v3_r1`).

## Superseded splits

- **`all_5fold_scanner_v1`** (2026-09-21 to 2026-09-28, seed 2263738815, `combined_v1` labels, see
  [combined labels](../public-labels/combined-labels.md)). The same design as v2 with different
  labels. Validation sizes 905 / 874 / 887 / 872 / 869, gold 13 / 11 / 11 / 11 / 12. GE 3T landed 27
  in fold 0 against 15-16 elsewhere. Manifest sha256 `2ed3ad16b5d34d2547b5b0d202543d24d386c6f96275ee2f44d488dcee4fdf74`.
  Every fold-0 decision on it was made on the same 905 studies, and the board disagreed: S59, a
  fold-0 adoption, scored 0.939 and S58, a fold-0 reject, 0.944.
- **`all_5fold_v3`** (2026-09-12, seed 42; effective labels: official binary targets on the 58 gold
  studies and pinned Steven v4 probabilities on the other 4,349; report-derived targets thresholded
  at 0.5 only to form indicators and AUC, training kept the soft probabilities). Stratified like v2
  without the scanner indicators; validation sizes 904 / 873 / 870 / 886 / 874, gold 11 / 10 / 14 / 12 / 11; every fold has three gold Medial
  OA positives (fold 0 also has eight negatives), so its gold diagnostics rest on eleven studies. Manifest `c709cd27ff54dd10d833abfa966622f9a5ecadef5d37670da658d81c26f7b3ed`; the membership audit is
  [archived](../../outputs/archive/maintenance/all-study-folds-v3/membership-audit.json) (all 4,407 study ids
  once, no study or report-group overlap, all twelve targets with both classes in the full, gold
  and report-derived validation subsets).
- **`all_5fold_v2`** (2026-09-12): added `is_gold` to the indicators but did not stratify diagnoses
  inside the gold subset. Validation sizes 874 / 921 / 864 / 865 / 883, gold 12 / 12 / 11 / 11 / 12.
  Fold 0 had zero gold Medial OA positives and twelve negatives, leaving gold macro AUC undefined
  throughout `baseline-gold-balanced`. Manifest `c71a71243c690e0ee076845c427f942dee0085249013ee3584b0d0bd48ee2853`.
- **`all_5fold_v1`** (2026-09-10): the same indicators and seed without `is_gold`. Validation sizes
  903 / 872 / 883 / 869 / 880, gold 15 / 10 / 10 / 7 / 16; fold 3 had no gold MCL positives, leaving
  gold MCL and gold macro AUC undefined. Manifest `59232770bad9d5d3b7f141180aea75fc830eca67b27aaa006445169af7130a62`. Its historical audit is
  `outputs/archive/maintenance/all-study-folds-20260910T183229Z/membership-audit.json`.
- **`gold_5fold_v1`** (protocol version 1, baseline1): stratified only the 58 reference studies, with
  gold validation counts 12 / 12 / 11 / 12 / 11 and training counts 46 / 46 / 47 / 46 / 47; MCL
  positives 2 / 1 / 2 / 2 / 2; all targets had both classes; the 58 studies had 54 distinct target
  combinations and 58 distinct report groups. The report-only pool was shared across folds
  (`fold == -1`) except four report matches assigned to fold 0 and excluded from that model's
  training (4,345 report-labeled training studies in fold 0, 4,349 in the others). That was
  reference-only CV rather than an 80/20 split of the whole dataset, and it is why baseline1 had
  twelve validation studies. Manifest `f32934f4e13d55fa53aa4245cc7ad26b6747ababe35b7c54e52ef20f07eef176`. Version-1 support
  was removed from the label loader and scorer on 2026-09-16; scoring a historical 58-study OOF
  against it needs the archived baseline1-era sources. `outputs/training/baseline1` selected epoch 4
  at 0.8439966380070546 over its twelve fold-0 studies; its root OOF CSV holds only twelve studies,
  so it cannot satisfy either complete OOF endpoint, and its `last.ckpt` holds epoch 4 although the
  logs run to epoch 9 (current training saves a separate latest checkpoint every epoch). The
  [baseline review](../history/baseline1-review.md) has the run identity and per-target metrics.
  baseline3 later completed the 903-study all-study fold 0.
