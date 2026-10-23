# Student-distillation campaign contract

## Objective

**Find the best recipe for distilling a student from teachers.** A student is the model in force
trained again on labels that carry the predictions of teachers of the same recipe. The first
round ([prior work](prior-work.md#the-first-round)) used the simplest recipe, half silver labels
and half raw teacher predictions, and five Refined ResNet-50 students scored 0.957 public against
0.954 for one teacher. This campaign compares recipes on a split where the comparison is fair.

Its product is the recipe. The models it trains are thrown away: once the recipe is found it is
applied outside the campaign to the Refined ResNet-50 on the five folds of
`all_5fold_scanner_v2`.

## What a recipe decides

Four choices; every seed answers one of them.

1. **What the teacher gives.** Its joint prediction per label, or also its single-plane readings:
   the model can be pooled one series type at a time, so a teacher can give one opinion per series
   type and label instead of one per label.
2. **How the teacher's values are prepared.** Each study gets its value from the one fold teacher
   that held it out, and the teachers do not share a level. Raw, reranked (each teacher's values
   replaced by their rank within its fold, per label, and mapped back to the pooled distribution),
   or recalibrated per teacher and label.
3. **How teacher and silver are mixed.** The share is not given: from silver only to teacher
   only, as one number, one per label, or one per study and label that follows how much the
   teacher's planes agree.
4. **What the student is trained on.** The joint target only, or also a target per plane.

## The split

`gold_holdout_5fold_v1` (`data/folds/gold_holdout_5fold_v1`, built 2026-10-10 with
`validation prepare --gold-holdout`, seed 1110484336, labels `rebuild_state_v2`), with the report
grouping of `all_5fold_scanner_v2` and its stratification over the silver report groups:

| Fold | Contents | Use |
|---|---|---|
| 0, held out | 932 studies: all 58 gold studies, the 4 silver studies that share a report with one of them, and 870 other silver studies | scoring only |
| 1 to 4 | 3,475 silver studies, in folds of 880, 861, 857 and 877 | teachers out of fold, then students |

No teacher and no student trains on fold 0, and gold labels are used for scoring only. Fold 0
is a little larger than a fold of `all_5fold_scanner_v2` (874 to 912 studies), because the gold
report groups are added to one stratified fifth of the silver ones, so the noise measured on
past campaigns applies to it almost unscaled; each teacher trains on three of the four teacher
folds, about 59% of the data. Every scanner group and both classes of every label are in every
fold (fewest positives: MCL 42, fracture 56). Scores on this split are not
comparable with scores on `all_5fold_scanner_v2`.

## Teachers and their predictions

Four teachers of the recipe in force, ResNet-18, one per teacher fold, each trained on the other
three. They are trained once, before the campaign starts, and no seed retrains them.
Each is the recipe with `validation.folds: [k]` and `validation.unused_folds: [0]`, so fold 0
stays out of its training.

One prediction table, written once, holds for every study of folds 1 to 4 and from the teacher
that held it out: the joint logits and the logits of each series type pooled alone, per label,
at ×6 with reflection TTA and τ 0.85 (user decision, 2026-10-10), with the fold of the teacher and which series types the
study has. Every seed builds its training labels from this table and from `rebuild_state_v2`.
Nothing of it is read at validation or prediction.

The table is `paths.teacher_table` of the recipe
(`data/labels/prepared/student-distillation-teachers/teacher_table.parquet`): one row per study
and reading, with the columns `StudyInstanceUID`, `fold`, `reading` (`joint` or a series type)
and one logit per label, written by `scripts/tools/predict_teacher_table.py`. The series-type
readings are the six views of the recipe (`axial_t1pd_120mm`, `axial_fluid_120mm`,
`coronal_t1pd_84mm`, `coronal_fluid_93mm`, `sagittal_t1pd_93mm`, `sagittal_fluid_84mm`); a
study has a row for a view only when it has that series type. The recipe's `distillation.teacher_share` (0 in the reference) mixes
it into the training labels in `data/labels.py`: label = (1 − share) × silver + share ×
sigmoid(joint logit), for the studies a run trains on; the scoring targets are untouched.

## Reference

**Main's recipe, already trained:** the run `sagittal-fluid-canvas-fine` of the view-poses
campaign (×4 0.928121 on fold 1 of `all_5fold_scanner_v2`), linked as
`outputs/training/student-distillation-reference`. It gives the runtime its recipe, seed and time
budget and is never scored against: it was validated on another fold, so no run of this campaign
can be compared with it (user decision, 2026-10-10).

**The first seed is adopted by force and without a comparison.** `teacher-share-050`, the
first-round recipe (half silver labels, half raw teacher predictions), becomes the incumbent as
soon as its run is valid (`forced_adoptions` in [the definition](campaign.yaml)). Every later
seed is compared with the incumbent on fold 0.

## Scope

A seed may change:

- how the training labels of the students are built from the teacher table and the silver
  labels: the share and its form, the preparation of the teacher's values, which of the teacher's
  readings enter;
- the student's training targets and their loss when the seed adds a target per plane, behind
  one option.

Fixed in every seed: the split and its folds, the teachers and their table, the scoring targets
(`rebuild_state_v2` with gold where it exists), the backbone and its weights, the views, the
slices, the augmentations, the optimiser, the schedule and the epochs, the weights of the other
loss terms, and the inference multiplier and temperature of the student.

Out of bounds: retraining or replacing the teachers; teachers of another backbone or pretraining
([prior work](prior-work.md#what-is-settled)); reading the teacher table at validation or
prediction; reading fold 0's labels or its gold labels to build training labels or to choose
anything; rounding the teacher's values ([prior work](prior-work.md#fold-teachers-are-levelled-differently));
a second stage on the student's outputs; weighting the gold studies in the loss; fewer epochs;
weight averaging; multi-seed or confirmation runs; hyperparameter sweeps outside the share.

## How a seed is written

Each seed is one training run and states:

- **Evidence:** the numbers that motivate it, with their run or file.
- **Mechanism:** what changes in the label table or the loss, exactly, with the formula.
- **Replaces:** what it removes from the incumbent's recipe.
- **Shown wrong by:** the result that would contradict it.
- **Label:** evidence-backed or a guess.

A seed never only measures. What it needs to know about its label table (how far it moved from
the incumbent's, how it agrees with the silver labels per label) it reports once, before
training, from the table itself.

## Scoring and tracking

Two scores on fold 0 are followed for every run and kept side by side across the campaign:

- **Overall AUC:** macro AUC on the whole fold, at ×1 and at ×4.
- **Gold AUC:** macro AUC on the 58 gold studies, at ×4. Gold positives per label: effusion 35,
  synovitis 27, medial meniscus 26, ACL 24, lateral meniscus 23, PF OA 21, contusion 19,
  fracture 18, medial OA 15, Baker's cyst 12, lateral OA 11, MCL 9.

Both are logged at every epoch. Every result reports both, their differences to the incumbent
with paired report-group bootstrap intervals, and the per-label differences
of both, sorted from the largest gain to the largest loss. [Run notes](run-notes.md) holds one
table for the campaign, one row per run, with the overall and the gold score and their
differences in adjacent columns.

## Adoption

**Confirmed by the user on 2026-10-10.** Applied by the runtime after every run, without a
manual step: the three numbers are the `acceptance` block of [the definition](campaign.yaml)
([campaign rule](../../protocol.md#adoption-policy)). A run is adopted when all three hold, candidate minus
incumbent:

| Check | Number |
|---|---|
| Overall AUC at ×4 | ≥ +0.0009 |
| Overall AUC at ×1 | ≥ 0 |
| Gold AUC at ×4 | ≥ −0.002 |

Otherwise the run is recorded and discarded and the campaign continues. An adopted run becomes
the incumbent. One exception, also applied by the runtime: the valid run of `teacher-share-050`
is adopted without a comparison. How the numbers were chosen and what the rule does is in
[prior work](prior-work.md#the-adoption-rule).

**No submissions.** The campaign works locally and submits nothing to Kaggle; whether and when
students of the final recipe are submitted is the user's decision.

## Evaluation

The [shared evaluation](../../protocol.md#shared-evaluation-and-adoption) supplies the paired
bootstrap and the saved-prediction comparison; this campaign's split, its gold score and its
adoption rule replace the shared fold, the shared metric alone and the shared adoption rule.
Both are in the runtime.
