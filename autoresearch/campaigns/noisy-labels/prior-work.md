# Noisy-labels prior work

Evidence map for the [noisy-labels campaign](contract.md); not a backlog.

## How noisy the training labels are

- **Gold calibration of the targets.** On the 58 gold studies (696 cells), `combined_v1` targets in
  [0.3, 0.5) are gold-positive 32% of the time and targets in [0.9, 1] 78%. A single logistic map
  over all labels flattens and lowers them (slope 0.65–0.81)
  ([soft AUC evidence](../../../docs/validation/results/labels-and-losses/2026-09-23-validation-auc-variants.md#why-soft-targets-are-closer-to-the-truth)).
- **Where the incumbent ranks worst.** On fold 0, `k_05` ranks low-grade positives worst (AUC
  0.73–0.87 against 0.94–0.97 for high-grade), and only about half of those gold cells are
  positive. The negatives it ranks worst are ones the report never addressed, and 14% of such gold
  cells are positive ([report attributes](../../../docs/public-labels/report-attributes.md#agreement-with-the-incumbents-predictions)).
  A down-weighting rule must not simply silence low-grade positives, some of which are real.
- **Report text is exhausted as a fix.** Re-spacing report calls lowered gold agreement, so the
  remaining error sits within call classes and in omissions, which only images can correct
  ([image-blend result](../../../docs/validation/results/labels-and-losses/2026-09-23-image-blend-labels-result.md#why-an-image-teacher)).

## Measured effects of changing label trust

- **Class-balance exponent** (same seed, scanner split, `combined_v1`): public 0.943 unweighted,
  0.944 at k=0.25, 0.945 at k=0.5, 0.942 at k=0.75, 0.940 at k=1, while fold-0 hard AUC rose
  monotonically with k. The weights sharpen the report extractor's decision boundary on rare labels;
  the largest k=1 weights (MCL 17.4, Fracture 14.5) sit on the labels whose report positives
  disagree most with gold
  ([result](../../../docs/validation/results/labels-and-losses/2026-09-21-class-balance-exponent.md)). Against nine
  independent extractors the k=1 advantage halves but does not reverse, so no report-derived
  rescoring reproduces the public order; soft macro AUC does.
- **Offline image-teacher blend** (`image-blend`): 0.5 × `combined_v1` + 0.5 × an out-of-fold
  teacher from an older recipe. Gold agreement of the table +0.020 [95% +0.006, +0.036]; fold 0
  +0.0035 hard and +0.0065 soft (leaky, the teacher saw fold 0); public 0.945 (S50) against 0.942
  for the same-code control (S48), a tie by the rule fixed in advance. PF OA lost on every measure
  because the teacher learned looser definitions
  ([result](../../../docs/validation/results/labels-and-losses/2026-09-23-image-blend-labels-result.md)). The matched
  five-fold version is endgame work and outside this campaign.
- **Public reports.** Participants at the top of the board name label handling as their main axis;
  one reports that mixing image-model predictions into the extracted labels beat both replacing and
  masking them, and that rounding soft labels to 0/1 hurt
  ([public research](../../../docs/research/competition-and-prior-art/public-research.md#labels-are-the-reported-lever-at-the-top-and-extractor-tuning-is-not)).

## Already tested, do not repeat

- **Masking unaddressed cells** out of the loss: −0.011455
  ([result](../../../docs/history/sharing/2026-09-13-label-masking-result.md)).
- **Losses campaign** on `all_5fold_v3` with Steven v4 labels
  ([findings](../losses/findings.md)): all label-derived. Soft-mass balancing −0.0092, prior offset
  −0.0119, biproportional weights −0.0043 and pair/contrast rankings were rejected; focal (−0.0026),
  Brier, BCE + ranking, boundary-proximity weighting, evidence and target-logit margins, and the
  draw-consistency anchor were inconclusive. Its researcher closed the GCE, symmetric-CE,
  Huberised-BCE and forward-correction family by analysis, without a run. All were judged on hard
  AUC against report labels, the judge the exponent result discredits for weighting, so a
  rejection there does not settle a prediction-driven mechanism here. None used the model's own
  predictions to correct or down-weight labels.

## Ruled out by user decision

- Per-label selection, coefficients, losses or schedules.
- Weighting the gold studies in the loss, or meta-learning weights on them.
