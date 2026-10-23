# Local-to-public score relationship

Full-fold macro AUC was the best local predictor of the public score, but local differences below
about 0.005 in a single run were unresolved, gold-only diagnostics carried no signal, and for the
late ensembles only the board could decide. The measurements are in
[local-public metric comparison](results/validation-and-ensembles/local-public-metric-comparison/INDEX.md).

## Measurements

- **2026-09-18, nineteen comparable submissions** (split, label table, validation multiplier and
  inference settings all matching; [analysis](results/validation-and-ensembles/local-public-metric-comparison/2026-09-18-comparable-submissions.md)):
  fold-0 macro AUC correlates with the public score at Pearson +0.907 and soft-target BCE at +0.833,
  while **gold-only AUC and gold-only BCE carry no signal**. Since 2026-09-13 the submitted models'
  local scores differ by less than one training run's noise, so no local metric ranks them.
- **2026-09-28, S23-S77 in eight cohorts** ([update](results/validation-and-ensembles/local-public-metric-comparison/2026-09-28-label-sets-and-multipliers.md)):
  on public differences of at least 0.003, predictions scored against `rebuild_state_v2` agree with the
  leaderboard on 78-83% of pairs, against 71-73% for `combined_v1`; ×1 and ×4 track equally; gold-only
  AUC stays below chance; and each label set favors models trained on labels like itself by about
  +-0.005, so a change of training labels needs a submission.
- **2026-09-29, grouped AUC** ([search](results/validation-and-ensembles/local-public-metric-comparison/2026-09-29-grouped-auc.md)):
  no better tracker holds up out of sample. Averaging AUC over sites, scanner groups, scanner models,
  vendors or field strengths either overfits or hinges on an arbitrary group-size threshold. The
  scanner adversarial runs rank the same on every metric.
- **2026-10-19, label tables:** the v3 table was chosen over v2, v3.1 and v3.2 by its agreement with
  the board over a fixed 26-submission panel (Spearman 0.429769 against v2's 0.196407)
  ([v3.2 labels](../public-labels/audits/2026-10-19-v32-labels.md)).
- **2026-10-20 to 2026-10-22, attention temperature:** local silver-scored validation always preferred
  tau 0.85 while the board preferred 0.4-0.45, because silver labels come from reports and sharpening
  moves predictions toward what radiologists see. The gold analyses of 2026-10-21 had been scored
  against silver by mistake
  ([attention temperature](results/pooling-and-attention/2026-10-22-attention-temperature.md)).
- **2026-10-16, gold-fitted combiners:** none of the stackers fitted on the 58 gold studies beat the
  equal average of S163's three families held out on gold; S164 (per-label weights tuned on gold, in-sample
  +0.0046) scored 0.963 public against 0.964 for S163
  ([result](results/validation-and-ensembles/2026-10-16-gold-fitted-stacking.md)).

## What follows

- Keep full-fold macro AUC as the selection metric; use BCE only as supporting evidence inside one
  label table, and do not let the fold's 11-13-study gold diagnostics influence a decision.
- Treat a local difference below about 0.005 macro AUC in a single run as unresolved regardless of
  which metric agrees, and prefer a leaderboard result over a sub-0.005 fold-0 difference when the two
  disagree ([what a comparison can resolve](validation-protocol.md#what-a-comparison-can-resolve)).
- Compare only submissions that differ in the trained model. Test density, attention temperature and
  the label table each move the public score by more than recent recipe changes do.
- Tune tau on the board or on radiologist labels only, never on silver; label changes need a
  submission, because fixed labels favor their own training.

## Earlier working notes

These predate the measurements above and are kept for context.

| Stage | Reported observation |
|---|---|
| 2026-08-28 | Label refinement using out-of-fold image predictions improved agreement with gold labels |
| 2026-09-02 | Attempts to obtain more strong labels repeatedly ran into circular reasoning |
| 2026-09-03 | Local improvements appeared to transfer to public scores, even with roughly 11 gold studies per validation fold |
| 2026-09-09 | Repeated fold-0 training and label iterations produced local AUC 0.977, which was later considered no longer meaningful |
| 2026-09-09 | Five-fold gold-label performance was still described as correlated with Kaggle results |

Holding out all 58 gold studies and training on the rest was proposed as an alternative; the notes
do not establish that this split was implemented then (`gold_holdout_5fold_v1` of 2026-10-10 is a
later split for comparing student recipes, see [validation splits](validation-splits.md)). The small
gold set and prior use of its labels remain relevant: preserve independent evaluation when refining
labels or selecting windows, and treat repeatedly tuned validation and public scores as selection
evidence, not a fresh estimate of generalization. The notes give no split manifests, patient grouping
rules or detail enough to diagnose a specific leakage mechanism.

## Generalization evidence from the sharing experiments

The [sharing experiment record](../history/sharing/INDEX.md) preserves the 1A, 1B and 1C per-target and
subgroup comparisons, compartment interventions, public-score audits, the completed eleven-run 1C
hyperparameter search and its sigma and learning-rate follow-ups. It also records the rejected
evidence-mask experiment: removing originally unmentioned report targets reduced validation AUC by
**0.011455**. These results use their original configurations and source archives.

The [generalization investigation](../history/sharing/generalization-investigation.md) combines paired
bootstrap and case-mixture sensitivity with the actual submission records. The
[disagreement audit](../history/sharing/oa-and-meniscus-disagreement-audit.md) adds 48 blinded report
reviews and representative acquisition headers for all 4,407 studies. OA severity conflicts and
manufacturer-dependent ranking changes are measured vulnerabilities; neither establishes the cause of
Kaggle's hidden ranking. The [transfer research](../research/geometry-and-views/medial-lateral-transfer.md)
owns the retained approaches.
