# Local-to-public score relationship

Measured on this project's own submissions:
[2026-09-18 local-public metric comparison](results/2026-09-18-local-public-metric-comparison.md).
Over the nineteen submissions whose split, label table, validation multiplier and inference
settings all match, fold-0 macro AUC correlates with the public score at Pearson +0.907 and
soft-target BCE at +0.833, while **gold-only AUC and gold-only BCE carry no signal**. Since
2026-09-13 the submitted models' local scores differ by less than one training run's noise, so no
local metric ranks them.

The [2026-09-28 update](results/2026-09-18-local-public-metric-comparison.md#2026-09-28-update)
covers S23–S77 in eight cohorts: on public differences of at least 0.003, predictions scored
against `rebuild_state_v2` agree with the leaderboard on 78–83% of pairs, against 71–73% for
`combined_v1`; ×1 and ×4 track equally; gold-only AUC stays below chance; and each label set
favors models trained on labels like itself by about ±0.005, so a change of training labels needs
a submission.

The [2026-09-29 grouped-AUC search](results/2026-09-18-local-public-metric-comparison.md#2026-09-29-grouped-auc)
found no better tracker that holds up out of sample. Averaging AUC over sites, scanner groups,
scanner models, vendors or field strengths either overfits or hinges on an arbitrary
group-size threshold. The scanner adversarial runs rank the same on every metric.

What follows from that:

- Keep full-fold macro AUC as the selection metric; use BCE only as supporting evidence inside one
  label table, and do not let the fold's 11–13-study gold diagnostics influence a decision.
- Treat a local difference below about 0.005 macro AUC in a single run as unresolved regardless of
  which metric agrees, and prefer a leaderboard result over a sub-0.005 fold-0 difference when the
  two disagree ([validation strategy](validation-strategy.md#what-a-comparison-can-resolve)).
- Compare only submissions that differ in the trained model. Test density, attention temperature
  and the label table each move the public score by more than recent recipe changes do.

## Earlier working notes

These predate the measurement above and are retained for context only.

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
