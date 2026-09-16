# Local-to-public score relationship

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
