# Target-level diagnostics

- Earlier notes used macro ROC-AUC, now confirmed by the
  [official metric](official-metric-and-public-evidence.md). Single-class-column
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

## Cost of a reversed pair

A target's ROC-AUC is the fraction of its (positive, negative) study pairs ranked correctly, so
with `P` positives and `N` negatives, reversing one pair costs `1/(P × N)` of that target's AUC
and **`1/(12 × P × N)` of the macro AUC**. Targets with few positives therefore carry the most
expensive mistakes, while the target with the largest total deficit (1 − AUC) is the one pulling
the average down most; the two rankings need not agree.

Fold 0 of `all_5fold_v3` (904 studies, positives are targets ≥ 0.5, as in
`validation_metrics`), computed on 2026-09-16:

| Target | Positives | Negatives | Pairs | Macro AUC per reversed pair |
|---|---:|---:|---:|---:|
| Fracture | 63 | 841 | 52,983 | 1.573 × 10⁻⁶ |
| Synovitis | 115 | 789 | 90,735 | 9.18 × 10⁻⁷ |
| MCL | 134 | 770 | 103,180 | 8.08 × 10⁻⁷ |
| Lateral Meniscus | 138 | 766 | 105,708 | 7.88 × 10⁻⁷ |
| Contusion | 153 | 751 | 114,903 | 7.25 × 10⁻⁷ |
| ACL | 184 | 720 | 132,480 | 6.29 × 10⁻⁷ |
| Baker's | 216 | 688 | 148,608 | 5.61 × 10⁻⁷ |
| Lateral OA | 241 | 663 | 159,783 | 5.22 × 10⁻⁷ |
| Medial OA | 329 | 575 | 189,175 | 4.41 × 10⁻⁷ |
| Medial Meniscus | 356 | 548 | 195,088 | 4.27 × 10⁻⁷ |
| Effusion | 525 | 379 | 198,975 | 4.19 × 10⁻⁷ |
| PF OA | 406 | 498 | 202,188 | 4.12 × 10⁻⁷ |

- A fracture pair costs 3.8× a patellofemoral OA pair.
- **Example:** the [crop224 run](results/2026-09-16-crop224-result.md) lost 0.014250 fracture AUC against
  `00-baseline`, about 755 reversed pairs, which alone costs 0.001187 macro AUC, more than that
  run's whole −0.001139 gap.
- A table shared on 2026-09-16 from an unrecorded source (881 studies per target, 43 MCL and 47
  fracture positives, possibly another split or the sparser official labels) made the same point:
  one MCL pair cost 5.23× a medial-meniscus pair there, while synovitis had the largest total
  deficit. Its counts do not match this split.
