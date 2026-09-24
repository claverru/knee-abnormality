# 2026-09-23 noisy-labels campaign: prediction-driven label handling does not act here

**Six training-objective changes aimed at wrong report labels, none adopted.** Every run used
fold 0 of `all_5fold_scanner_v1`, `combined_v1` labels, seed 2263738815, ×4 τ 1 predictions and
the same recipe as the reference `k05-clean` (S48, 0.942 public; ×4 hard 0.902760, soft 0.798458).
The [campaign](../../../autoresearch/campaigns/noisy-labels/INDEX.md) holds the contract, the
per-experiment findings and the user decisions.

| Idea | Hard AUC Δ | Adoption 80% interval | Soft AUC Δ |
|---|---:|---|---:|
| Bootstrapped targets, 0.8·y + 0.2·detached p from epoch 2 | −0.0005 | [−0.0014, +0.0006] | −0.0003 |
| Running-average targets, 0.7·y + 0.3·memory (momentum 0.7) | +0.0002 | [−0.0010, +0.0016] | −0.0004 |
| Early-learning regularization, λ 0.075, momentum 0.7 | +0.0013 | [−0.0004, +0.0031] | +0.0008 |
| Confident-contradiction down-weighting, weight 0.25 | −0.0060 | [−0.0074, −0.0045] | −0.0014 |
| Class-balance exponent 1→0 linear over steps | −0.0000 | [−0.0017, +0.0017] | +0.0007 |
| Schedule plus early-learning regularization | −0.0003 | [−0.0021, +0.0017] | +0.0002 |

## Why the family barely acts

- **The model does not memorize its training labels in this regime.** Label-noise methods built
  on the model's own predictions (bootstrapping, temporal ensembles, early-learning
  regularization, small-loss down-weighting) protect against a late memorization phase. Ten epochs
  of pretrained fine-tuning with augmentation and a fresh triplet draw per visit do not reach it:
  the reference's validation BCE still falls at the last epoch and its best epoch is 8 of 10. The
  campaign researcher measured the same on saved predictions: EMA weights trail the live model
  until epochs 8–9, averaging EMA and live predictions gains +0.0006, and two independent seeds
  averaged gain +0.0028. A teacher from the same run carries no new information; only an
  independent model does, which is the endgame
  [out-of-fold self-distillation](../../backlog/endgame.md#out-of-fold-self-distillation).
- **Bootstrapping with a detached own prediction is degenerate.** The BCE gradient toward
  β·y + (1 − β)·p is exactly β times the ordinary gradient, which AdamW normalizes away. Its null
  result says nothing about bootstrapping from a real teacher.
- **Running-average targets moved the labels only 0.05–0.07 on average,** because the model's
  average already agrees with the soft labels it fits. They cost the OA family on both judges
  (Lateral OA −0.0067 hard).
- **Down-weighting hurt on both judges.** Part of its hard loss is on labels it chose to ignore
  (soft only −0.0014), but a −0.006 loss is well beyond the gap between the two judges.

## Class-weighting changes need the public board

The 1→0 schedule behaved as designed (exponent 0.95 → 0.05 per epoch, maximum weight ratio 17.4 →
1.35) and tied on hard AUC with a small soft gain. Per label it showed the judge split expected
from the [class-balance exponent result](2026-09-21-class-balance-exponent.md): Contusion −0.008
and MCL −0.005 on hard AUC, 0.000 and +0.001 on soft. Weighted by the learning-rate schedule its
effective exponent is 0.66, between k=0.5 (0.945 public) and k=0.75 (0.942). Fixed exponents moved
the public score by 0.005 while fold 0 ranked them in reverse, so a local tie does not settle it.
Submitted from its experiment source as S53 (×4, τ 0.75, batch 4, bundle 2036c2b7), it scored
0.941 public against 0.942 for S48, so decaying the weighting to zero does not recover k=0.5's
0.945. Its notebook and dataset are titled S55 (the ledger's S55 is a different submission) because a stalled first attempt was stopped
before submitting and the relaunch set the sequence by hand.

Combining them was worse than either alone. Early-learning regularization pulls predictions toward
their recent average while the schedule wants them to move late, and the pair lost Lateral OA
(−0.0086) and PF OA (−0.0102) with intervals excluding zero.

## Artifacts

Experiments, sources and checkpoints: `outputs/autoresearch/noisy-labels/experiments/`. The schedule
run is `relaxing-class-balance-schedule-20260923T133826Z-824ef437` (S53, bundle
`outputs/submission/exports/cbschedule-20260924-x4-tau075`); the unsubmitted early-learning
candidate is `early-learning-regularization-20260923T124737Z-7aba30fc`. Soft AUC deltas are recomputed from
each run's saved ×4 predictions with `soft_aucs`; they have no paired interval.
