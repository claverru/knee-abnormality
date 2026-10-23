# 2026-10-16 Gold-fitted stacking and label correction

**No combiner fitted on the 58 gold studies beats the equal average of S163's three families held
out on gold, and a per-label gold-fitted label correction does not beat a fixed teacher share.**
Keep equal member weights; S164 (per-label weights tuned on gold, in-sample gold +0.0046) scored
0.963 public against 0.964 for S163, the same verdict. Investigation closed; nothing adopted.

## Setup

- **Members:** S163's fifteen joint models (`{mambavision,convnext,resnet50}-silver15-joint-oai-soft-fold-{0..4}`),
  each family's five `validation_predictions.csv` concatenated into one out-of-fold set (×4, τ 1.0,
  no mirror TTA; not the submission settings).
- **Targets:** the 58 gold studies of `all_5fold_scanner_v2` (`load_split` membership `is_gold`),
  positive at ≥ 0.5. Gold positives per label: ACL 24, MCL 9, Medial Meniscus 26, Lateral
  Meniscus 23, Medial OA 15, Lateral OA 11, PF OA 21, Effusion 35, Synovitis 27, Baker's 12,
  Contusion 19, Fracture 18.
- **Baseline:** mean of the three family probabilities, which is what S163's bundle computes per
  study out of fold.
- **Stackers:** one per label. Inputs either the label's three family logits ("own", 58 × 3) or all
  36 logits ("all", 58 × 36). L2 logistic regression (standardised, C = 1) and TabPFN v2 (the
  ungated `ModelVersion.V2` weights; v2.5 and v2.6 need a Prior Labs licence token), CPU.
- **Held-out score: leave-pair-out AUC.** Every positive–negative pair is held out together, the
  stacker is fitted on the other 56 studies and the pair counts as a win when the positive scores
  higher. Pooled leave-one-out AUC is biased against fitted models: removing a positive lowers the
  training base rate and so that study's own score, and the reverse for a negative. Logistic "own"
  went from −0.0144 under leave-one-out to −0.0005 under leave-pair-out; the leave-one-out numbers
  are withdrawn. The unfitted baseline scores the same under both.

## Stacking result

| Combiner | Input per label | Macro AUC on gold | Δ |
|---|---|---:|---:|
| Equal average (S163) | | 0.949380 | |
| Logistic regression, own | 58 × 3 | 0.948893 | −0.0005 |
| TabPFN v2, own | 58 × 3 | 0.945299 | −0.0041 |
| TabPFN v2, all | 58 × 36 | 0.934439 | −0.0149 |
| Logistic regression, all | 58 × 36 | 0.909443 | −0.0399 |

The more a combiner fits, the more it loses. Per label, largest gain to largest loss:

- **Logistic, own:** Lateral OA +0.0077, Baker's +0.0018, five labels ±0.0000, Lateral Meniscus
  −0.0012, Medial OA −0.0016, MCL −0.0023, Synovitis −0.0048, Fracture −0.0056.
- **TabPFN, own:** Synovitis +0.0203, Lateral OA +0.0097, Baker's +0.0036, Medial Meniscus +0.0024,
  ACL −0.0012, Effusion −0.0012, Fracture −0.0069, Contusion −0.0094, Medial OA −0.0109, Lateral
  Meniscus −0.0137, MCL −0.0159, PF OA −0.0257.
- **TabPFN, all:** Baker's +0.0036, ACL +0.0012, Effusion ±0.0000, then losses to Synovitis −0.0502.

Equal-average gold AUC per label: ACL 0.996324, Effusion 0.991304, Medial OA 0.986047, Baker's
0.983696, MCL 0.977324, Medial Meniscus 0.977163, Contusion 0.952767, Fracture 0.945833, PF OA
0.929215, Lateral Meniscus 0.926708, Lateral OA 0.899420, Synovitis 0.826762.

TabPFN at prediction time would cost seconds per label (one fit on 58 rows, one batched predict);
cost was not the reason to drop it.

## Label correction

The same idea applied to training targets: fit `P(gold | silver, teacher)` per label on the gold
studies and apply it to every training study. Silver on gold rows is `combined_v1` (gold rows keep
those values in `rebuild_state_v2`); the teacher is the mean of the ResNet-50 and ConvNeXt
silver-only teachers' out-of-fold predictions (`resnet50-silver-fold{0..3}`, `rn50-silver-fold-4`,
`convnext-silver-fold-{0,1,3,4}`, `convnext-silver-only-imagenet-fold-2`; ×4, τ 1.0, no TTA).
Agreement of each label table with gold, macro AUC:

| Table | Gold agreement |
|---|---:|
| Silver (`combined_v1` on gold rows) | 0.9075 |
| Teacher alone | 0.9416 |
| 0.75 silver + 0.25 teacher | 0.9379 |
| 0.5 + 0.5 | 0.9422 |
| 0.25 silver + 0.75 teacher | 0.9473 |
| Per-label logistic on (silver logit, teacher logit), leave-pair-out | 0.9417 |

The fitted correction lands at the teacher's level. Its full-data weights favour the teacher on ten
labels (all but ACL and MCL), but 56 studies do not set a per-label share better than one global
value. The 0.75 teacher share leads the fixed shares (+0.0094 over 0.25, largest on Contusion
+0.039 and Effusion +0.010); it was picked among three shares on these same studies, table
agreement is not student AUC, and the [self-distillation rule](../labels-and-losses/2026-10-04-two-family-oof-distillation.md)
fixes the share without gold, so it stays a reading, not a queued run.

## Artifacts

One-off scripts and outputs were kept in the session scratchpad, not the repository
(`baseline.py`, `loo.py`, `lpo.py`, `correction.py`); TabPFN ran in a separate CPU venv
(`tabpfn` 9.2.0, torch 2.14.1+cpu).
