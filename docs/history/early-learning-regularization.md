# Early-learning regularization on the current recipe

**Closed, 2026-09-25: not adopted.** Ported unchanged (λ 0.075, momentum 0.7) as seed #5 of the
[near-misses campaign](../validation/results/2026-09-25-near-misses-campaign.md), it scored
+0.0002 hard and +0.0004 soft ×4 macro against `knee-geometry-merged-2` at seed 638120755. The
analysis below motivated that run.

## Candidate

- **Run:** `early-learning-regularization-20260923T124737Z-7aba30fc`
  (`outputs/autoresearch/noisy-labels/experiments/`), from the
  [noisy-labels campaign](../validation/results/2026-09-23-noisy-labels-campaign.md).
- **Recipe:** `k05-clean` (fold 0 of `all_5fold_scanner_v1`, combined labels, k=0.5, seed
  2263738815, cleaned source) plus `trainer.label_noise: {momentum: 0.7, weight: 0.075}`.
- **Term:** a per-study running average of the model's own predictions, plus the penalty
  `log(1 − overlap)` between that average and the current prediction. The code (about 80 lines in
  `config.py` and `training/lightning.py`) lives only in the run's `changes.diff` and `source/`.
- **Selected checkpoint:** epoch 8. ×1 0.900934, ×4 0.904068.
- **Public:** 0.943 as S57 (×4, τ 0.75, batch 4, bundle `1d22fc40`), against 0.942 for S48
  (`k05-clean`).

## Per-label comparison

**Method:** paired report-group bootstrap, 2,000 draws, seed 20260914, on the saved ×4 fold-0
predictions. Hard AUC thresholds targets at 0.5; soft AUC is the repo's `soft_auc`. Both
reproduce `evaluation.metrics` exactly on the unweighted draw.

**Comparisons:**

- **Paired control:** `k05-clean`, with the same seed and source.
- **Siblings:** the three near-null noisy-labels runs on the same seed and reference, namely
  bootstrapped targets, running-average targets and the 1→0 class-balance schedule. Their macro
  moved −0.0005, +0.0002 and −0.0000. Subtracting their mean removes luck in the shared
  reference.
- **Seed noise:** `adopted-new-seed` against `meniscus-half-knee-mil`, the same recipe at two
  seeds.

**Artifacts:** `outputs/analysis/elr-per-label-20260924/` (`elr_per_label.py`,
`elr_per_label.json`), computed on 2026-09-24 at commit `70a9a24`.

### Against `k05-clean`

| Group | Hard Δ [80%] | Soft Δ [80%] |
|---|---|---|
| Ligaments (ACL, MCL) | +0.0041 [−0.0021, +0.0101] | +0.0034 [+0.0004, +0.0062] |
| Menisci | +0.0030 [+0.0003, +0.0055] | +0.0015 [−0.0006, +0.0034] |
| OA | −0.0012 [−0.0034, +0.0011] | −0.0015 [−0.0031, +0.0002] |
| Effusion, Synovitis, Baker's | −0.0001 | −0.0003 |
| Contusion, Fracture | +0.0027 [−0.0032, +0.0088] | +0.0025 [−0.0001, +0.0052] |
| Macro | +0.0013 [−0.0004, +0.0031] | +0.0008 [−0.0002, +0.0018] |

**Per label, hard:**

- Gains: ACL +0.0056 (80% [+0.0008, +0.0103]), Fracture +0.0037, Lateral Meniscus +0.0036,
  Medial OA +0.0030, MCL +0.0025, Medial Meniscus +0.0023.
- Losses: Lateral OA −0.0032, PF OA −0.0032, Baker's −0.0017.

### Minus the sibling mean

| Group | Hard Δ [80%] | Soft Δ [80%] [95%] |
|---|---|---|
| Ligaments | +0.0045 [+0.0012, +0.0079] | +0.0037 [+0.0017, +0.0057] [+0.0007, +0.0067] |
| Menisci | +0.0031 [+0.0005, +0.0058] | +0.0019 [−0.0001, +0.0037] |
| OA | +0.0000 | −0.0003 |
| Baker's | −0.0050 [−0.0100, −0.0001] | −0.0045 [−0.0077, −0.0013] |
| Macro | +0.0014 [−0.0000, +0.0027] | +0.0008 [−0.0000, +0.0016] |

## Reading

- **Ligaments are the strongest effect.** Soft ligaments exclude zero at 95% once the siblings'
  mean is removed.
- **Menisci gain at the family level.**
- **Baker's is the only loss specific to ELR.**
- **The OA dip is shared.** The siblings also lose Lateral OA (bootstrapped −0.0038, running
  average −0.0067), so against their mean ELR's OA change is 0.0000.
- **Multiple comparisons:** twelve labels would put about two outside an 80% interval by chance.
  The family-level results carry more weight than single labels.
- **ELR harms OA in combination.** Schedule plus ELR, measured against the schedule alone, lost
  Lateral OA −0.0086 and PF OA −0.0075. The OA risk comes from interaction with another loss
  term, not from ELR alone.
- **Seed noise is larger and unrelated.** Re-seeding the adopted model moved Contusion −0.0205,
  PF OA −0.0154, Fracture −0.0137 and MCL −0.0129 (macro −0.0063). Its per-label pattern is
  uncorrelated with ELR's (r = +0.01).
- **Same-seed pairing is much tighter.** The siblings stay within ±0.0005 macro, so a test of ELR
  must use the incumbent's seed.
- **ELR sharpens predictions rather than re-ranking them.**
  - Mean |p − 0.5| rises on every label, by 0.011 to 0.022.
  - Soft-target BCE worsens from 0.4386 to 0.4401.
  - Rank correlation with `k05-clean` is 0.97 to 0.99.
  - Together with the campaign's finding that the model does not memorize its labels, this
    points to confidence sharpening, not noise robustness.

## Expectations for the current recipe (hypotheses)

- **Ligaments are the likeliest additive gain.** The adopted model's own gains over `k05-clean`
  sit in the menisci (+0.0039) and Contusion/Fracture (+0.0083); its ligaments moved only
  +0.0016. Its per-label pattern correlates with ELR's at r = +0.32.
- **The meniscus gain may overlap** with the half-knee MIL loss.
- **Main risk:** an OA interaction with the half-knee MIL loss, like the one seen with the
  schedule.
- **Checks to fix before the run:** ligaments above zero, OA no worse than about −0.002, and a
  Baker's loss no larger than about −0.005.
- **Run shape:** one run with λ 0.075 and momentum 0.7 unchanged, against
  `knee-geometry-merged-2` at seed 638120755. A single run will probably not clear the adoption
  rule's 80% lower bound even if the +0.001 is real. Decide beforehand whether a public pair
  against S55 (0.940) settles it.
