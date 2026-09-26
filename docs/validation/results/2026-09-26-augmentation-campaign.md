# 2026-09-26 augmentation campaign

Bilateral smoothing and DSU feature-statistics augmentation were adopted in the campaign. After
it closed, the user adopted the **half stack**: smoothing + DSU + intra-study contrast mixing +
GIN + one-axis resolution degradation. It scored +0.0011 ×4 macro locally and **0.943 public
(S64)**, against 0.940 for smoothing + DSU (S63) on the same fold and seed. Almost every
augmentation helped by about +0.001, but the gains did not add up when stacked.

## Setup

- **Campaign:** [augmentation](../../../autoresearch/campaigns/augmentation/INDEX.md), 13
  user-ordered seeds, then researcher refills, from 2026-09-25 to 2026-09-26. It started on Codex
  and switched to Claude after a Codex authentication failure (401 on a server-issued key that
  survived re-login). The candidates and their evidence are in
  [MRI augmentation research](../../research/mri-augmentation.md).
- **Reference:** `report-attributes-control`, seed 2464012069, fold 2, ×4 macro 0.898663.
- **Rule:** the [shared adoption rule](../../../autoresearch/protocol.md#shared-evaluation-and-adoption).
  Soft AUC is a diagnostic.

## Results

Hard and soft Δ are ×4 macro AUC against the incumbent at the time: the original control at seed
2464012069 (rows 1–8 and 12), the smoothing control retrain at seed 1704794368 (rows 9–10), and
smoothing + DSU at seed 1704794368 (rows 13–19).

| # | Idea | Hard Δ | 80% interval | Soft Δ | Verdict |
|---|---|---:|---|---:|---|
| 1 | One-axis k-space resolution degradation | +0.0012 | [−0.0008, +0.0032] | +0.0011 | reject |
| 2 | Unsharp sharpening | +0.0004 | [−0.0016, +0.0025] | +0.0002 | reject |
| 3 | Aligned image MixUp | — | — | — | invalid: checkpoint reload crash |
| 4 | Token MixUp | −0.0001 | [−0.0022, +0.0020] | +0.0002 | reject |
| 5 | Pooled-output MixUp | +0.0007 | [−0.0015, +0.0028] | +0.0019 | reject |
| 6 | Late affine shutdown | −0.0005 | [−0.0021, +0.0010] | +0.0009 | reject |
| 7 | Bias field | +0.0006 | [−0.0013, +0.0024] | +0.0009 | reject |
| 8 | Axial/coronal channel reversal | +0.0012 | [−0.0009, +0.0031] | +0.0003 | reject |
| 9 | **Bilateral smoothing** | **+0.0020** | [+0.0002, +0.0039] | +0.0006 | **adopted** |
| 10 | Intra-study contrast mixing | +0.0015 | [−0.0005, +0.0035] | +0.0016 | reject |
| 11 | **DSU feature statistics** | **+0.0022** | [−0.0005, +0.0046] | +0.0043 | **adopted by the user** |
| 12 | ReXNet stochastic depth 0.1 | +0.0011 | [−0.0008, +0.0030] | −0.0004 | reject |
| 13 | Aligned image MixUp rerun | +0.0006 | [−0.0017, +0.0029] | −0.0024 | reject |
| 14 | DSU with per-type covariance | +0.0007 | [−0.0014, +0.0030] | −0.0019 | reject |
| 15 | Study-coherent DSU | +0.0013 | [−0.0010, +0.0037] | −0.0020 | reject |
| 16 | GIN v3 | +0.0013 | [−0.0012, +0.0038] | −0.0016 | reject |
| 17 | Noise-referenced smoothing | +0.0018 | [−0.0003, +0.0040] | −0.0016 | reject |
| 18 | Half stack (10 + 16 + 1) | +0.0011 | [−0.0015, +0.0038] | −0.0019 | **adopted by the user** |
| 19 | Full stack (18 + 8 + 12) | +0.0016 | [−0.0010, +0.0044] | −0.0035 | not adopted |

Rows 18 and 19 were trained outside the campaign, with the campaign's comparison. GIN's first two
attempts never trained: a runner defect (an `overrides` key in the generated config), then a
per-slice valid-rectangle indexing bug in the candidate.

## Decisions outside the rule

- **Smoothing withdrawn, then restored.** Its fresh-seed control retrain scored 0.895831, 0.0049
  below the adopted run and 0.0028 below the original control. It was first withdrawn, then
  restored: the retrain is a re-seeded control, not a confirmation, and seed swings of this size
  are expected. Records: `outputs/autoresearch/augmentation/decisions/`.
- **DSU adopted on a replicated near-miss.** Its first run (+0.0022, on top of smoothing) missed
  the lower bound. A user-ordered rerun without smoothing at the original seed scored 0.899905,
  +0.0012, 80% [−0.0006, +0.0031] (`outputs/training/manual/dsu-without-smoothing`). The user
  adopted it because both runs cleared +0.001 with upper bounds several times the small negative
  lower bounds.
- **Campaigns no longer retrain controls.** The runner's automatic fresh-seed retrain after an
  adoption was removed; the adopted run itself becomes the incumbent
  ([seed policy](../../../autoresearch/protocol.md#training-seed-policy)).

## Seed noise

| Run | Seed | ×4 macro |
|---|---|---:|
| Original control | 2464012069 | 0.898663 |
| Smoothing | 2464012069 | 0.900702 |
| DSU only | 2464012069 | 0.899905 |
| Smoothing, retrain | 1704794368 | 0.895831 |
| Smoothing + DSU | 1704794368 | 0.897986 |

Smoothing alone moved 0.0049 between the two seeds, more than any paired augmentation effect.

## Public scores

| Submission | Checkpoint | Local ×4 | Public |
|---|---|---:|---:|
| S63 | Smoothing + DSU, epoch 9 | 0.897986 | 0.940 |
| S64 | Half stack, epoch 7 (selected) | 0.899054 | **0.943** |
| S65 | Half stack, epoch 9 (last, best soft AUC) | ×1 0.893525 | 0.943 |

S63 and S64 are paired (fold 2, seed 1704794368): +0.003 public is one pair at the edge of the
±0.003 public seed swing, weak evidence. S64 against S65 is the second hard-pick against
last/soft-pick pair after S41 (0.945) against S49 (0.944): a tie, so checkpoint selection stays on
hard AUC ([selection check](2026-09-23-validation-auc-variants.md#checkpoint-selection-check-s49)).
For reference, S62 (report-attribute heads v3.1, fold 1) scored 0.940.

The new control, `half-stack-control` (seed 2617320043, fold 3), scored ×4 0.905205 (×1 0.90207 at
epoch 9, where hard AUC, soft AUC and the last epoch coincide); fold 3 scores are not comparable
with fold 2.

## Findings

- **Augmentation helps a little, almost regardless of kind.** 15 of the 17 scored
  single-mechanism runs (including the DSU rerun) were positive, mostly +0.0004 to +0.0022, against three different controls. That fits
  a general regularization effect more than mechanism-specific gains.
- **The gains do not add.** Contrast mixing, GIN and resolution scored +0.0015, +0.0013 and
  +0.0012 alone and +0.0011 together; adding channel reversal and stochastic depth gave +0.0016.
- **Soft AUC falls with the dose.** The stacks lost 0.0019 and 0.0035 of soft macro AUC while hard
  AUC rose. In the half stack, hard AUC peaked at epoch 7 and soft AUC at epoch 9.
- **MixUp gains at ×1 did not survive ×4.** All three variants started slower and ended +0.002 to
  +0.006 ahead in in-training ×1 validation, but scored −0.0001 to +0.0007 at ×4.
- **Heavier augmentation learns slower.** The stacks trailed early and were still improving at
  epoch 9, so the 10-epoch budget may limit them; `trainer.max_epochs: 12` is the next test.
- **Public and local agree in direction only.** S64 gained +0.003 public on +0.0011 local.
- **The researcher stalled on the policy, not on ideas.** Its config prefixes covered only the
  seed families, so it could not propose a new family. Future campaigns need generic prefixes.
