# 2026-09-28 plane-specialization campaign

**Plane-specific backbones are not adopted (user decision, 2026-09-28).** One backbone per plane
(axial, coronal, sagittal), each seeing only its plane's images, gained from one change only: dense
per-plane readouts, where every plane predicts every label from its own tokens (+0.0047). Everything
built on top of that tied or lost. On B0 the best model trails the ReXNet control by 0.0057 locally
and 0.003 publicly; the same structure on three ReXNet-200s ties the control (+0.0020 locally, 0.945
against 0.946 publicly) at about 35% more training time and twice the Kaggle inference time. The
line's history is in [plane-specific backbones](../../history/plane-backbones.md).

## Setup

- **Campaign:** [plane-specialization](../../../autoresearch/campaigns/plane-specialization/INDEX.md)
  on Codex, 2026-09-27 to 2026-09-28: 9 user seeds, researcher refills, then 10 user-forced ideas
  added on the fly. Stopped by the user.
- **Reference:** `20260927-210053-b0-plane-backbones`, three EfficientNet-B0s, one per plane, joint
  attention pooling over all planes, batch 16, `backbone_lr` 8e-4, `head_lr` 1.2e-3, DSU off, series
  dropout 0, fold 4 of `all_5fold_scanner_v1`, seed 703791404: ×4 0.898434.
- **Yardstick:** `scanner-confusion-control` (one ReXNet-200, batch 8, DSU 0.5, series dropout 0.2),
  same fold and seed: ×4 0.909733, public 0.946 (S73).
- **Rule:** the [shared adoption rule](../../../autoresearch/protocol.md#shared-evaluation-and-adoption).
  Δ is ×4 macro AUC against the incumbent at the time: the reference for runs 1–3, `dense-plane-readouts`
  for 4–7, `dense-readouts-learned-fusion` from 8 on.

## Results

| # | Idea | Δ | 80% interval | Verdict |
|---:|---|---:|---|---|
| 1 | Random label-to-plane partition each step, plane mean at inference | −0.0190 | [−0.0228, −0.0154] | reject |
| 2 | Same, learned per-label plane fusion | −0.0149 | [−0.0184, −0.0116] | reject |
| 3 | **Dense per-plane readouts** (every plane predicts every label), plane mean | **+0.0047** | [+0.0024, +0.0069] | **accepted** |
| 4 | **Dense readouts, learned fusion** (softmax weights on detached plane logits) | +0.0009 | [−0.0006, +0.0024] | **adopted by the user** (×4 0.904035) |
| 5 | Per-plane classifiers | +0.0005 | [−0.0009, +0.0019] | reject |
| 6 | Fixed radiology label-to-plane mask | −0.0011 | [−0.0036, +0.0015] | reject |
| 7 | Joint pooling plus training-only per-plane readouts | +0.0002 | [−0.0021, +0.0023] | reject |
| 8 | Winner-take-all plane routing | −0.1371 | [−0.1457, −0.1287] | reject (collapse) |
| 9 | Full independent head per plane | −0.0002 | [−0.0018, +0.0014] | reject |
| 10 | Fusion-guided dense routing | −0.0021 | [−0.0037, −0.0006] | reject |
| 11 | Joint attention fusion on detached tokens | −0.0009 | [−0.0029, +0.0011] | reject |
| 12 | Per-label MLP fusion plus 20% triplet dropout | −0.0066 | [−0.0085, −0.0048] | reject |
| 13 | Per-label MLP fusion (6 → 16 → GELU → 1) | −0.0008 | [−0.0026, +0.0009] | reject |
| 14 | Cross-label MLP fusion (39 → 32 → GELU → 12) | −0.0007 | [−0.0022, +0.0008] | reject |
| 15 | Radiology-mapped auxiliary plane loss | −0.0003 | [−0.0018, +0.0011] | reject |
| 16 | Leave-one-plane-out distillation | −0.0043 | [−0.0060, −0.0027] | reject |
| 17 | Attention-entropy auxiliary | — | — | invalid (force-stopped) |
| 18 | DSU inside each B0 at the stride-8 map | −0.0042 | [−0.0059, −0.0024] | reject |
| 19 | One OA detector per plane | −0.0016 | [−0.0029, −0.0003] | reject |
| 20 | Report-attribute heads per plane | −0.0026 | [−0.0041, −0.0012] | reject |
| 21 | Half-knee MIL per plane | −0.0006 | [−0.0019, +0.0006] | reject |
| 22 | Scanner confusion per plane | −0.0002 | [−0.0015, +0.0012] | reject |
| 23 | Evidence-statistics fusion (attention shape, token scores, slice spread, coverage) | −0.0004 | 95% [−0.0023, +0.0015] | tie, decided on the next resume |

A shared stride-8 stem across the plane backbones was blocked at review: it contradicts the
campaign's premise that each backbone learns from its own plane.

## Findings

- **Dense supervision is the only gain.** Training every plane on every label every step beat joint
  pooling; removing supervision (random partitions, routing, winner-take-all, triplet dropout) lost.
  Winner-take-all collapsed because planes that lose early stop receiving gradient.
- **Fusion capacity does not matter** once each plane predicts every label: linear weights, per-label
  and cross-label MLPs, joint attention and evidence statistics all tie within ±0.001.
- **Plane-specific versions of the auxiliary pieces** (OA detector, attribute heads, half-knee MIL,
  scanner confusion) tie or lose on B0.
- **Regularizers hurt the underfit B0:** DSU lost 0.0042 here although it gained on ReXNet
  (+0.0022, rerun +0.0012). Regularization findings from this campaign do not transfer to ReXNet.
- **The recurring per-label pattern is incumbent noise:** Contusion fell and Lateral OA rose against
  the incumbent in most runs, across unrelated mechanisms; the incumbent's Contusion (0.898) sat
  above its parent's (0.891). Averaging the incumbent with the 12 runs within ±0.002 of it (logit
  mean, from saved ×4 predictions) scored 0.9066, +0.0026 (80% [+0.0015, +0.0036]), better on 10
  of 12 labels: a seed-ensemble effect for the endgame, not a design gain.

## Transfer to ReXNet-200

`outputs/training/20260928-105516-rexnet-plane-dense-fusion`: the adopted structure (dense readouts,
learned fusion, series dropout 0) on three ReXNet-200s with the control's batch 8, learning rates
2e-4 / 6e-4 and DSU 0.5 inside each plane backbone. Fold 4, seed 703791404.

| Model | ×4 local | Public |
|---|---:|---:|
| ReXNet control, one shared backbone | 0.909733 | 0.946 (S73) |
| 3 × B0 plane backbones (campaign incumbent, epoch 8) | 0.904035 | 0.943 (S74) |
| 3 × ReXNet plane backbones (epoch 9) | 0.911714 | 0.945 (S77) |

- Three ReXNets against the control: +0.0020, study bootstrap 80% [−0.0003, +0.0044]; MCL +0.020,
  Fracture +0.009, Contusion −0.007, PF OA −0.005. Training 2,600 s against 1,925 s, peak 21.9 GiB.
- Three ReXNets against three B0s: better on six labels with 95% intervals above zero (MCL, Lateral
  OA, Fracture, Synovitis, Medial OA, Medial Meniscus), no label better on B0, and every plane's own
  ×1 AUC higher (axial +0.007, coronal +0.006, sagittal +0.009). Both learn the same plane weighting
  (Baker's on axial, menisci on coronal and sagittal, PF OA on sagittal).

## Attention temperature

Fold-4 predictions at ×4 on Kaggle T4s in FP16, one bundle per τ through
`scripts/tools/profile_kaggle_inference.py launch --split-dir data/folds/all_5fold_scanner_v1 --fold 4
--conditions '["fp16-b4-w16"]'`, scored locally against τ 0.75 with a paired study bootstrap
(`outputs/analysis/t/tau-sweep-20260928/`). τ 0.75 stays optimal; Kaggle τ 1 matches the local bf16 τ 1
score within 0.00006.

| τ | 3 × B0 | Δ vs 0.75 (80%) | 3 × ReXNet | Δ vs 0.75 (80%) |
|---:|---:|---|---:|---|
| 0.6 | 0.905664 | +0.0002 [−0.0003, +0.0007] | 0.913494 | +0.0000 [−0.0005, +0.0006] |
| 0.75 | 0.905460 | — | 0.913466 | — |
| 0.9 | 0.904725 | −0.0007 [−0.0012, −0.0003] | 0.912713 | −0.0008 [−0.0012, −0.0003] |
| 1.0 | 0.903976 | −0.0015 [−0.0022, −0.0008] | 0.911729 | −0.0017 [−0.0025, −0.0010] |
