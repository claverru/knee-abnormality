# 2026-10-04 two-family out-of-fold distillation

**Ten children trained on half silver labels and half teacher predictions beat their teachers on
every fold, and their equal blend scores 0.938887 pooled out of fold at ×6 with reflection TTA.**
**S103 scores 0.955 public** (S106, its five ResNet-50 alone, scores 0.957, the best) against 0.954 for the ResNet-50 teacher of fold 0
alone (S102) and for the 28-network blend (S97); one submission, rounded to three decimals.
The teachers are the five Refined ResNet-50 and five ReXNet-200 fold models of the 2026-10-03
recipes; the children are the same two recipes retrained on `oof_blend_v2`. Submitted as one
ten-member bundle (S103).

All scores are macro AUC on `all_5fold_scanner_v2` against the `rebuild_state_v2` targets. Child
validation is optimistic: each fold's labels include predictions of teachers that trained on that
fold. Each child is one run with its own seed, so child against teacher is not seed-matched.

## Labels

`oof_blend_v2` = 0.5 `rebuild_state_v2` + 0.25 ResNet-50 + 0.25 ReXNet-200 out-of-fold ×4
predictions (τ 1, no TTA), each study from the model that held it out. Built by
`scripts/tools/prepare_oof_blend_labels.py`
(`data/labels/prepared/oof-blend-v2/`, manifest
`configs/labels/oof-blend-v2-label-sources.json`). The children's saved configs still name the
manifest at its original path in the removed `oof-distill` worktree, so pass the manifest
explicitly when loading them.

| Table | Macro AUC on the 58 gold studies |
|---|---:|
| `rebuild_state_v2` | 0.9075 |
| ResNet-50 teachers | 0.9354 |
| ReXNet-200 teachers | 0.9169 |
| `oof_blend_v2` | 0.9358 |

The blend is +0.028 over the silver labels (80% [+0.018, +0.039], 2,000 draws).

## Teachers and children

×4, τ 1, no TTA. Teachers used seed 1533697213; children trained from main at `3fee190` with
`--paths.label-manifest` and `--labels.source oof_blend_v2`.

| Fold | ResNet-50 teacher | Child | Δ | Seed | ReXNet-200 teacher | Child | Δ | Seed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.937129 | 0.939182 | +0.0021 | 695381358 | 0.932523 | 0.938499 | +0.0060 | 29643675 |
| 1 | 0.929567 | 0.932914 | +0.0033 | 31342971 | 0.924694 | 0.933932 | +0.0092 | 1222880462 |
| 2 | 0.935441 | 0.938608 | +0.0032 | 1896751054 | 0.928668 | 0.936357 | +0.0077 | 1194693884 |
| 3 | 0.931062 | 0.932813 | +0.0018 | 1492259725 | 0.927017 | 0.934156 | +0.0071 | 2001671516 |
| 4 | 0.927016 | 0.932248 | +0.0052 | 522575569 | 0.921977 | 0.931470 | +0.0095 | 776324244 |
| Pooled | 0.931720 | 0.935119 | +0.0034 | | 0.925785 | 0.934825 | +0.0090 | |

- Runs: `outputs/training/20261003-224423-rexnet-200-views-label-bias-folds-1-4` (teachers),
  `outputs/training/*-rn50-oof-blend-v2-f0..4` and `*-rexnet-200-oof-blend-v2-f0..4` (children).
- Equal blend of the two families, pooled: teachers 0.933913, children 0.937700 (+0.0038).
- **The weaker family gains more and the two end level.** ReXNet-200 was 0.006 behind the
  ResNet-50 as a teacher and is 0.0003 behind as a child.
- Per label, child blend against teacher blend: every label improves. Largest: MCL +0.010,
  contusion +0.007, synovitis +0.006, lateral OA and lateral meniscus +0.004; smallest: PF OA
  +0.001, effusion +0.001.

## Attention temperature, multiplier and weights

Pooled out of fold with reflection TTA, each family alone
(`scripts/adhoc/sweep_child_temperatures.py`, `scripts/adhoc/score_child_blend.py`,
`outputs/analysis/oof-distill-20261004/`):

| Family | × | τ 0.7 | τ 0.8 | τ 0.9 | τ 1.0 | τ 1.1 |
|---|---|---:|---:|---:|---:|---:|
| ResNet-50 | 4 | 0.935583 | **0.935832** | 0.935712 | 0.935278 | 0.934608 |
| ResNet-50 | 5 | 0.935887 | **0.936102** | 0.935922 | 0.935455 | 0.934748 |
| ResNet-50 | 6 | 0.936062 | **0.936272** | 0.936060 | 0.935548 | 0.934786 |
| ReXNet-200 | 4 | 0.934498 | 0.935232 | **0.935411** | 0.935135 | 0.934526 |
| ReXNet-200 | 5 | 0.934560 | 0.935262 | **0.935391** | 0.935065 | 0.934370 |
| ReXNet-200 | 6 | 0.934476 | 0.935181 | **0.935298** | 0.934925 | 0.934208 |

- **Alone, the ResNet-50 wants τ 0.8 and ReXNet-200 τ 0.9 at every multiplier.** Per fold the
  ResNet-50 optimum ranges from 0.7 to 0.9 and does not move with the multiplier.
- **More images help the ResNet-50 (+0.0004 from ×4 to ×6) and not ReXNet-200 (−0.0001).**
- TTA adds about +0.0002 to a family alone and +0.00003 to the blend.

Equal blend at ×6 with TTA over the τ pairs (rows ResNet-50, columns ReXNet-200):

| τ | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 |
|---|---:|---:|---:|---:|---:|
| 0.7 | 0.938377 | 0.938709 | 0.938845 | **0.938887** | 0.938876 |
| 0.8 | 0.938301 | 0.938526 | 0.938608 | 0.938575 | 0.938487 |
| 0.9 | 0.938144 | 0.938280 | 0.938269 | 0.938161 | 0.938011 |
| 1.0 | 0.937952 | 0.938028 | 0.937936 | 0.937765 | 0.937521 |
| 1.1 | 0.937769 | 0.937777 | 0.937625 | 0.937345 | 0.937008 |

- **The blend's best pair is not the per-family pair.** τ 0.7 for the ResNet-50 with τ 1.0 for
  ReXNet-200 gives 0.938887 against 0.938608 for 0.8 with 0.9: +0.00028 (80% [+0.00017,
  +0.00039], positive on all five folds; PF OA +0.002, lateral OA and medial OA +0.001). The
  ResNet-50 value sits on the edge of the grid, so 0.6 is untested.
- **Weights: 50/50.** At the per-family pair the family weight peaks at 0.5 (0.45: −0.00006,
  0.55: −0.00002, 0.4 and 0.6: −0.0001 to −0.0002); the joint grid picks 0.5 at ×6 and 0.45 at
  ×4 and ×5.
- **Multiplier: ×6.** Joint-grid best 0.938677 at ×4, 0.938734 at ×5, 0.938887 at ×6. At the
  per-family pair ×6 is +0.00017 over ×4 (80% [+0.0000, +0.0003]) and ×5 +0.00006 (80%
  [−0.0001, +0.0002]).
- Tuning against τ 1 for both at equal weights is worth +0.0008 at ×6 (per-family pair, 80%
  [+0.0006, +0.0010]) and +0.0011 for the joint pair.

## Submission

S103: the ten children in one bundle
(`outputs/submission/exports/oof-blend-v2-5rn50-5rexnet-20261004-x6-tta`, manifest `16096ca3`),
one data pass, ×6, reflection TTA, τ 0.7 for the five ResNet-50 and τ 1.0 for the five
ReXNet-200, equal weights. None of the ten checkpoints appears in an earlier bundle. Submitted
2026-10-04 12:27 CEST from `claverru/s103-oof-blend-v2-ten-children-x6-b`, version 1. **Public 0.955**,
scored in about 42 minutes.

- On Kaggle's two T4s the public run takes 55 s of inference in FP16 with 3.8 GB peak per GPU
  (FP32: 40 s, up to 10.8 GB). Its predictions equal the local FP16 run within 0.00003.
- **Blending multipliers adds nothing.** The mean of the ×4, ×5 and ×6 predictions against ×6
  alone: ResNet-50 −0.00005 (80% [−0.00015, +0.00005]), ReXNet-200 +0.00017 (80% [+0.00008,
  +0.00025], but +0.00005 over its own ×4), the ten-model blend −0.00004 (80% [−0.00013,
  +0.00005]). Predictions at ×4 and ×6 correlate at 0.999. Multipliers below ×4 are untested.

## ResNet-50 children alone (S106)

**0.957 public, the project's best, with the five ResNet-50 children and no ReXNet** (×6,
reflection TTA, τ 0.8, their best standalone setting; bundle `c8168ad7`), against 0.955 for the
ten children (S103) and 0.954 for the fold-0 ResNet-50 teacher alone (S102).

- **The public set contradicts the local pairing gain.** Pooled out of fold the ReXNet children
  add +0.0026 to the ResNet-50 children (0.938887 against 0.936272 at each one's settings); on
  the public set removing them gains 0.002. One submission each, and the ResNet-50 temperature
  differs (0.8 here, 0.7 in S103), so part of the step may be the temperature.
- **Distillation and five folds are worth +0.003 public for the ResNet-50** (S106 against S102).
- The ReXNet children were scored only inside S103; they have no public score alone. The ReXNet
  recipe differs from the ResNet-50's in its pretraining (ImageNet, not RadImageNet) and has no
  L2-SP anchoring, the two changes adopted for unseen sites; whether that explains the public
  result is untested.

## ResNet-18 beside the ResNet-50 (S107)

**0.954 public, the same as the ResNet-50 alone (S102):** the fold-0 teachers of S102 (Refined
ResNet-50) and S101 (ResNet-18 control, 0.951 alone) in one bundle at their submitted settings
(×4, τ 0.9, no TTA, equal weights; bundle `fc5dd0ab`). The notebook output equals the mean of the
two single-model outputs within 5e-8.

- Neutral on the public set as on fold 0, where the equal blend scores 0.936891 against 0.937129
  for the ResNet-50 (−0.0002, 80% [−0.0012, +0.0007]); the two models' logits correlate at 0.972.
- One submission, rounded to three decimals, undistilled models of one fold. Whether a distilled
  ResNet-18 child adds beside the five ResNet-50 children is a separate, untested question.

## Type share in place of the temperatures (S104)

**0.951 public against 0.955 for S103**, the same ten checkpoints at ×6 with reflection TTA and
equal weights, with τ 1 for every model and type-share pooling (user request). One submission,
and it changes two things at once: the tuned temperatures are dropped (0.7 for the ResNet-50,
1.0 for ReXNet-200) and type share is added, so the public result does not say which one cost
the 0.004. No local score: the out-of-fold pass needs the GPU, which was in use.

- **Implementation** (branch `pooling-options`, not merged): `--type-share` on export writes
  `runtime.type_share`; `inference/engine.py` counts each study's triplets per view, takes the ×1
  reference as `min(study_budgets at ×1, count)` and adds `log(reference / count)` to the
  attention scores through a forward hook, so no hashed source changes and the checkpoints load.
  It differs from the `better-than-tau` token cache, which replays the sampler's realised ×1
  selection.
- **Checks:** identical to plain pooling at ×1 on three validation studies (difference 0.0); at
  ×6 the offsets equal the definition on a study with two saturated views, one with none and one
  with two series types missing; predictions move by 0.003 on average and 0.026 at most. On the
  three public studies S104 sits 0.019 from S103 on average (0.053 at most).
- **Local reference for the temperature part alone:** τ 1 for both families at ×6 with TTA is
  0.0011 below the tuned pair pooled out of fold.
- **Open:** the four-way out-of-fold comparison (tuned τ or τ 1, with and without type share) at
  ×6, and type share at bags larger than ×6, where the ResNet-18 gain appeared.

## A temperature from the bag size, without tuning

**τ = ln N_train / ln N_test recovers most of the tuned gain with nothing fitted.** The rule keeps
the attention entropy constant as the softmax grows (Chiang and Cholak 2022; Nakanishi 2025,
Scalable-Softmax), with N in tokens (49 per triplet). For about 30 realised triplets at ×1 and
115, 140 and 160 at ×4, ×5 and ×6 it gives τ 0.844, 0.826 and 0.813, the same for both families.
Scored on the sweep above with TTA, the rule's predictions interpolated in logit space between
the two neighbouring grid values (an interpolation across twice that gap overstates the AUC by
0.0001, so read the rule's figures as at most that optimistic):

| × | Rule τ | Blend, rule | Blend, τ 1 | Blend, per-family τ | Blend, joint τ | Rule minus τ 1 (80%) | Rule minus per-family (80%) | Rule minus joint (80%) |
|---|---:|---:|---:|---:|---:|---|---|---|
| 4 | 0.844 | 0.938329 | 0.937743 | 0.938434 | 0.938665 | +0.0006 [+0.0004, +0.0008] | −0.0001 [−0.0002, −0.0001] | −0.0003 [−0.0005, −0.0002] |
| 5 | 0.826 | 0.938412 | 0.937764 | 0.938490 | 0.938730 | +0.0007 [+0.0004, +0.0009] | −0.0001 [−0.0001, −0.0000] | −0.0003 [−0.0005, −0.0002] |
| 6 | 0.813 | 0.938525 | 0.937765 | 0.938608 | 0.938887 | +0.0008 [+0.0005, +0.0010] | −0.0001 [−0.0002, −0.0000] | −0.0004 [−0.0005, −0.0002] |

- Alone at ×6: ResNet-50 0.936279 at the rule against 0.936272 at its tuned 0.8; ReXNet-200
  0.935224 against 0.935298 at its tuned 0.9 (−0.0001), since the rule gives both the same value.
- The rule takes about nine tenths of what tuning one τ per family gives over τ 1, and about two
  thirds of what the joint pair gives. What it misses is the difference between the families, not
  the multiplier.
- With the nominal 26 triplets as the reference the values are 0.838, 0.816 and 0.800 and the
  blend scores move by less than 0.00001.

## A temperature per label

**Effusion and synovitis want a softer temperature than the other ten labels; a free τ per label
is not stable.** From the same sweep with TTA, the pair (ResNet-50 τ, ReXNet-200 τ) chosen per
label on four folds and applied to the fifth, against the global pair chosen the same way:

| × | Global pair, cross-fitted | Per-label pair, cross-fitted | Δ (80%) | Per-label, in-sample |
|---|---:|---:|---|---:|
| 4 | 0.938604 | 0.938507 | −0.0001 [−0.0003, +0.0001] | 0.939223 |
| 6 | 0.938755 | 0.939163 | +0.0004 [+0.0002, +0.0006] | 0.939528 |

- The sign changes between ×4 and ×6, and the in-sample figure is 0.0004 to 0.0007 above the
  cross-fitted one: 24 values chosen on one metric overfit. Baker's (−0.0026 at ×4) and fracture
  (−0.0012, −0.0008) lose when their τ is chosen per fold.
- **The stable part:** effusion picks (1.1, 1.1) in every fit at ×6 and synovitis (1.0 or 1.1,
  1.1); ACL, MCL and the menisci pick 0.7 or 0.8 for both families. With τ 1.1 on both families
  for effusion and synovitis only and the global pair (0.7, 1.0) elsewhere, the blend gains
  +0.00026 at ×4, +0.00029 at ×5 and +0.00032 at ×6 (80% [+0.0002, +0.0004] at each; four of five
  folds positive, fold 0 flat). Effusion +0.0017 to +0.0020, synovitis +0.0010 to +0.0017. The two
  labels were picked after looking at this table, so the gain is somewhat optimistic, and 1.1 and
  0.7 are the ends of the grid.

- **Submitted as S105: 0.954 public against 0.955 for S103** (same ten checkpoints, ×6, TTA and
  pair; only effusion and synovitis at τ 1.1; bundle `33719c5b`). On the three public studies the
  other ten labels equal S103's output exactly and the two labels move by 0.016 on average. One
  rounding step down where the pooled out-of-fold figure said +0.0003: the step can be anything
  from just under 0 to −0.002, so it does not confirm the local gain and does not show a large
  loss. The per-target temperature is `export --label-attention-temperature`, baked into each
  member's scorer row and view label bias (branch `pooling-options`, not merged).
- **Both softer settings scored lower on the public set than S103** (S104 with τ 1 and type share
  0.951, S105 0.954), as τ 0.75 beat τ 1 there by more than locally in September. Whether the
  test studies want a sharper temperature than the training folds do is untested: the sharp end
  of the grid (0.7) is where the ResNet-50 and the ligament and meniscus labels already sit.

## Sweep pitfalls

- **Mixed-precision weight cache.** `torch.autocast` caches the cast copy of a weight for the
  whole context, so changing τ in place inside one context keeps the first τ. The sweep pools
  each τ in its own context; every τ was checked against a fresh forward (within 0.00003), and
  τ 1 at ×4 without TTA reproduces the saved validation predictions exactly.
- **GPU memory under WSL.** ReXNet-200 at ×5 and ×6 filled the 24 GB at batch 4 and at batch 2,
  slowed and then failed with a CUDA error. Capping the process at 85% of the memory and freeing
  the cache after each batch kept it at about 4 GB and 2 to 4 minutes per pass.
- **Unordered joins.** A Polars join without `maintain_order` scrambled rows on the pooled
  4,407 studies; the scoring script aligns explicitly.
