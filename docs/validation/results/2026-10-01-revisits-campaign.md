# 2026-10-01 revisits campaign

**Ran 2026-09-30 22:15 to 2026-10-01 16:57 CEST, stopped by the user.** The `revisits` autoresearch
campaign re-tested, on the RadImageNet ResNet-18 recipe, the ideas whose verdict changed when past
comparisons were re-scored on `rebuild_state_v2`
([re-scoring](2026-09-28-campaign-rescoring.md)), then searched hyperparameters and measured the
backbones the user forced in. 39 scored runs (fold 0 of `all_5fold_scanner_v2`, seed 1533697213,
×4 macro AUC, adoption at Δ ≥ +0.001 with the 80% paired report-group interval above 0):

- **Adopted in the campaign:** plane-contrast readouts, then removal of the anatomical attention
  bias. The campaign incumbent scores ×4 0.929835 against 0.928062 for the reference
  `rn18-l2sp-no-confusion`. Neither is on main yet.
- **Refined-RadImageNet ResNet-50 beats ResNet-18 by +0.0046**, but is a measurement only: the user
  does not adopt a backbone at its training time. Alone it scored 0.948 public (S88). Blended with
  the ReXNet-200 control, with reflection TTA, it scored **0.951 (S89), the best public score**.
- **One backbone per plane ties a shared ResNet-18**, and dense per-plane readouts lose on every
  backbone.
- **Nothing else passed:** the remaining removals, adapters, plane-conditioned BatchNorm, stochastic
  depth, EMA, the report-attribute removal and 18 hyperparameter runs.

Campaign records: [campaign](../../../autoresearch/campaigns/revisits/INDEX.md),
[decision summary](../../../autoresearch/campaigns/revisits/findings.md).

## Re-tested ideas (the nine forced seeds)

Each seed ran against the incumbent at its turn, so later seeds include the adopted ones.

| Idea | ×4 macro | Δ | 80% interval | Verdict |
|---|---:|---:|---|---|
| Dense per-plane readouts | 0.922814 | −0.0052 | [−0.0071, −0.0034] | reject |
| Plane-contrast readouts | 0.928980 | +0.0009 | [+0.0001, +0.0017] | accept |
| No DSU | 0.926727 | −0.0013 | [−0.0027, −0.0001] | reject |
| No anatomical bias | 0.929835 | +0.0009 | [+0.0002, +0.0016] | accept |
| No bilateral smoothing | 0.928954 | −0.0000 | [−0.0011, +0.0011] | reject |
| No resolution degradation | 0.929383 | −0.0005 | [−0.0015, +0.0006] | reject |
| Plane/contrast feature adapters | 0.929442 | −0.0004 | [−0.0010, +0.0002] | reject |
| Plane-conditioned BatchNorm | 0.928828 | −0.0010 | [−0.0018, −0.0002] | reject |
| Stochastic depth | 0.926345 | −0.0035 | [−0.0047, −0.0024] | reject |

Both acceptances fell 0.0001 short of the +0.001 threshold, so the rule rejected them; the user
adopted both in the campaign, plane-contrast readouts on their per-label gains. Each is one run, so the evidence is weak (seed noise on this recipe is about
0.002–0.003). DSU, which lost its local support in the re-scoring, still helps ResNet-18; bilateral
smoothing and resolution degradation are ties to keep.

## Backbones

All on the campaign incumbent's recipe; Δ against the incumbent.

| Backbone | Backbone LR | ×4 macro | Δ | 80% interval | Training s |
|---|---|---:|---:|---|---:|
| Refined ResNet-50 | 5e-5 | **0.934423** | **+0.0046** | [+0.0027, +0.0064] | 1,969 |
| Refined ResNet-50 | 5.5e-5 (weight-RMS rule) | 0.933316 | +0.0035 | [+0.0013, +0.0056] | 1,997 |
| ResNet-18 (incumbent) | 5e-5 | 0.929835 | 0 | | 1,197 |
| Refined ResNet-10t | 6e-5 (weight-RMS rule) | 0.925267 | −0.0046 | [−0.0068, −0.0025] | 1,154 |
| Refined ResNet-10t | 5e-5 | 0.924526 | −0.0053 | [−0.0076, −0.0031] | 1,144 |

- **The Refined-RadImageNet checkpoints rank as their pretraining scores do** (RadImageNet macro
  F1 0.3998, 0.4545 and 0.5097 for ResNet-10t, ResNet-18 and ResNet-50), unlike the original
  RadImageNet ResNet-50, which lost to ReXNet-200
  ([RadImageNet backbones](../../ongoing/radimagenet-backbone.md)). The Refined ResNet-50 uses the
  ResNet-18 settings unchanged (ImageNet mean/std, ten epochs); the user raised the campaign's
  time limit to 1.65× the reference (2,016 s) so it could keep ten epochs.
- **Both ResNet-50 runs passed the adoption rule** and were reverted to the ResNet-18 incumbent by
  manual adoption, on the user's decision that its training time (about 1.65× ResNet-18) is too
  high for the iteration loop.

## One backbone per plane

Three copies of the pretrained backbone, one per plane, each anchored by L2-SP to its pretrained
weights, DSU inside each copy. A keeps the joint pool and plane-contrast readouts; B uses dense
per-plane readouts with learned fusion. Measurements only, by the user's decision.

| Backbones | Readout | ×4 macro | Δ | 80% interval | Training s |
|---|---|---:|---:|---|---:|
| 3 × ResNet-18 | A, joint pool | 0.929880 | +0.0000 | [−0.0011, +0.0011] | 1,301 |
| 3 × ResNet-10t | A, joint pool | 0.927563 | −0.0023 | [−0.0044, −0.0003] | 1,226 |
| 3 × ResNet-18 | B, dense + fusion | 0.924135 | −0.0057 | [−0.0075, −0.0040] | 1,310 |
| 3 × ResNet-10t | B, dense + fusion | 0.920367 | −0.0095 | [−0.0119, −0.0072] | 1,216 |

Three ResNet-18s cost only 9% more training time than one (1,301 s against 1,197 s) but tie it, as three ReXNet-200s tied one
([plane backbones](../../history/plane-backbones.md)). Three ResNet-10ts recover about half of
ResNet-10t's gap to ResNet-18 (+0.0023 over the single ResNet-10t, a cross-run comparison) but stay
below it. Dense readouts lose with or without separate backbones.

## Ensembles (fold 0, ×4, equal weights)

| Blend | ×4 macro | Δ | 80% interval |
|---|---:|---:|---|
| ReXNet-200 + Refined ResNet-50 | **0.935999** | +0.0016 over ResNet-50 | [+0.0003, +0.0029] |
| ReXNet-200 + ResNet-50 + ResNet-18 | 0.935733 | −0.0003 against the pair | [−0.0009, +0.0003] |
| ResNet-50 + ResNet-18 | 0.934421 | −0.0000 over ResNet-50 | |
| ReXNet-200 + 3 × ResNet-18 | 0.933495 | +0.0000 against ReXNet + ResNet-18 | [−0.0006, +0.0006] |
| ReXNet-200 + ResNet-18 | 0.933460 | | |
| ReXNet-200 + 3 × ResNet-10t | 0.932439 | −0.0010 against ReXNet + ResNet-18 | [−0.0021, −0.0000] |

ReXNet-200 and the Refined ResNet-50 complement best (prediction correlation 0.95); ResNet-18 and
its plane-backbone variant correlate at 0.99 and add nothing to that pair. Public: S89, this pair
with reflection TTA, scored 0.951 against 0.948 for the ResNet-50 alone (S88) and 0.949 for ReXNet
+ ResNet-18 at 36 triplets (S80). S89 changed two things at once, the partner and TTA, so the TTA
share is unmeasured. It ran as two bundles in one custom notebook, because the two models were
trained from different code.

## Other runs

- **EMA**, forced by the user as an exception to the weight-averaging exclusion: decay 0.999
  −0.0004, decay 0.9995 −0.0032, with the existing option (BatchNorm statistics averaged with the
  weights). The first two EMA runs were invalid: their seed text asked for a full BatchNorm refresh
  every epoch, which pushed epochs over the time limit; a refresh on a quarter of the studies also
  scored −0.0004.
- **Report-attribute heads removed:** −0.0006 [−0.0011, −0.0001], 10% faster. They stay.
- **Hyperparameter search, 18 runs:** none passed; twelve epochs +0.0002, every other run between
  −0.0013 (backbone LR 2.5e-5) and −0.0001. The user stopped the search as unproductive.

## Taken to main (user, 2026-10-01)

- Plane-contrast readouts and no anatomical bias on `configs/training.yaml`; the ReXNet-200 recipe
  keeps neither change. The ResNet-50 was trained with both, so S88 and S89 need them.
- `configs/refined-resnet50.yaml`: the same recipe on the Refined ResNet-50, the submission model.
- ResNet-18 stays the control: it is 1.65× faster, and what helps it is assumed to carry over to
  ResNet-50. That transfer is unmeasured: there is one ResNet-50 run and no ablation on it.
- The ported code loads both campaign checkpoints strictly, and both configs equal the runs'
  saved configs apart from name and paths. The new control is the campaign incumbent
  `no-anatomical-bias`; exports of it and of the ResNet-50 use their archived `source/`.
- Not taken: plane backbones, every rejected seed and the hyperparameter search. A plane-backbone
  model would also need the Kaggle CUDA-graph backbone path adapted to three
backbones: the current one wraps a single backbone, so the plane model falls back to eager
inference.
