# 2026-10-02 learned-view and blend submissions

**0.954 public, the project's best, from 28 networks in 13 bundles (S97).** Four submissions on
2026-10-02 after S93: the learned-view ResNet-18 alone (S94) and three blends of bundles already
on Kaggle (S95 to S97). All members are fold models at ×4; none was retrained.

| Submission | Blend | Fold 0 ×4 | Scoring | Public |
|---|---|---:|---:|---:|
| **S97** | 28 networks, 13 bundles, mirror TTA on all, weighted by architecture | none | 92 min | **0.954** |
| S96 | ReXNet-200 + Refined ResNet-50 (both mirror TTA) + ResNet-18 84/93 mm | 0.936926 | 15 min | 0.953 |
| S95 | ResNet-18 84/93 mm + ResNet-18 with learned views | 0.933996 | 9 min | 0.950 |
| S94 | ResNet-18 with learned views, alone | 0.932489 | 5 min | 0.947 |
| S89 (2026-10-01) | ReXNet-200 + Refined ResNet-50, mirror TTA | 0.935999 | | 0.951 |
| S93 (earlier that day) | ResNet-18 84/93 mm, alone | 0.932880 | 5 min | 0.950 |

Fold-0 scores are equal-weight means of the saved ×4 validation predictions at τ 1 without TTA
(912 studies, `all_5fold_scanner_v2`, `rebuild_state_v2` labels); the submitted members keep
their own τ and TTA.

## What the numbers say

- **Another architecture complements; another ResNet-18 does not.** The 84/93 mm ResNet-18 adds
  +0.002 to the S89 pair on the public set (S96) and +0.0009 on fold 0 (80% [+0.0003, +0.0016]).
  The learned-view ResNet-18 added to the 84/93 mm one gains +0.0011 on fold 0 (80% [+0.0003,
  +0.0019]) and nothing on the public set (S95 0.950 against 0.950 for S93). One submission
  each; the public score is rounded to three decimals.
- **The learned-view ResNet-18 trails the hand views on the public set** (S94 0.947 against
  0.950) after tying them on fold 0 at ×4 (0.932489 against 0.932880). Its ×1 validation was
  0.002 lower, and the public score followed that. The Kaggle prediction path is not the cause:
  the same checkpoint predicted through the worker path in FP16 scores 0.932472 on fold 0. The
  model comes from the `view-optimisation` branch, where it is the reference of the
  learnable-views campaign.
- **More folds and architectures gave one more step** (S97 0.954 against 0.953). Its parts are
  not separated: more ReXNets from folds 3 and 4, four other-fold ResNet-18s, mirror TTA on the
  members that lacked it and 0.10 of older-label models went in at once.
- **Nothing checkable on fold 0 beats the S96 trio.** Each of the 20 other fold-0 models with a
  checkpoint, added as a fourth equal member, gives −0.0001 to −0.0009. Weights 1:2:1 (ReXNet,
  ResNet-50, ResNet-18) give +0.0002 (80% [−0.0001, +0.0005]). Rank averaging loses 0.0006;
  logit, geometric and power means change nothing; per-label weights gain +0.0012 fitted on all
  of fold 0 and lose 0.0010 cross-validated.

## Fold-0 blends

| Blend | Fold 0 | Against ReXNet + ResNet-50 |
|---|---:|---:|
| ReXNet + ResNet-50 + ResNet-18 84/93 mm (S96) | 0.936926 | +0.0009 |
| The same + learned-view ResNet-18 | 0.936721 | +0.0007 |
| ReXNet + ResNet-50 + learned-view ResNet-18 | 0.936498 | +0.0005 |
| The trio + 120 mm ResNet-18 | 0.936348 | +0.0003 |
| ReXNet + ResNet-50 (S89) | 0.935999 | |
| ResNet-50 + ResNet-18 84/93 mm | 0.935938 | −0.0001 |
| ReXNet + ResNet-18 84/93 mm | 0.935095 | −0.0009 |
| ResNet-18 84/93 mm + learned-view ResNet-18 (S95) | 0.933996 | −0.0020 |
| ResNet-18 84/93 mm + 120 mm ResNet-18 | 0.933289 | −0.0027 |

Singles: Refined ResNet-50 0.934423 (`refined-rn50-incumbent-recipe`), ResNet-18 84/93 mm
0.932880 (`fluid-detail-93-replacement`), learned-view ResNet-18 0.932489
(`views-budget-shift-centre`), 120 mm ResNet-18 0.930666 (`rn18-field-120-control`), ReXNet-200
0.928288 (`20260928-150724-rebuild-state-v2-control`).

## S97 members and weights

Each bundle predicts with its own code; the notebook takes a weighted mean of the 13 prediction
files. A bundle with several networks averages them internally.

| Group | Weight | Bundles (networks) |
|---|---:|---|
| ResNet-50 | 0.30 | Refined ResNet-50, fold 0 (1) |
| ResNet-18 | 0.30 | 84/93 mm, fold 0, 0.18 (1); L2-SP 140 mm, folds 1 to 4, one bundle, 0.12 (4) |
| ReXNet-200 | 0.30 | fold 0 v2 control, 0.10 (1); fold 3 confusion head, control and site bias, 0.10 (3); fold 4 rebuild-state, rebuild-mean, control and the three-backbone plane model, 0.10 (6) |
| Extras | 0.10 | EfficientNet-B3 + ReXNet, fold 3, ×5, 0.05 (2); five EfficientNet-B0 + five ReXNet of 2026-09-19, 0.05 (10) |

- **Shares per architecture.** On fold 0 a further ResNet-18 as an equal member cost the trio
  up to 0.0009, while ResNet-18s averaged inside one slot were neutral (0 to +0.0002), so each
  architecture has a fixed share however many models it holds. Other weightings of the trio
  were flat, so the three are equal.
- **Judgement, not fitted:** the shares inside a group and the 0.10 of extras. Most members
  trained on fold 0, so no weight could be fitted and the blend has no local score.
- **Label versions differ.** The fold-3, fold-4 and 2026-09-19 members were trained on earlier
  label sources; on the three public test studies they sit 0.3 to 0.5 from the blend on single
  probabilities, against about 0.16 for the current-label members. A plain mean therefore
  weights them unevenly.
- **Checks before submission:** the blend equals the weighted mean of the 13 files; ResNet-50
  and fold-0 ReXNet reproduce S89 on the public test studies; the 84/93 mm ResNet-18 with TTA
  sits 0.007 from its S93 output on average.

## How the blend notebooks are built

- **No upload for members already on Kaggle.** A notebook attaches the private bundle datasets,
  runs each bundle's `scripts.pipeline.predict` with that bundle's `code` on `PYTHONPATH`, and
  averages the outputs, as S80 and S89 did. The one new bundle is
  `rn18-l2sp-folds-1-4-20261002-x4-tau09-w8-tta`, four checkpoints of
  `outputs/training/rn18-l2sp-no-confusion-folds-1-4` exported with the run's
  `source_snapshot.zip` as the executing source (it matches the S86 bundle's code).
- **Pin the Kaggle environment.** A new notebook without `docker_image` in its kernel metadata
  was given Python 3.13, where the bundled cp312 codec wheels do not install. S96 and S97 ran
  in `gcr.io/kaggle-private-byod/python@sha256:37c64f7d…`.
- **Mirror TTA on a bundle exported without it.** The notebook copies the bundle, sets
  `inference.reflection_tta` in `config.json` and `runtime.reflection_tta` in the manifest, and
  updates the manifest's hash of `config.json`; the bundle then verifies and predicts with TTA.
- **Bundles of 2026-09-26 fail mirror TTA in FP16.** Their `inference/engine.py` calls
  `reflect_batch` inside autocast (`Index put requires the source and destination dtypes
  match`). The notebook wraps that call with autocast off in its working copy and updates the
  file's hash; the engine is outside the checkpoint identity. A CPU test in FP32 does not show
  the fault. Bundles from 2026-09-27 on have the call outside autocast.
- **The Kaggle client pages the submission list at 50 rows,** so a runner reads the newest
  row's label to find the next sequence number and does not count rows.
- **Scripts and records.** The notebook generators and runners are in the session scratchpad
  (`blend/`, `mega/`); the records are `outputs/submission/records/ens-*-20261002-x4.json` and
  `mega-ens-28-nets-20261002-x4-tta.json`, with each notebook's output beside them.

## Checkpoints left on disk

After the 2026-10-02 disk cleanup, 42 of 541 runs with saved predictions keep their checkpoint:
26 of fold 0 (23 on the current labels and split), 3 each of folds 1 and 2, and 5 each of folds
3 and 4. Bundles already uploaded stay usable on Kaggle without the local checkpoint.
