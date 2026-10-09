# 2026-09-27 EfficientNet-B3 overnight queues

**On the current recipe B3 nearly matches ReXNet-200: the best run, `backbone_lr` 4.5e-4, scores
×4 0.904429 against 0.905205 for `half-stack-control`, −0.0008 (95% [−0.0041, +0.0028]), at the
same training time and 1 GiB less memory.** On 2026-09-18, under ordinary BCE, B3 trailed by 0.009.
A [second round](#second-round-learning-rates-and-schedule) on top of it found one gain, a two-epoch
warmup: ×4 0.905638, +0.0012 over its B3 control and +0.0004 over ReXNet, the first B3 run above
the ReXNet control; every other learning-rate and schedule arm lost.

## First queue

Queue `outputs/training/overnight-20260927T003247Z-kdc9bncm`, fold 3 of `all_5fold_scanner_v1`,
seed 2617320043 for every run (the ReXNet control's), one run each.

Every run changes the recipe in three places: `model.backbone: efficientnet_b3.ra2_in1k`, DSU off
(`model.feature_statistics_augmentation.probability: 0`, since DSU is ReXNet-only), and
`backbone_lr` 3e-4 from the [weight-scale rule](../../modeling/backbone-swaps.md#checkpoint-weight-scale)
(convolution-weight RMS 0.0734 against 0.0508).

| Run | ×1 selected | ×4 macro | Δ ×4 vs B3 control | Δ ×4 vs ReXNet |
|---|---:|---:|---:|---:|
| `02-backbone-lr-0.00045` | 0.899446 | **0.904429** | +0.0005 | −0.0008 |
| `00-control` (3e-4) | 0.898022 | 0.903955 | — | −0.0012 |
| `06-backbone-lr-0.0006` | 0.897924 | 0.903822 | −0.0001 | −0.0014 |
| `05-epochs-20` | 0.899132 | 0.903742 | −0.0002 | −0.0015 |
| `04-both-lr-x2` (6e-4 / 1.2e-3) | **0.899969** | 0.903697 | −0.0003 | −0.0015 |
| `07-head-lr-0.0012` | 0.897494 | 0.903590 | −0.0004 | −0.0016 |
| `03-epochs-15` | 0.896588 | 0.902797 | −0.0012 | −0.0024 |
| `08-contrast-mixing-off` | 0.896459 | 0.902666 | −0.0013 | −0.0025 |
| `09-gin-off` | 0.894290 | 0.902341 | −0.0016 | −0.0029 |
| `01-backbone-lr-0.0002` | 0.895464 | 0.902248 | −0.0017 | −0.0030 |

- **The weight-scale rule points the right way.** The unscaled 2e-4 is the worst run; 3e-4 to 6e-4
  lie within 0.0006 of each other.
- **More epochs do not help at ×4.** 15 epochs −0.0012, 20 epochs −0.0002, unlike B0's
  ordinary-BCE +0.0029.
- **B3 uses the half-stack augmentation.** Removing contrast mixing or GIN costs 0.0013–0.0016.
- **Against ReXNet** (paired report-group bootstrap, 851 groups): the B3 control is −0.0012
  [−0.0045, +0.0022], with meniscus −0.0046 and MCL +0.022; the 4.5e-4 run is −0.0008
  [−0.0041, +0.0028], meniscus −0.0009, OA −0.0041, MCL +0.016
  (`outputs/analysis/b3-*-vs-half-stack-control/comparison.json`). Training 1,937 s against
  1,921 s, peak 18.2 GiB against 19.2 GiB.
- **As an ensemble partner:** averaging the 4.5e-4 run with `half-stack-control` scores ×4
  0.908106, +0.0029 over ReXNet alone (study bootstrap 95% [+0.0012, +0.0048]), with mean
  per-label Spearman 0.94. Both checkpoints were selected on this fold, so the gain is optimistic.
  **The pair at ×5, τ 0.75 with mirror TTA (bundle `5bf80349`) scored 0.946 public against 0.947 for the ReXNet
  control alone at ×4 (S68)**: the local gain did not transfer, within the ±0.006 public noise.
  It also changes ×4 → ×5 and adds TTA, so it does not isolate B3. Its Kaggle description
  reads S70, a number reused from the auxiliary-signals campaign's S70 submitted 16 minutes
  earlier; by submission time it is the 72nd. Its first two attempts
  exposed an FP16 crash in mirror TTA (`reflect_batch` inside autocast), fixed in
  `inference/engine.py` by mirroring before autocast; a Kaggle run stuck in `QUEUED` forced a
  new notebook.

All single-run differences inside the B3 queue are below 0.002 and within seed noise; they rank
the next queue's axes, they do not establish settings.

## Second round: learning rates and schedule

Queue `outputs/training/overnight-20260927T101441Z-_1ey2phz`, same fold and seed, one run each,
against `02-backbone-lr-0.00045` from the first queue (×4 0.904429): B3, DSU off,
`backbone_lr` 4.5e-4. The arms vary the head rate and the schedule (linear warmup from
`warmup_start_ratio`, then cosine decay to `minimum_lr_ratio`); runs and rationale in the
[queue guide](../overnight-hyperparameter-queue.md#efficientnet-b3-second-round-ran-2026-09-27).

| Run | Selected epoch | ×1 selected | ×4 macro | Δ ×4 vs B3 control | Soft ×1 | Training s |
|---|---:|---:|---:|---:|---:|---:|
| **`03-warmup-2`** | 10 | **0.900079** | **0.905638** | **+0.0012** | 0.7966 | 1,959 |
| control `02-backbone-lr-0.00045` | 8 | 0.899446 | 0.904429 | — | 0.7966 | 1,937 |
| `01-head-lr-0.0009` | 10 | 0.898615 | 0.904382 | −0.0000 | 0.7969 | 1,974 |
| `02-head-lr-0.0004` | 10 | 0.899056 | 0.903729 | −0.0007 | 0.7962 | 1,937 |
| `05-minimum-lr-ratio-0.1` | 10 | 0.898194 | 0.903670 | −0.0008 | 0.7967 | 1,944 |
| `07-epochs-20` | 15 | 0.899805 | 0.903498 | −0.0009 | **0.7986** | 3,865 |
| `06-minimum-lr-ratio-0.001` | 10 | 0.897783 | 0.903148 | −0.0013 | 0.7965 | 1,972 |
| `04-warmup-0` | 10 | 0.898272 | 0.902048 | −0.0024 | 0.7961 | 1,960 |

Peak memory was 18.2 GiB for every run.

- **A longer warmup is the only gain.** Two warmup epochs scored +0.0012 against the B3 control,
  95% [−0.0012, +0.0038] (paired report-group bootstrap, 851 groups); no warmup cost 0.0024, so
  the schedule's start matters more than its floor or the head rate.
- **Against ReXNet** (`half-stack-control`, ×4 0.905205): `03-warmup-2` is +0.0004, 95%
  [−0.0029, +0.0040]; OA −0.0030, meniscus +0.0003, the other eight +0.0014, with MCL +0.017
  and Effusion −0.0055 (`outputs/analysis/b/b3-warmup-2-vs-half-stack-control/comparison.json`,
  `outputs/analysis/b/b3-warmup-2-vs-b3-lr-0.00045/comparison.json`).
- **The head rate and the cosine floor are flat or negative** (−0.0000 to −0.0013), and 20 epochs
  selected epoch 15 and lost 0.0009 at ×4 while gaining soft AUC (+0.0020), as in the first queue.
- **Not submitted.** Single runs below 0.002 are within seed noise; the recipe has since moved to
  the scanner confusion head on fold 4 (`scanner-confusion-control`, ×4 0.909733), so a B3 follow-up
  needs a control on that recipe.

