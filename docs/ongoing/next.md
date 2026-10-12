# Next

Pending work only. The recipe, control and rules in force are in the [project state](state.md);
finished runs and decisions live in the [validation results](../validation/results/INDEX.md), the
[history](../history/INDEX.md) and the canonical topic documents; this file is not a log.

**No campaign is active.** Starting one is a separate user action; historical queues and decisions
live under [autoresearch](../../autoresearch/campaigns/INDEX.md). New candidates pair with the
current control at its fold and seed.

**Closed on 2026-10-10: the
[student-distillation campaign](../../autoresearch/campaigns/student-distillation/INDEX.md)**, in
the worktree `../knee-abnormality-student-distillation`; its training code is on main, its
per-campaign adoption rule and the view stacker are not. Recipe found on the split
`gold_holdout_5fold_v1`: 0.75 × silver + 0.25 × the raw joint teacher prediction (fold-0 ×4
0.935809). Not done: applying it to the Refined ResNet-50 on the five folds of
`all_5fold_scanner_v2`, the share below 0.25 and the share-0 control on this split
([run notes](../../autoresearch/campaigns/student-distillation/run-notes.md)).

Pending on the recipe in force (meniscus map and posed canvases, [state](state.md#recipe-in-force)):

- **ResNet-50 on the recipe in force (2026-10-10):** `configs/refined-resnet50.yaml` carries
  the meniscus map and the posed canvases and has no run yet; the best submission (S126, 0.959)
  is the ResNet-50 on the anatomy-map recipe.
- **Open after the structure-routing line (2026-10-09):** the best routing from the masks
  (`no-side`) is −0.0003 at ×4 on centred crops and −0.0006 on the recipe's crops, with MCL
  alone carrying more than the gap; the multiplier curve (×1 to ×4) of the recipe against one
  experiment is not measured. Detail:
  [view-poses run notes](../../autoresearch/campaigns/view-poses/run-notes.md).
- **Left open by the view-size runs (2026-10-10):** both sagittal views at 84 mm was prepared
  and not run (one line: sagittal T1/PD 93.33 to 84); the first look's tokens are computed and
  discarded and could be pooled; a second view on one series type trains 1.47 times slower at
  the same image count, cause not found (a step timing of both configs is the next step).
  Detail: [view-poses run notes](../../autoresearch/campaigns/view-poses/run-notes.md).
- **Soft-tissue masks line, started 2026-10-10:** worktree `knee-abnormality-soft-tissue-masks`;
  first idea, reading the meniscus and OA labels from third-stage features under the predicted
  masks. Working list in `SOFT_TISSUE_MASK_IDEAS.md` at that worktree's root, uncommitted.
- **Submission ledger:** S122 to S133 are not in it yet.
- **Submit job:** it cannot pick up a notebook version that has already completed; S126 had to
  be submitted by hand through the API.

Pending from the [fixes-and-removals campaign](../validation/results/2026-10-05-fixes-and-removals-campaign.md)
(closed 2026-10-05, on main since 2026-10-06):

- **Control: the campaign incumbent** `slab-rule-and-positive-std-requeue` (user decision
  2026-10-06, no retraining): ×4 0.924049, fold 1, seed 1110484336
  ([state](state.md#current-control)). Its checkpoint and those of S108 and S111 are the only
  ones of the campaign still kept.
- **Check the OA detector's removal before a submission relies on it:** one run with the
  detector restored on the recipe in force, read on the three OA labels, and on the ResNet-50
  ([open items](../validation/results/2026-10-05-fixes-and-removals-campaign.md#open)).

Pending from the [Refined-RadImageNet campaign](../validation/results/2026-10-04-refined-radimagenet-campaign.md)
(closed 2026-10-04; the checkpoint normalization is on main since 2026-10-05):

- **Control until 2026-10-06: the campaign incumbent** (user decision 2026-10-05; it was the adopted recipe,
  `configs/training.yaml` with `data.mean` 0.3162 and `data.std` 0.3213, so no retraining):
  `outputs/autoresearch/refined-radimagenet/experiments/radimagenet-input-statistics-20261004T103433Z-ff2f3786/training`,
  ×4 0.933058, ×1 0.929948. New candidates pair with it. Its checkpoint is the only one of the
  campaign still kept.
- **Decide on the mask-trained (heated) ResNet-50 weights.** Two ResNet-50 runs were started
  and stopped by the user before completion; no result
  ([run notes](../../autoresearch/campaigns/refined-radimagenet/run-notes.md)).
- **Nothing pending from the
  [pretraining-input campaign](../validation/results/2026-10-05-pretraining-input-campaign.md)**
  (closed 2026-10-05): its two ResNet-10t acceptances did not carry over to the ResNet-18 and
  are not adopted. Its build is not on main; the worktree was removed on 2026-10-05 and a full copy is frozen in
  `outputs/autoresearch/pretraining-input/baseline/source/`. The heated
  model as a second member or teacher is in the
  [endgame backlog](../backlog/endgame.md#ensembling).

Awaiting a public check:

- **[Out-of-fold self-distillation](../backlog/endgame.md#out-of-fold-self-distillation) at
  0.5/0.5** (user request 2026-09-30). Teachers: the five `rn18-l2sp-no-confusion` folds
  (`outputs/training/rn18-l2sp-no-confusion` for fold 0, `…-folds-1-4` for folds 1–4; ×4 macro
  0.928062, 0.919191, 0.929444, 0.919376, 0.922596). The `oof-blend` worktree repoints
  `prepare_image_blend_labels` to build `oof_blend_v1` (half `rebuild_state_v2`, half the ×4
  out-of-fold predictions); gold agreement 0.9397 against 0.9075 for the base (80% [+0.022,
  +0.043]), teacher alone 0.9271. The fold-0 child `outputs/training/rn18-oof-blend` (control's
  seed) scored ×4 0.929536 against 0.928062: +0.0015, 95% paired report-group [+0.0001, +0.0031],
  epoch 9 (`outputs/analysis/r/rn18-oof-blend-vs-rn18-l2sp-no-confusion`). The fold-1–4 teachers
  trained on fold 0, so that is an upper bound. Public: S87 0.944 at ×4 τ 0.9 (bundle `56787f47`)
  against 0.943 for S86, one rounding step; pending a user decision.

Pending:

- **Follow-ups of the [two-family out-of-fold distillation](../validation/results/2026-10-04-two-family-oof-distillation.md)**
  (S106, the five ResNet-50 children alone, 0.957 public, the best; S103 with the ReXNets 0.955;
  S104 with type share 0.951; S105 with τ 1.1 for effusion and synovitis 0.954). Open, in the backlog: the sharp side of the temperature and type share with
  the tuned temperature ([small changes](../backlog/small-changes.md#pooling-dense-bags-at-inference)),
  the pooling head refit and the per-plane teacher readings
  ([big bets](../backlog/big-bets.md#pooling-head-at-the-inference-bag-size)), a second
  distillation round and the teacher share of the labels
  ([endgame](../backlog/endgame.md#out-of-fold-self-distillation)).

- **ConvNeXt as a third blend family** (started 2026-10-03, worktree
  `../knee-abnormality-convnext` off main, uncommitted): DINOv3 and supervised ImageNet
  ConvNeXt-Tiny and ConvNeXt-Small on the ReXNet-style recipe, fold 0, one run each against the
  ResNet-18 control (`rn18-view-label-bias`, ×4 0.932477). The rules allow pretrained models
  under another licence (section 2.5.a) when reasonably accessible (2.6.b). Paused 2026-10-04:
  models stay on the Refined-RadImageNet weights; the worktree's notes hold the four runs (best
  ×4 0.927235, no blend gain).
- **Read the slice-budget sensitivity run on Kaggle** when it finishes: per-label AUC with each
  series type's slices halved and doubled, on the S93 model
  (`outputs/analysis/s/slice-budget-20261002/sensitivity/`).

- **Decide on the `inference-opt` branch.** It runs the Kaggle command in about 270 s against
  429.5 s with bit-identical prepared images, but its GPU preparation changes hashed sources, so
  the next submission needs a checkpoint trained on it; check local training memory first
  ([status](loading-speedup.md)).
- **Re-export the slice-importance map with left knees mirrored**, a user action outside this
  repository: the current export centres and scales patient L but never mirrors by knee side, so
  the compartment lobes on the sagittal axis cannot be told apart from two superimposed mirrored
  knees ([spatial importance map](../data-preparation/peripheral-coverage.md#spatial-importance-map)).
  It gates the [side-conditioned sagittal sampling centre](../backlog/big-bets.md#position-and-knee-side).
- **Re-measure the B0 gap under class-balanced BCE** before pricing the next efficiency recipe:
  the [backbone and fold comparisons](efficientnet-backbones.md#established-evidence) used ordinary
  BCE.
