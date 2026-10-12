# Next

Pending work only. The recipe, control and rules in force are in the [project state](state.md);
finished runs and decisions live in the [validation results](../validation/results/INDEX.md), the
[history](../history/INDEX.md) and the canonical topic documents; this file is not a log.

**No campaign is active.** Starting one is a separate user action; historical queues and decisions
live under [autoresearch](../../autoresearch/campaigns/INDEX.md). New candidates pair with the
current control at its fold and seed.

## Decisions waiting for the user

- **Per-label temperature option:** merge or drop the worktree
  `knee-abnormality-label-temperature` (branch `label-attention-temperature`, uncommitted;
  `export --label-attention-temperature`). The values do not carry between models, so it is
  useful only with values fitted per model
  ([result](../validation/results/2026-10-12-dense-bag-pooling-on-the-resnet-50.md#a-temperature-per-label)).
- **S147's reference:** the same fold-0 weights at ×4, TTA and τ 0.9 were not submitted, so
  S147's 0.959 cannot be attributed to its per-label values.
- **The mask-trained (heated) ResNet-50 weights:** two runs were started and stopped before
  completion; no result
  ([run notes](../../autoresearch/campaigns/refined-radimagenet/run-notes.md)). As a second
  member or teacher it is in the [endgame backlog](../backlog/endgame.md#ensembling).
- **Re-export the slice-importance map with left knees mirrored**, a user action outside this
  repository: the current export centres and scales patient L but never mirrors by knee side, so
  the compartment lobes on the sagittal axis cannot be told apart from two superimposed mirrored
  knees ([spatial importance map](../data-preparation/peripheral-coverage.md#spatial-importance-map)).
  It gates the [side-conditioned sagittal sampling centre](../backlog/big-bets.md#position-and-knee-side).

## Open on the recipe in force

- **ResNet-50 folds:** `rn50-aligned-mixing-folds-0-2` holds folds 0, 1 and 2 of
  `configs/refined-resnet50.yaml` as on main; folds 3 and 4 are not trained
  ([result](../validation/results/2026-10-12-dense-bag-pooling-on-the-resnet-50.md#models)).
- **Teacher share on the ResNet-50:** the
  [student-distillation](../../autoresearch/campaigns/student-distillation/INDEX.md) recipe
  (0.75 silver + 0.25 raw joint teacher) is measured on one ResNet-50 fold (S143, 0.960); not
  done: the five folds of `all_5fold_scanner_v2`, a share below 0.25 and the share-0 control on
  `gold_holdout_5fold_v1`
  ([run notes](../../autoresearch/campaigns/student-distillation/run-notes.md)).
- **Dense-bag pooling:** each series at its ×1 share as an inference option, and a sharpness per
  label fitted per model ([small changes](../backlog/small-changes.md#pooling-dense-bags-at-inference),
  [big bets](../backlog/big-bets.md#pooling-head-at-the-inference-bag-size)).
- **Check the OA detector's removal before a submission relies on it:** one run with the
  detector restored on the recipe in force, read on the three OA labels, and on the ResNet-50
  ([open items](../validation/results/2026-10-05-fixes-and-removals-campaign.md#open)).
- **After the structure-routing line (2026-10-09):** the best routing from the masks (`no-side`)
  is −0.0003 at ×4 on centred crops and −0.0006 on the recipe's crops, with MCL alone carrying
  more than the gap; the multiplier curve (×1 to ×4) of the recipe against one experiment is not
  measured ([view-poses run notes](../../autoresearch/campaigns/view-poses/run-notes.md)).
- **Left by the view-size runs (2026-10-10):** both sagittal views at 84 mm was prepared and not
  run (one line: sagittal T1/PD 93.33 to 84); the first look's tokens are computed and discarded
  and could be pooled; a second view on one series type trains 1.47 times slower at the same
  image count, cause not found (a step timing of both configs is the next step)
  ([view-poses run notes](../../autoresearch/campaigns/view-poses/run-notes.md)).
- **Follow-ups of the [two-family out-of-fold distillation](../validation/results/2026-10-04-two-family-oof-distillation.md):**
  the per-plane teacher readings ([big bets](../backlog/big-bets.md#per-plane-teacher-readings)),
  a second distillation round, the teacher share of the labels and aligning the fold teachers per
  label ([endgame](../backlog/endgame.md#out-of-fold-self-distillation)).

## Tooling

- **Submission ledger:** S122 to S133 and S147 are not in it; the update of 2026-10-12 stopped on
  Kaggle's rate limit (429).
- **Submit job:** it cannot pick up a notebook version that has already completed (S126 was
  submitted by hand through the API), and its description shows `tau1` for a bundle with
  per-label temperatures.
- **Temperature sweep script:** `scripts/adhoc/sweep_child_temperatures.py` reuses the first τ's
  position prior for every later τ
  ([pitfall](../validation/results/2026-10-12-dense-bag-pooling-on-the-resnet-50.md#method-a-token-cache)).
- **Loading speedup:** the central-window resampling on the GPU and the changes that alter the
  prepared images are open ([status](loading-speedup.md)).
- **Slice-budget sensitivity run on Kaggle (2026-10-02):** per-label AUC with each series type's
  slices halved and doubled, on the S93 model; its output was never read and the local folder
  `outputs/analysis/s/slice-budget-20261002/` no longer exists.
