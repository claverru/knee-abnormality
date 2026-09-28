# Next

Pending work only. Finished runs and decisions live in the
[project state](state.md), the [validation results](../validation/results/INDEX.md) and the
canonical topic documents; this file is not a log.

**No campaign is active.** Historical queues and decisions live under
[autoresearch](../../autoresearch/campaigns/INDEX.md). The [overnight queue](../backlog/overnight.md)
holds the ReXNet `trainer.max_epochs: 12` run. New candidates use
the incumbent's seed and the [shared evaluation and adoption rule](../../autoresearch/protocol.md#shared-evaluation-and-adoption).
Starting a campaign is a separate user action.

Pending:

- **`scanner-confusion-control` on fold 4 is the incumbent** (`configs/training.yaml`, seed
  703791404, ×4 0.909733); candidates pair with it ([state](state.md)).
- **Combined labels scored 0.940 public (S40)** against 0.933 for the last single models on
  Steven v4 labels ([result](../validation/results/2026-09-21-combined-labels-first-run.md)). A
  Steven-v4-trained control on the same split (`labels.source: steven_v4`) would separate the
  label change from the new training fold.
- **Decide on the `inference-opt` branch.** It runs the Kaggle command in about 270 s against 429.5 s
  with bit-identical prepared images, but its GPU preparation changes hashed sources, so the next
  submission needs a checkpoint trained on it; check local training memory first
  ([status](loading-speedup.md)).
- **Re-export the slice-importance map with left knees mirrored**, a user action outside this
  repository: the current export centres and scales patient L but never mirrors by knee side, so
  the compartment lobes on the sagittal axis cannot be told apart from two superimposed mirrored
  knees ([spatial importance map](../data-preparation/peripheral-coverage.md#spatial-importance-map)). It gates the
  [side-conditioned sagittal sampling centre](../backlog/big-bets.md#position-and-knee-side).
- **Re-measure the b0 gap under class-balanced BCE** before pricing the next efficiency recipe:
  the [backbone and fold comparisons](efficientnet-backbones.md#established-evidence) used ordinary BCE.
