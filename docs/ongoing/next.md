# Next

Pending work only. The recipe, control and rules in force are in the [project state](state.md);
finished runs and decisions live in the [validation results](../validation/results/INDEX.md), the
[history](../history/INDEX.md) and the canonical topic documents; this file is not a log.

**No campaign is active.** Starting one is a separate user action; historical queues and decisions
live under [autoresearch](../../autoresearch/campaigns/INDEX.md). New candidates pair with the
current control at its fold and seed.

Pending:

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
