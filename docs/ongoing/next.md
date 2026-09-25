# Next

Pending work only. Finished runs and decisions live in the
[project state](state.md), the [validation results](../validation/results/INDEX.md) and the
canonical topic documents; this file is not a log.

**The position/knee-side experiment sequence is closed.** Learned family strength is
adopted; appearance-only pooling, learned widths and the pooled compartment adapter
are removed. There is no pending confirmation run or adapter-removal task. See
[adoption evidence](../validation/results/2026-09-21-anatomical-attention-result.md).

**No training is queued for automatic launch.** No campaign is active: the three oblique
campaigns are retired with the [oblique line](../history/oblique-groups.md), closed 2026-09-23.
Historical queues and decisions live under [autoresearch](../../autoresearch/campaigns/INDEX.md).
The general [overnight queue](../backlog/overnight.md) is empty. New candidates use
the incumbent's seed and the [shared evaluation and adoption rule](../../autoresearch/protocol.md#shared-evaluation-and-adoption).
Starting a campaign is a separate user action.

Pending:

- **OA-detector control on fold 1 is the incumbent** (`configs/training.yaml`, seed 1485198267,
  ×4 τ 1 0.900568, S60 0.938). Token dropout at 5% was rejected against it
  ([follow-up](../validation/results/2026-09-25-near-misses-campaign.md#follow-up-on-main)).
- **After the knee-geometry campaign (stopped 2026-09-24).** The control is
  `knee-geometry-merged-2` (seed 638120755, ×4 0.900955, S55 0.940). Untried there: the
  curve-density sampler, the side-normalized 3D encoding replacing 1D/2D, and the fallback-safe OA
  readout ([campaign ideas](../../autoresearch/campaigns/knee-geometry/INDEX.md));
  `meniscus-core-band-contrast-readout` trained but was never compared.
- **Noisy-labels campaign stopped 2026-09-23, nothing adopted.** Hard fold-0 AUC against
  `k05-clean` (S48, 0.942): bootstrapped targets −0.0005, running-average targets +0.0002,
  early-learning regularization +0.0013 (soft +0.0008), confident-contradiction down-weighting
  −0.0060, class-balance schedule 1→0 −0.0000 (soft +0.0007), schedule plus early learning −0.0003.
  Prediction-memory methods barely act: the model does not memorize its labels in ten fine-tuning
  epochs. The schedule scored 0.941 public (S53) against 0.942 for S48. Early-learning regularization was
  exported from its experiment source (×4, τ 0.75, batch 4, bundle `1d22fc40`,
  `outputs/submission/exports/elr-20260924-x4-tau075`) and submitted as S57: 0.943 public against
  0.942 for S48. Porting it to the current recipe is
  [under consideration](early-learning-regularization.md)
  ([campaign](../../autoresearch/campaigns/noisy-labels/INDEX.md)).
- **Combined labels scored 0.940 public (S40)** against 0.933 for the last single models on
  Steven v4 labels ([result](../validation/results/2026-09-21-combined-labels-first-run.md)). A
  Steven-v4-trained control on the same split (`labels.source: steven_v4`) would separate the
  label change from the new training fold. Before the next queue, move the overnight launcher
  to `all_5fold_scanner_v1`; the autoresearch gate already uses it.
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
