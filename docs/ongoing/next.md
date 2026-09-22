# Next

Pending work only. Finished runs and decisions live in the
[project state](state.md), the [validation results](../validation/results/INDEX.md) and the
canonical topic documents; this file is not a log.

**The position/knee-side experiment sequence is closed.** Learned family strength is
adopted; appearance-only pooling, learned widths and the pooled compartment adapter
are removed. There is no pending confirmation run or adapter-removal task. See
[adoption evidence](../../autoresearch/anatomical-attention.md).

**No training is queued for automatic launch.** No campaign is running; historical
queues and decisions live under [autoresearch](../../autoresearch/status.md).
The general [overnight queue](../backlog/overnight.md) holds one run, the series-hygiene bundle (exact copies, superseded series, contrast relabel, near-empty frames). New candidates use
the incumbent's seed and the [shared evaluation and adoption rule](../../autoresearch/protocol.md#shared-evaluation-and-adoption).
Starting a campaign is a separate user action.

Pending:

- **Combined labels scored 0.940 public (S40)** against 0.933 for the last single models on
  Steven v4 labels ([result](../validation/results/2026-09-21-combined-labels-first-run.md)). A
  Steven-v4-trained control on the same split (`labels.source: steven_v4`) would separate the
  label change from the new training fold. Before the next queue, move the overnight launcher
  to `all_5fold_scanner_v1`; the autoresearch gate already uses it.
- **Evaluation count above ×4:** ×3 scored 0.938 and ×4 0.939 publicly, one rounding step apart,
  so where the count stops paying is unresolved, and its inference-time cost counts toward the
  efficiency prize
  ([submissions](../inference/submissions/2026-09-18-temperature-and-count-submissions.md)).
  Geometry caps the axis (fold-0 sampler replay of the `k_05` recipe, 2026-09-22): ×4 already
  takes 80% of the triplet windows of a slot's only 2D series and decodes 94% of its slices; ×5
  and ×6 reach 92% and 98% for +18% and +29% triplets but only +5% and +9% decoded slices per
  study, and every step past ×6 adds triplets from 3D series alone. On the 2026-09-17 token caches
  (older ordinary-BCE checkpoints) every candidate scored −0.0018 against 24 per type at τ 0.75
  ([bag size](../validation/results/2026-09-17-attention-temperature.md#bag-size)), so ×5 is the
  step to try if the count is tested again; ×6 and above approach that all-candidate bag
  (`outputs/analysis/acquisition-geometry-20260922/capacity_*`). On `oblique`, ×5 at τ 0.70 then
  scored 0.945 publicly against 0.944 at ×4 τ 0.75 (S47, S46), one rounding step, with +0.0003 on
  fold 0 ([result](../validation/results/2026-09-22-oblique-groups-result.md#temperature-and-count)).
- **Submit `control-check-x4-tau075`** when submissions reset: the first bundle carrying the
  main-thread warm-up, the index changes and the CPU cluster, profiled at 371.4 s of command wall
  against 440.5 s. Submit the main-thread warm-up with it.
- **Run the allocator A/B.** The tunables were implemented on 2026-09-19 and have never been run;
  A/B them on the old bundle so the shipped index, listing and CPU-cluster changes do not confound
  them, and watch per-worker RSS. Batch the remaining changes to hashed loading sources into
  one future implementation and training snapshot
  ([status](loading-speedup.md), [candidates](../backlog/loading-speedup/candidates.md)).
- **Sweep the loader worker count** around the new `INFERENCE_IO_THREADS = 8` — 8, 12 and 16 — and
  profile what inside the 15.5–18.4 s model load costs the time ([status](loading-speedup.md)).
- **Re-export the slice-importance map with left knees mirrored**, a user action outside this
  repository: the current export centres and scales patient L but never mirrors by knee side, so
  the compartment lobes on the sagittal axis cannot be told apart from two superimposed mirrored
  knees ([spatial importance map](../data-preparation/peripheral-coverage.md#spatial-importance-map)). It gates the
  [side-conditioned sagittal sampling centre](../backlog/big-bets.md#position-and-knee-side).
- **Re-measure the b0 gap under class-balanced BCE** before pricing the next efficiency recipe:
  the [backbone and fold comparisons](efficientnet-backbones.md#established-evidence) used ordinary BCE.
