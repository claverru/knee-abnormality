# Next

Pending work only. Finished runs and decisions live in the
[project state](state.md), the [validation results](../validation/results/INDEX.md) and the
canonical topic documents; this file is not a log.

**No queue is loaded.** The [overnight queue](../backlog/overnight.md) is empty: the
[2026-09-20 fourteen-run queue](../validation/results/2026-09-20-overnight-hyperparameter-results.md)
and the [2026-09-19 five-run queue](../validation/results/2026-09-19-untried-hyperparameter-queue.md)
both completed with no adoption, and the 2026-09-19 controls replicated class-balanced BCE at a
second seed. The autoresearch campaign is stopped, its run-level stratified sampler rejected for
adoption and reverted out of main
([result](../validation/results/2026-09-20-stratified-sampling-promotion.md)). Other experiments
run one by one against a control as one unseeded run each (random seed, recorded).

Pending:

- **Evaluation count above ×4:** ×3 scored 0.938 and ×4 0.939 publicly, one rounding step apart,
  so where the count stops paying is unresolved, and its inference-time cost counts toward the
  efficiency prize
  ([submissions](../inference/submissions/2026-09-18-temperature-and-count-submissions.md)).
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
