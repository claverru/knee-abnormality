# Next

Experiments run one by one against a control at the same validation multiplier, as one unseeded
run each (random seed, recorded). The [overnight queue](../backlog/overnight.md) holds the
hyperparameter candidates; nothing is queued for tonight. The b0 line's open leads are in
the [b0 queue result](../validation/results/2026-09-18-b0-hyperparameter-queues.md#open).

Also pending:

- **Evaluation count above ×4:** ×3 scored 0.938 and ×4 0.939 publicly, one rounding step apart,
  so where the count stops paying is unresolved, and its inference-time cost counts toward the
  efficiency prize
  ([submissions](../inference/submissions/2026-09-18-temperature-and-count-submissions.md)).
- Data-loading speedups: three retrain-free changes shipped and measured 2026-09-18, worth 1.10x
  of inference wall; the allocator A/B is the last retrain-free one, and the bit-exact cluster
  waits for the next retrain ([status](loading-speedup.md),
  [candidates](../backlog/loading-speedup/candidates.md)).
- Submit the main-thread warm-up with the next submission; torch-free coordinator
  ([runtime changes](../backlog/loading-speedup/candidates.md#runtime-changes-outside-the-loading-path)).
