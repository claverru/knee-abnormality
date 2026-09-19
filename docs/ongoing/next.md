# Next

The user-launched [fixed class-balanced BCE run](../validation/results/2026-09-19-balanced-bce-result.md)
finished at 0.895282, +0.008796 over the closest same-seed historical reference. An exact
same-source ordinary-BCE control has not been run; no further run is scheduled. Other experiments
run one by one against a control as one unseeded run each (random seed, recorded). The
[overnight queue](../backlog/overnight.md) holds the remaining hyperparameter candidates.
The b0 line's open leads are in
the [b0 queue result](../validation/results/2026-09-18-b0-hyperparameter-queues.md#open).

Also pending:

- **Evaluation count above ×4:** ×3 scored 0.938 and ×4 0.939 publicly, one rounding step apart,
  so where the count stops paying is unresolved, and its inference-time cost counts toward the
  efficiency prize
  ([submissions](../inference/submissions/2026-09-18-temperature-and-count-submissions.md)).
- **Submit `control-check-x4-tau075`** when submissions reset: the first bundle carrying the
  main-thread warm-up, the index changes and the CPU cluster, profiled at 371.4 s of command wall
  against 440.5 s.
- Data-loading speedups: the index changes and the CPU cluster are shipped and measured, together
  worth 470.5 → 350.8 s of worker wall. Left: the io-thread sweep, the allocator A/B, and the two
  candidates that wait for the next identity bump ([status](loading-speedup.md),
  [candidates](../backlog/loading-speedup/candidates.md)).
- **Re-fetch the two corrupted training studies** once Kaggle's downloads work again: the host
  re-uploaded the competition files on 2026-09-17 to repair them and says the update may have
  failed, and a successful repair would make our `KNOWN_BAD_*` exclusions discard usable slices
  ([data contract](../data-preparation/public-data-contract-and-decoding.md)).
- **Re-export the slice-importance map with left knees mirrored**, a user action outside this
  repository: the current export centres and scales patient L but never mirrors by knee side, so
  the compartment lobes on the sagittal axis cannot be told apart from two superimposed mirrored
  knees ([spatial importance map](peripheral-coverage.md#spatial-importance-map)). It gates the
  [side-conditioned sagittal sampling centre](../backlog/big-bets.md#position-and-knee-side); the
  per-plane sigma candidate does not wait for it.
- Submit the main-thread warm-up with the next submission; torch-free coordinator
  ([runtime changes](../backlog/loading-speedup/candidates.md#runtime-changes-outside-the-loading-path)).
