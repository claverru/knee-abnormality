# Historical meniscus-coupling sources

The snapshot files were removed from the tree on 2026-09-25. Each reference below is a
`commit:path`; read it with `git show <commit>:<path>`.

This exact text snapshot preserves the CPU comparison of the soft meniscus classifier
coupling run with the reflection/sigma-40 champion, in the standalone form it took after
the coupling loss was removed from training. It left the script catalog on 2026-09-16,
when the candidate run's checkpoints were deleted; it loads both runs' selected checkpoints,
so it can no longer be rerun. The [result](../../../validation/results/labels-and-losses/2026-09-14-meniscus-soft-coupling-result.md)
and [modeling summary](../../../modeling/retired-side-and-compartment-variants.md) own the findings.

| Original path | Source snapshot | SHA-256 |
|---|---|---|
| `scripts/adhoc/analyze_meniscus_coupling.py` (removed) | analyze_meniscus_coupling.py.txt (`f076a09:docs/history/meniscus-coupling-sources/analyze_meniscus_coupling.py.txt`) | `f52cbc311c6c22e0b5742faa8238af9a9f6e6be86321ac329b712f99f354ce47` |

The comparison that produced `outputs/analysis/m/meniscus-coupling-20260914/` ran the earlier
version kept in its `source/` directory, which imported the metrics from the since-removed
`src/knee_abnormality/losses.py` (removed). The run keeps its configuration, source snapshot, metrics
and predictions under `outputs/training/meniscus-soft-coupling/`; none of these are version controlled.
