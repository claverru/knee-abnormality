# Historical regional sources

The snapshot files were removed from the tree on 2026-09-25. Each reference below is a
`commit:path`; read it with `git show <commit>:<path>`.

These text snapshots preserve the removed regional implementation and its three
investigations as they stood before the spatial-token replacement. They are
historical source records, excluded from executable script catalogs. Reproduction
requires the corresponding original run or analysis source bundle and saved
configuration, including its old dataset and checkpoint contract; copying one
file into the current package does not reproduce that environment.

- Original modeling description (`f076a09:docs/history/regional-sources/modeling-before.md.txt`)
- Regional module (`f076a09:docs/history/regional-sources/spatial_pooling.py.txt`)
- Coverage measurement (`f076a09:docs/history/regional-sources/measure_fusion_coverage.py.txt`)
- Runtime profile (`f076a09:docs/history/regional-sources/profile_spatial_fusion.py.txt`)
- Regional usage diagnosis (`f076a09:docs/history/regional-sources/diagnose_fusion_usage.py.txt`)

Measurements and original artifact locations remain in
[validation](../../../validation/results/pooling-and-attention/2026-09-13-per-plane-fusion-usage-diagnosis.md) and
[modeling](../../../modeling/historical-regional-fusion.md#per-plane-coverage-radius).
