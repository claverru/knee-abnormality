# Historical crop-selection sources

The snapshot files were removed from the tree on 2026-09-25. Each reference below is a
`commit:path`; read it with `git show <commit>:<path>`.

The [crop-selection investigation](../../crop-selection-investigation.md) holds the findings. It ran
on branch `meniscus-soft-coupling` as uncommitted changes on parent commit `ea19772`.

- **2026-09-16, first change:** learned zoom replaced per-label crop selection (the shared-crops
  recipe, 0.879626).
- **2026-09-16, dropped:** learned zoom scored 0.869951 and the investigation was dropped. The
  code went back to the full-slice `00-baseline` recipe
  ([current model](../../../modeling/current-model.md)).

The [implementation plan](../../crop-selection-plan.md) and the
[learned-zoom diagram](../../learned-zoom.png) sit next to the investigation record.

## Documentation snapshots

These are the working-tree documents just before each rewrite. Links inside them are relative to
their original locations.

| Original path | Snapshot | Taken before | SHA-256 |
|---|---|---|---|
| `docs/modeling/active-spatial-token-model.md` | active-spatial-token-model.md.txt (`f076a09:docs/history/crop-selection-sources/active-spatial-token-model.md.txt`) | The learned-zoom rewrite (shared-crops recipe) | `352f44168f6a9702945d8a668e4793572de23775481f8989fa6cd98c6876934b` |
| `docs/model-shapes.md` | model-shapes.md.txt (`f076a09:docs/history/crop-selection-sources/model-shapes.md.txt`) | The learned-zoom rewrite (shared-crops diagram) | `16262d282f8ef2c41dfd3ae0b187720b9feba1d97eb4a4a3e552159a9e225d47` |
| `docs/modeling/active-spatial-token-model.md` | learned-zoom-active-spatial-token-model.md.txt (`f076a09:docs/history/crop-selection-sources/learned-zoom-active-spatial-token-model.md.txt`) | The full-slice restoration (learned-zoom recipe and profile) | `946f3567846854fdc311f3a3613bcb877850c196747ed4e1c5a83174da53170b` |
| `docs/model-shapes.md` | learned-zoom-model-shapes.md.txt (`f076a09:docs/history/crop-selection-sources/learned-zoom-model-shapes.md.txt`) | The full-slice restoration (learned-zoom diagram) | `65e096815722cba6f44acd9eb7207e8126a339d7f1737dad78a33b87d2b5e57b` |
| `docs/modeling/backbones/backbone-swaps.md` | learned-zoom-backbone-swaps.md.txt (`f076a09:docs/history/crop-selection-sources/learned-zoom-backbone-swaps.md.txt`) | The full-slice restoration (learned-zoom budget, scout formula and requirements) | `0e3796bb645ff68f74d264ddfe5447b27f024aa286b4febbf8ee3f5af08a0d35` |
| `docs/ongoing.md` | ongoing.md.txt (`f076a09:docs/history/crop-selection-sources/ongoing.md.txt`) | The full-slice restoration (results, findings, open options, glimpse plan and candidate runs) | `f51a0b38eec78b2f13ba0e225390edb7bf240d3020573d82554c2ee13e00152c` |

## Removed code

Each training run directory holds `source_snapshot.zip`, and those archives reproduce the runs.

- **Learned zoom:** `outputs/training/learned-zoom/source_snapshot.zip` (SHA-256
  `781aadfd8fdd2ab6bc74bf740e5f74215551174bcab9328b267c0b1ed61974ee`) holds the implementation
  as trained. That includes `crop_selection.py` (`CoarseScout`, `GlimpseLayout`, the per-study
  clusters and greedy facility location), the glimpse model and `scripts/adhoc/probe_glimpses.py` (removed).
- **Shared crops:** `outputs/training/shared-crops/source_snapshot.zip` (SHA-256
  `28f63b69bc70877144cafc39d5d6663120fb3a812ef056d68958a0f12ada4f43`) holds that implementation.
  Its preprocessing sources hash to the identity recorded by the run
  (`460d9461a6fa25370d14c2f0b578df8b83a44dcbfc4db7a50342fb00479fb831`). Every file removed or
  rewritten by the learned-zoom change had the same line count in the working tree before that
  change.

| Removed with learned zoom | Member SHA-256 in the shared-crops archive |
|---|---|
| `src/knee_abnormality/crop_selection.py` (removed) (`label_crops`, `crop_scores`, `value_map`, `cut_crops`) | `eaf128ec150b2608f8a859fb27b5d303f031e1f791a4deff670f0221e8256a62` |
| `src/knee_abnormality/modeling/model.py` (gated coarse and fine pools, coupled fine scores) | `d5e9750c58f2d492634b1352ad28e756db2163b9974b5fa2c71eaf6437e08ab1` |
| `scripts/adhoc/analyze_crop_diversity.py` (removed) (per-label crop placement) | `50d61a249f96c3ff9377b3ee24d9cb2418a493180613b92f45c5d61a522f5407` |
| `scripts/adhoc/evaluate_crop_placement.py` (removed) (learned, uniform and centred placement) | `1ab7bb161df654be2a256bdc45ee81940bbb9b8190396fc3170742af08477e9d` |
| `configs/crop130.yaml` (removed) (full-triplet 130 mm rerun, edited to validate against crop fields) | `c63aff64bc600b1a815ee0253973cf66ffa2a0f017e5b1bc98ec10b5ff4d7758` |

The crop130 run's original input configuration is
`outputs/training/overnight-20260914T220805Z-ozsz4qbu/runs/crop130/input_config.yaml`. Its result is in
the [overnight results](../../../validation/results/campaigns-and-queues/2026-09-15-overnight-results.md).

Restoration had already begun when this record was written, so it is not verified whether the
working-tree learned-zoom sources changed after the run archive was taken.
`scripts/adhoc/probe_glimpses.py`, since retired, matched the archive.
