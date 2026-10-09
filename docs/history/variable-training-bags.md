# Variable training bags

**Closed 2026-09-28 without a completed run (user decision).** Varying each training visit's
triplet count, to close the gap between training bags (about 27 triplets) and inference bags (about
104 at ×4), was implemented on 2026-09-25 as `data.variable_bags` and never reached `main`. Its only
run stopped after three of ten epochs, behind its control at every epoch.

## Design

- **Draw:** each training visit's bag multiplier came from a shifted Lomax power law on [0.8, 5],
  `m = 0.8 + λ((1 − u)^(−1/0.4) − 1)` with λ 0.00704, seeded by training seed, study and epoch, at an
  unchanged average.
- **Bag:** the study's ×1 budget total (after the `study_triplet_total` top-up) times `m`, limited
  to what its pools could supply and split over its types in proportion to their ×1 budgets, as in
  ×4 inference. Types scaled together because count-normalized attention had lost −0.0035.
- **Batch ceiling of 270 triplets:** an over-limit batch kept its largest study and scaled the others
  down, never below 0.5×.
- **Rationale:** the only bag-size-dependent part of the train/inference gap was about 0.001
  (τ 0.75 gained +0.0014 at ×1 and +0.0025 at ×4) and ×3 tied ×4, so visits at ≥3× carried the
  useful exposure; moving triplets between studies at a fixed total had been free, while cutting
  every study by 20% cost −0.0041. Expected gain 0.001–0.0025, low confidence.

## Simulation

On the 3,533 fold-1 training studies over 50 shuffled epochs of batches of 8
(`outputs/analysis/b/bag-capacity-20260925/`, `outputs/analysis/v/variable-bags-scale-20260925/`): average
1.000× today's triplets; per study median 0.83×, 2.3% of visits at ≥3× (about 80 per epoch), 0.75% at
≥4×, 7.7% below 0.8× (all from the ceiling); per batch median 206 and maximum 270 triplets, 13% of
batches shrunk. 17% of studies cannot supply 4× and 37% cannot supply 5×.

## Run

`outputs/training/oa-variable-bags`: fold 1 of `all_5fold_scanner_v1`, seed 1485198267, against
`oa-detector-control` (×4 0.900568). It stopped on 2026-09-25 after epoch 2 and never produced a
×4 prediction.

| Epoch (×1) | Variable bags | Control | Δ |
|---:|---:|---:|---:|
| 0 | 0.8225 | 0.8270 | −0.0045 |
| 1 | 0.8533 | 0.8627 | −0.0094 |
| 2 | 0.8740 | 0.8782 | −0.0042 |

Three early epochs of one run are weak evidence; the idea was not measured to completion.

## Recovering the code

Never committed. The worktree's changes against `665af0e` (configuration, dataset batching, the
sampler's bag draw and ceiling, Lightning logging and the documentation above) are archived in
`outputs/analysis/v/variable-bags-worktree-20260925/` as `worktree.patch`, with
`untracked-files.tar.gz` holding `configs/variable-bags.yaml` and the two simulation scripts. The
run's `source_snapshot.zip` holds the code that trained.
