# 2026-09-17 random per-type allocation in training

**Training on random per-type splits ties the fixed budgets: 0.890535 against 0.889573, +0.0010
[−0.0025, +0.0043].** The model tolerates a different split every epoch at no measurable cost, so
allocation rules can now be searched on a checkpoint that was not trained on one fixed split.

## Run

- **Candidate:** `outputs/training/allocation-concentration`, one unseeded run (seed 2634103341
  drawn at random), `data.allocation_concentration: 10.0`, everything else
  `configs/training.yaml`, fold 0 of `all_5fold_v3`. The option was removed from the code after this
  result and the [test-time allocation](2026-09-17-test-time-allocation.md#the-allocation-trained-checkpoint-responds-the-same)
  follow-up; the run keeps its own archived sources. It worked as follows: each epoch each study's per-type triplet counts are redrawn from a Dirichlet around today's
  budgets, keeping the study's total. Validation, prediction and Kaggle inference keep the fixed
  budgets.
- **Reference:** `outputs/training/ranked-filtered-seed42` (0.889573, seed 42,
  [result](2026-09-17-ranked-filtered-result.md)). The code differs by the option, so the
  comparison allowed the preprocessing-identity difference.
- **Comparison:** `scripts/tools/compare_fold_runs.py`, 2,000 paired report-group bootstrap draws.
  Artifacts: `outputs/analysis/allocation-concentration-vs-ranked-filtered-seed42/`.

| | Selected epoch | Macro AUC | Training | Peak CUDA |
|---|---:|---:|---:|---:|
| `ranked-filtered-seed42` | 10 | 0.889573 | 1,732 s | 19.98 GiB |
| `allocation-concentration` | 10 | 0.890535 | 1,634 s | 20.39 GiB |

- **Macro Δ:** +0.0010 [−0.0025, +0.0043], a tie. Run-to-run noise alone has a standard deviation
  of 0.0015–0.0018, and the two runs used different seeds.
- **Families:** meniscus +0.0029 [−0.0035, +0.0099], OA −0.0008 [−0.0072, +0.0056], the other
  eight +0.0008 [−0.0035, +0.0051].
- **Per target:** fracture +0.0075, ACL +0.0057, lateral meniscus +0.0052, Baker's +0.0029 against
  MCL −0.0095, medial OA −0.0018 and contusion −0.0014.
- **Predictions** differ by a mean 0.045 with per-target Spearman 0.92–0.97, the scale seen
  between two runs of the same configuration.
- **Speed:** training ran 5.7% faster. A skewed draw concentrates triplets in fewer series, which
  decodes fewer unique slices.

## What the draws looked like

At concentration 10, for a type with a base of 6 triplets in a 26-triplet study, the drawn count
spans 1 to 12 between the 5th and 95th percentiles, and about 60% of draws leave some present type
with a single triplet, on top of series dropout 0.2. The tie therefore covers a wide range of
splits, not a mild perturbation.

## Reading

- **The model does not need its training split.** This is the first training-time evidence in the
  [allocation investigation](../../history/triplet-budget-allocation.md), after test-time
  reallocation on fixed-split checkpoints
  [failed to confirm](2026-09-17-test-time-allocation.md).
- **It unblocks the search.** A token cache built on this checkpoint scores any allocation
  on-distribution, which the fixed-split checkpoints could not do.
- **It is not itself an improvement,** and the search it unblocked found nothing, so the option was
  removed.
- **Caveats:** one run against one seed-42 control; the MCL drop is inside the noise; and the
  allocation option changes the preprocessing source digest.
