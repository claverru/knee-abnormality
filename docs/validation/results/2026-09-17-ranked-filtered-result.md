# 2026-09-17 ranked-filtered result

**Three seeds of the series filtering and deterministic source ranking change scored 0.889573,
0.889650 and 0.892795 fold-0 macro AUC (mean 0.890673, standard deviation 0.0018), against
0.891407 (0.0015) for the re-scored seed controls: a −0.0007 tie with unchanged run-to-run noise.
The seed-42 checkpoint scored 0.933 public, above all three controls.** Runs
`outputs/training/ranked-filtered-seed{42,43,44}` (fold 0 of `all_5fold_v3`, preprocessing
identity `ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10`), launched by
the user on the uncommitted tree; seeds 43 and 44 used
`outputs/training/configs/ranked-filtered-seed{43,44}.yaml`, copies of `configs/training.yaml`
with the seed, name and absolute paths changed. Each run keeps its `source_snapshot.zip`. The
change is described in
[current preparation](../../data-preparation/current-preparation.md#series-indexing-guards-and-exclusions)
and the [tie-break record](../../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#implementation-2026-09-17).

## Against the seed controls

Controls are the 2026-09-17 runs ([overnight results](2026-09-17-overnight-results.md)). Their
selected checkpoints were re-scored under the new evaluation inputs; their per-epoch values below
were scored with the old, seed-dependent tie-breaks (selected checkpoints moved by at most 0.0003
when re-scored).

| Epoch | New 42 | New 43 | New 44 | Control 42 | Control 43 | Control 44 | New mean | Control mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7981 | 0.7890 | 0.7948 | 0.7978 | 0.7881 | 0.7876 | 0.7940 | 0.7912 |
| 2 | 0.8417 | 0.8332 | 0.8363 | 0.8387 | 0.8318 | 0.8292 | 0.8371 | 0.8332 |
| 3 | 0.8580 | 0.8599 | 0.8634 | 0.8518 | 0.8557 | 0.8638 | 0.8604 | 0.8571 |
| 4 | 0.8710 | 0.8684 | 0.8725 | 0.8704 | 0.8683 | 0.8748 | 0.8706 | 0.8712 |
| 5 | 0.8804 | 0.8814 | 0.8823 | 0.8806 | 0.8804 | 0.8849 | 0.8814 | 0.8820 |
| 6 | 0.8848 | 0.8845 | 0.8859 | 0.8871 | 0.8824 | 0.8855 | 0.8851 | 0.8850 |
| 7 | 0.8893 | 0.8888 | 0.8892 | 0.8929 | 0.8877 | 0.8865 | 0.8891 | 0.8890 |
| 8 | 0.8876 | 0.8888 | 0.8912 | 0.8930 | 0.8885 | 0.8906 | 0.8892 | 0.8907 |
| 9 | 0.8895 | 0.8891 | 0.8920 | 0.8922 | 0.8896 | 0.8906 | 0.8902 | 0.8908 |
| 10 | 0.8896 | 0.8897 | 0.8928 | 0.8926 | 0.8898 | 0.8916 | 0.8907 | 0.8914 |
| **Selected, new evaluation** | **0.889573** | **0.889650** | **0.892795** | 0.892697 | 0.889796 | 0.891729 | **0.890673** | 0.891407 |
| Public | 0.933 | — | — | 0.928 | 0.931 | 0.931 | | |

All three runs selected epoch 10. Selected-epoch soft-target BCE was 0.44893, 0.44799 and 0.44632
(controls under the new evaluation 0.44651, 0.44806 and 0.44749). Training took 1,732, 1,679 and
1,641 s with 18.6–19.0 GiB peaks (seed-42 control 1,641 s).

- **Near-paired runs (inferred from code).** Training draws targets, band order and supplier
  indices from per-operation streams seeded by (seed, study, type, operation, epoch), unchanged by
  this change, as are initialisation, loader order and augmentation. Only the order of the
  supplier list changed (a random permutation before, the fixed ranking now), so the same random
  index picks a different series in repeated slots, with the same uniform distribution; the 6
  dropped series touch 2 training studies. Each new run is therefore a perturbed replicate of the
  control with the same seed.
- **Paired differences:** seed 42 −0.0031, seed 43 −0.0001, seed 44 +0.0011 (mean −0.0007,
  standard deviation 0.0021). The first two runs agreeing to 0.00008 was chance (about 3–4% for
  two runs with a 0.0015 standard deviation); the third landed 0.0031 higher.

## Prediction agreement

Pairwise comparison of fold-0 validation predictions
(`outputs/analysis/p/prediction-agreement-20260917/`: `summary.json`, `pairs.csv`; script
`analyze.py` (`041bd16:docs/history/series-and-sampling-sources/prediction-agreement/analyze.py.txt`)).

| Pair | Logit correlation | Mean \|Δ probability\| | RMS Δ logit | Mean \|Δ per-target AUC\| | Two-model average gain |
|---|---:|---:|---:|---:|---:|
| Same control checkpoint, old versus new evaluation inputs (3) | 0.9992 | 0.0033–0.0035 | 0.07 | 0.0004–0.0006 | +0.0001 |
| New versus new (3) | 0.971–0.973 | 0.043–0.045 | 0.41–0.43 | 0.0035–0.0055 | +0.0031 to +0.0034 |
| Control versus control, new evaluation (3) | 0.969–0.971 | 0.045–0.047 | 0.43–0.44 | 0.0040–0.0057 | +0.0035 to +0.0036 |
| New versus same-seed control (3) | 0.972–0.976 | 0.041–0.046 | 0.39–0.43 | 0.0035–0.0046 | +0.0025 to +0.0035 |
| New versus other-seed control (6) | 0.971–0.972 | 0.044–0.046 | 0.42–0.43 | 0.0035–0.0054 | +0.0032 to +0.0035 |

- **Evaluation sampling explains little of the run-to-run spread:** changing a checkpoint's
  evaluation inputs moves its predictions about six times less than changing the training run
  (RMS logit difference 0.07 against 0.43), roughly 3% of the variance.
- **The new runs are no more alike than the controls**, and the same-seed pairs 43 and 44 are
  only slightly closer than unrelated runs.
- **Calibration varies per run** (mean predicted probability 0.328–0.331 for the new runs,
  0.332–0.337 for the controls), and averaging any two runs gains about +0.003 macro AUC.

## Reading

The change ties on fold 0 and does not alter run-to-run noise; the seed-42 checkpoint scored
**0.933 public** ([submission](../../inference/submissions/2026-09-17-ranked-filtered-submission.md)),
inside the same recipe's 0.928–0.934 range. Run-to-run noise comes from training, not from
evaluation sampling: reassigning which repeated series supplies some triplets moved a same-seed
run by up to 0.0031, as much as changing the seed. Single-run differences below about 0.003
carry no information, even at a matched seed.
