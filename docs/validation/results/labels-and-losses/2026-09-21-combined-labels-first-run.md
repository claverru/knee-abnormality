# 2026-09-21 first run on combined labels and the scanner split

**`fold0-combined-scanner` scores 0.9020 macro AUC on fold 0 of `all_5fold_scanner_v1` at ×1 and
τ 1. At submission settings, ×4 and τ 0.75, it scores 0.9079.** This is the first run on the new split
and the new fold-0 reference. On the 209 studies that both the old and new fold 0 hold out, it ties
the learned-strength champion. MCL is no longer the weakest label under the new definitions.

**Public score: 0.940 (S40, ×4 and τ 0.75, single fold-0 model).** The last two single fold-0
models on Steven v4 labels, S38 control B and S39, both scored 0.933. The previous best single model
was 0.939 (S31, ranked-filtered seed 42). The 0.943 best overall is a ten-model ensemble. The new labels and new training
fold changed together, so this is one submission, not an isolated label effect.

## Run

- Recipe: the learned-strength research recipe with the
  [combined labels](../../../public-labels/combined-labels.md) on
  [`all_5fold_scanner_v1`](../../validation-splits.md#superseded-splits), fold 0, seed 2263738815. Ten epochs;
  epoch 9 selected at 0.90205.
- Artifacts: `outputs/training/fold0-combined-scanner/`, built from source at `a960863` plus
  uncommitted label and split changes, captured in `source_snapshot.zip`.
- Training took 1,576 s, with 21.0 GB peak allocated GPU memory.

## Per-label AUC on its own fold

Targets are the combined labels, gold where available. There are 905 studies, with 46 MCL
positives.

| Label | ×1, τ 1 | ×4, τ 0.75 |
|---|---:|---:|
| ACL | 0.946 | 0.951 |
| MCL | 0.923 | 0.925 |
| Medial Meniscus | 0.927 | 0.930 |
| Lateral Meniscus | 0.900 | 0.909 |
| Medial OA | 0.898 | 0.903 |
| Lateral OA | 0.871 | 0.877 |
| PF OA | 0.864 | 0.870 |
| Effusion | 0.860 | 0.860 |
| Synovitis | 0.892 | 0.889 |
| Baker's | 0.926 | 0.929 |
| Contusion | 0.891 | 0.918 |
| Fracture | 0.925 | 0.934 |
| **Macro** | **0.9020** | **0.9079** |

The weakest labels are now Effusion, PF OA and Lateral OA. Scored against Steven v4's definitions
instead, MCL falls to 0.788 and the macro to 0.888. The model ranks the sprains that Steven v4 calls
positive low, as its training labels intend.

## Against the previous champion and control

The old runs trained on studies that now sit in the new fold 0. The only fair shared set is the
209 studies held out by both `all_5fold_v3` and the new split's fold 0. It has 2 gold studies and
only 5 MCL positives under the combined labels.

| Run | Shared 209, combined targets | Shared 209, Steven v4 targets |
|---|---:|---:|
| New run | 0.922 | 0.903 |
| Learned strength (champion) | 0.920 | 0.903 |
| Control repeat | 0.916 | 0.904 |

Paired report-group bootstrap of the new run's macro difference:

| Comparison | Combined targets | Steven v4 targets |
|---|---|---|
| vs learned strength | +0.0017, 80% interval [−0.0052, +0.0085] | −0.0002, 80% interval [−0.0082, +0.0077] |
| vs control repeat | +0.0056, 80% interval [−0.0006, +0.0113] | −0.0010, 80% interval [−0.0084, +0.0059] |

On the shared set, the new run is higher on ACL and PF OA under the combined targets and lower on
Fracture. None of this separates the models; 209 studies are too few. The label change's real test is
the public leaderboard. Own-fold scores use different studies and targets and are not comparable:
learned strength scores 0.897 on Steven v4 targets and 0.904 on combined targets, both over the old
fold 0.

## Attention temperature

The same checkpoint was run through repository inference on fold 0 at each τ. Deltas are against
τ 1 at the same multiplier, with 80% paired report-group bootstrap intervals.

| τ | ×1 macro | ×1 Δ | ×4 macro | ×4 Δ |
|---:|---:|---|---:|---|
| 0.50 | 0.8974 | −0.0045 | 0.9041 | −0.0013 |
| 0.60 | 0.9012 | −0.0008 | 0.9071 | +0.0017 [+0.0000, +0.0034] |
| 0.65 | 0.9022 | +0.0003 | 0.9077 | +0.0024 [+0.0009, +0.0038] |
| 0.70 | 0.9030 | +0.0010 | 0.9079 | +0.0025 [+0.0013, +0.0038] |
| **0.75** | 0.9033 | +0.0014 [+0.0004, +0.0023] | **0.9079** | **+0.0025 [+0.0015, +0.0035]** |
| 0.80 | 0.9034 | +0.0014 [+0.0006, +0.0022] | 0.9077 | +0.0023 [+0.0015, +0.0031] |
| 0.85 | 0.9032 | +0.0012 | 0.9074 | +0.0020 |
| 0.90 | 0.9029 | +0.0009 | 0.9069 | +0.0015 |
| 1.00 | 0.9020 | — | 0.9054 | — |

**Keep τ 0.75.** At ×4, 0.70 and 0.75 tie, and 0.75 has the tighter interval. It also matches the
[earlier finding](../pooling-and-attention/2026-09-17-attention-temperature.md) and the current bundles. At ×1 the peak
moves slightly to 0.75–0.80, and the gain is smaller. Moving from ×1 to ×4 at τ 1 adds 0.0034.
τ was chosen on the same fold, so the intervals are slightly optimistic.

## Reproduction

- Comparison: `scripts.adhoc.compare_across_splits` →
  `outputs/analysis/c/combined-scanner-vs-learned-strength/`, on CPU from saved predictions:

  ```bash
  env CUDA_VISIBLE_DEVICES= uv run --locked python -m scripts.adhoc.compare_across_splits --candidate outputs/training/fold0-combined-scanner --references '{"learned_strength":"outputs/training/anatomical-attention/learned-strength","control_repeat":"outputs/training/anatomical-attention/control-repeat"}' --output-dir outputs/analysis/combined-scanner-vs-learned-strength-new
  ```

- Sweep: `outputs/analysis/t/tau-sweep-fold0-combined-scanner/run_sweep.sh` writes one
  `x<multiplier>-tau<value>.csv` per setting with `scripts.pipeline.predict --split validation`;
  `env CUDA_VISIBLE_DEVICES= uv run --locked python -m scripts.adhoc.score_temperature_sweep --sweep-dir outputs/analysis/t/tau-sweep-fold0-combined-scanner`
  scores them on CPU into `scores.csv` and `summary.json` in the same folder.
  Each τ is one full inference pass: 26 s at ×1 and 65 s at ×4. No token cache exists for the
  current model; the 2026-09-17 caches belong to older checkpoints.
- The ×1, τ 1 pass reproduces the saved validation score to within 5e-5 (0.90200 against 0.90205,
  bf16).
