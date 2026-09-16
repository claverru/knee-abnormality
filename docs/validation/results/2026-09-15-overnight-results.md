# 2026-09-15 overnight results

Queue `outputs/training/overnight-20260914T220805Z-ozsz4qbu` started at 00:08 CEST on
2026-09-15 and completed seven of sixteen runs. Its `07-crop130` log ended without an
error at 03:59, during epoch 8, when the machine or WSL stopped; no process was left
running. Runs `08-clip-0.01` through `15-backbone-eval` never started. `07-crop130` was
retrained from scratch as `runs/crop130`; its configuration differs from `00-baseline`
only in `data.spacing_mm`. All runs use seed 42, fold 0, `graph_layers: 0` and no
additive 3D embedding.

| Run | Best macro AUC (epoch) | vs `00-baseline` | BCE at best | Gold at best (11) | Last epoch | Training | Peak allocation |
|---|---:|---:|---:|---:|---:|---:|---:|
| `00-baseline` | 0.892763 (10) | — | 0.44767 | 0.932011 | 0.892763 | 1,670 s | 20.02 GB |
| `01-sigma30` | 0.892802 (7) | +0.000039 | 0.45116 | 0.935317 | 0.890847 | 1,760 s | 20.11 GB |
| `02-higher-lr` | 0.892541 (10) | −0.000222 | 0.44603 | 0.940146 | 0.892541 | 1,791 s | 20.02 GB |
| **`03-min-lr-0.1`** | **0.893171** (8) | **+0.000408** | 0.44839 | 0.906316 | 0.892674 | 1,892 s | 20.02 GB |
| `04-series-dropout-0.3` | 0.892271 (10) | −0.000492 | 0.44735 | 0.930886 | 0.892271 | 1,627 s | 18.52 GB |
| `05-wd-0.05` | 0.891926 (10) | −0.000837 | 0.44807 | 0.920503 | 0.891926 | 1,760 s | 20.02 GB |
| `06-affine-strong` | 0.890374 (7) | −0.002389 | 0.45272 | 0.923280 | 0.889980 | 1,743 s | 20.02 GB |
| `crop130` (rerun) | 0.892500 (8) | −0.000263 | 0.44991 | 0.929762 | 0.892232 | 1,660 s | 20.02 GB |

- `03-min-lr-0.1` is the best run, +0.000408 over its control. It peaks at epoch 8 and
  still holds 0.892674 at epoch 10.
- `06-affine-strong` is the only large regression. `01-sigma30` and `crop130` are ties.
  `02-higher-lr`, `04-series-dropout-0.3` and `05-wd-0.05` are small losses.
- `00-baseline`, `02-higher-lr`, `04-series-dropout-0.3` and `05-wd-0.05` peak at their
  final epoch, so ten epochs may not saturate training; `09-epochs12` was not run.
- Small differences matter on the leaderboard, but these are single-seed, single-fold
  results without paired intervals. Confirm a candidate with the
  [fold-run comparison](../fold-run-comparison.md) and a replicate before adopting it.
- `00-baseline` scored 0.892763, 0.000403 above the champion on fold 0, but its
  [public score was 0.934](../../inference/submissions/2026-09-15-overnight-00-baseline-submission.md),
  0.002 below the champion's 0.936.

These runs share the no-3D, graph-off base. Their deltas are relative to that base, not to
the champion recipe.
