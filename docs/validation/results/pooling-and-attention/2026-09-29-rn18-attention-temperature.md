# 2026-09-29 RadImageNet ResNet-18 attention temperature

**The RadImageNet ResNet-18 incumbent peaks at τ 0.9–1.0, not the ReXNet τ 0.75.** τ 0.9 gains
+0.0005 macro AUC over 0.75 (80% [+0.0001, +0.0010]); τ 1.0 gains +0.0006 with a wider interval.
The ReXNet optimum ([2026-09-21](../labels-and-losses/2026-09-21-combined-labels-first-run.md#attention-temperature))
does not transfer. At τ 0.9 this model scored 0.944 public (S79), tying the ReXNet
`rebuild_state_v2` fold-0 control at τ 0.75 (S78).

## Run

- Checkpoint: `radimagenet-rn18` campaign incumbent `rn18-more-triplets` (36 triplets, no series
  dropout), fold 0 of `all_5fold_scanner_v2`, rebuilt labels `rebuild_state_v2`.
- Inference: one Kaggle T4 FP16 pass per τ at ×4, through the submission bundle
  (`outputs/submission/exports/rn18-radimagenet-36trip-f0-20260929-x4-tau*`).
- Deltas are against τ 0.75 with 80% paired report-group bootstrap intervals (seed 20260914).

| τ | ×4 macro | Δ vs 0.75 | 80% interval | P(Δ > 0) |
|---:|---:|---:|---|---:|
| 0.50 | 0.9253 | −0.0038 | [−0.0046, −0.0029] | 0.00 |
| 0.60 | 0.9274 | −0.0017 | [−0.0022, −0.0012] | 0.00 |
| 0.75 | 0.9291 | — | — | — |
| 0.90 | 0.9296 | +0.0005 | [+0.0001, +0.0010] | 0.94 |
| 1.00 | 0.9297 | +0.0006 | [−0.0001, +0.0012] | 0.86 |
| 1.25 | 0.9288 | −0.0004 | [−0.0015, +0.0007] | 0.34 |

τ was chosen on the same fold, so the gain is slightly optimistic. It is one checkpoint; whether
the shift comes from the backbone or from removing series dropout is not separated.

## Reproduce

```bash
env CUDA_VISIBLE_DEVICES= uv run --locked python -m scripts.adhoc.score_temperature_sweep --sweep-dir outputs/analysis/t/tau-sweep-rn18-20260929/scored --split-dir data/folds/all_5fold_scanner_v2 --fold 0 --reference-tau 0.75 --seed 20260914
```
