# 2026-09-30 ResNet-18 with L2-SP and without the scanner confusion head

**Adopted by the user on 2026-09-30.** On the ResNet-18 recipe, L2-SP weight anchoring added and
the scanner confusion head removed, together in one run, tie the champion: ×4 +0.00026 on fold 0
and 0.943 public (S86) against 0.943 (S82). The reason to adopt is unseen-site robustness measured
in the [site-generalization campaign](../../../autoresearch/campaigns/site-generalization/mechanism-audit.md),
not the fold-0 or public score.

## Changes

- **L2-SP:** `optimizer.anchor_alpha_early: 0.01`, `optimizer.anchor_alpha_late: 0.001`. The
  backbone conv weights are pulled toward the RadImageNet weights, stem and layer1 at 0.01 and the
  rest at 0.001; anchored weights get no weight decay.
- **No scanner confusion head:** `trainer.scanner_confusion_weight: 0`; at 0 the head is not built.
  It never engaged: its accuracy rose every epoch on ResNet-18 (0.19 → 0.33) and in 16 ReXNet runs,
  and its KL term was about 0.1% of the loss ([mechanism audit](../../../autoresearch/campaigns/site-generalization/mechanism-audit.md#per-experiment-verdicts)).
- `configs/rexnet-200.yaml` keeps the head and has no anchoring; neither change was measured on
  ReXNet.

## Result

Run `outputs/training/rn18-l2sp-no-confusion`, fold 0 of `all_5fold_scanner_v2`,
`rebuild_state_v2` labels, seed 1533697213, against `rn18-no-series-dropout` at the same fold and
seed. Comparison: `outputs/training/rn18-l2sp-no-confusion/comparison-vs-champion/comparison.json`.

| | Champion | Candidate | Δ |
|---|---:|---:|---:|
| ×4 macro AUC | 0.927806 | 0.928062 | +0.00026, 95% [−0.0013, +0.0018] |
| ×4 soft macro AUC | 0.791372 | 0.791306 | −0.00007 |
| Selected epoch | 10 | 9 | |
| Training seconds | 1,202 | 1,222 | +1.6% |
| Public, ×4 τ 0.9 | 0.943 (S82) | 0.943 (S86) | 0 |

Per label, Baker's +0.0090 and Lateral OA +0.0033 offset ACL −0.0023, Medial OA −0.0023 and
Effusion −0.0015; every other label is within ±0.0013. Seed noise on this recipe is about
0.002–0.003, so the fold-0 result is a tie.

## Unseen-site evidence

From the sites-out splits (fold 0 with whole sites held out), all on ResNet-18:

| Measurement | Held-out-site Δ | Seen-site Δ |
|---|---:|---:|
| L2-SP added (campaign) | +0.0015 | — |
| L2-SP removed (campaign) | −0.0014 | — |
| L2-SP on split B (Spanish, Turkish, Croatian sites held out) | +0.0009 [−0.00002, +0.0020] | −0.0010 [−0.0023, +0.0003] |
| L2-SP in the matched 2×2 | +0.0023 [+0.0012, +0.0036] | +0.0003 [−0.0009, +0.0015] |
| Confusion head removed, two seeds | −0.0001, −0.0007 | −0.0012, +0.0009 |

Public at matched seed on the sites-out split: control 0.941 (S83), with L2-SP 0.943 (S84). The
held-out cost of an unseen site comes mostly from between-site label-rate offsets, which L2-SP
does not address; its gain there is small and one run per measurement.

## Reproduce

```bash
rtk proxy uv run --locked python -m scripts.pipeline.train --config configs/training.yaml \
  --run-id rn18-l2sp-no-confusion --validation.folds '[0]' --seed 1533697213
```
