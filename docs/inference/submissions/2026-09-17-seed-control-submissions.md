# Seed-control submissions

The three controls of queue `outputs/training/overnight-20260916T214520Z-t8k_kw2z` are the restored
full-slice recipe trained with seeds 42, 43 and 44 on the same code as the
[EMA submission](2026-09-17-ema-submission.md); seed 42's weights are exactly the live weights of
that EMA run. Each was exported on 2026-09-17 with the fast runtime (2× images, four studies per GPU,
16 total loader workers, automatic FP16 backbone chunks on T4) and submitted through the detached
workflow; seeds 43 and 44 ran in parallel.

| Submission | Seed | Selected epoch | Fold-0 macro AUC | Public AUC | ID | Launch to score |
|---|---:|---:|---:|---:|---:|---:|
| `knee-control-seed42-20260917` | 42 | 8 | 0.892995 | 0.928 | 56298600 | 12m 16s |
| `knee-control-seed43-20260917` | 43 | 10 | 0.889837 | 0.931 | 56299151 | 12m 47s |
| `knee-control-seed44-20260917` | 44 | 10 | 0.891627 | 0.931 | 56299154 | 10m 47s |

Bundles are under `outputs/submission/exports/`, workflow records under
`outputs/submission/records/` with the same names; datasets and notebooks are
`claverru/<name>` and `claverru/<name>-inference`. Checkpoint SHA-256: seed 42
`66988d6191f590ec241cd3cc6f64a1bbdfffdab270c851544377d3e4834ed860`, seed 43
`8cce373a1039e73afcb394a10ee9c68bb90fe96780d0f8e9a0b030e30d8642fd`, seed 44
`05060efdcc5e37ffe3495644ecdd23c1cdbb65c8adf7d1346c58a087de862831`.

## Reading

| Model | Fold-0 macro AUC | Public AUC |
|---|---:|---:|
| Seed 42 control | 0.892995 | 0.928 |
| EMA of the seed-42 run | 0.893155 | 0.929 |
| Seed 43 control | 0.889837 | 0.931 |
| Seed 44 control | 0.891627 | 0.931 |
| `00-baseline` (same recipe, pre-restore code, 2026-09-15) | 0.892763 | 0.934 |
| Reflection/sigma-40 champion | 0.892360 | 0.936 |

- **Fold-0 order does not predict public order.** Among the three seeds the public order is the
  reverse of the fold-0 order. The same recipe trained four times (three seeds plus
  `00-baseline`) scored 0.928–0.934 publicly (mean of the three seeds 0.930) and
  0.8898–0.8930 on fold 0.
- **EMA gained +0.001 publicly** over its own live weights, the only paired comparison here.
- **The seed also changes the scored inputs.** Validation and test triplet sampling break
  ties between repeated series of one type with the seed, so these public and fold-0 differences
  mix training noise with evaluation-sampling noise.
- Public differences below about 0.006 between separately trained runs are noise; ranking such
  models needs several runs per recipe or multi-fold evidence.
