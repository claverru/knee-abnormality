# 2026-09-13 local and public agreement after the split change

Superseded by the [2026-09-18 comparison](2026-09-18-local-public-metric-comparison.md), which covers nineteen comparable submissions, adds gold-only AUC, BCE and per-metric run-to-run noise, and withdraws the gold-only diagnostic recommended below. The six-run cohort and its numbers remain valid as measured.

Before the Tonylin experiment, local full-fold AUC tracked the public leaderboard strongly across the six
distinct submitted checkpoints trained on `all_5fold_v3`. All use fold 0, seed 42,
the same 904 validation studies, matching label/split/membership/input identities,
and 2× image limits for both validation and submission. Earlier baseline3 and
baseline4 submissions use `all_5fold_v1` and are excluded; repeated submissions
of one checkpoint at different densities are not independent model comparisons.

| Submitted model | Local macro AUC | Public AUC | Submission |
|---|---:|---:|---:|
| Overnight faster head | 0.877244 | 0.903 | 56181919 |
| Overnight higher LR | 0.882938 | 0.918 | 56181821 |
| Overnight longer training | 0.883536 | 0.913 | 56181804 |
| Regional spatial fusion | 0.884169 | 0.920 | 56190442 |
| `crop140mm` | 0.886122 | 0.925 | 56182948 |
| Spatial tokens, five embeddings | **0.889506** | **0.929** | **56206027** |

Descriptive Pearson correlation is **0.9634** and Spearman rank correlation is
**0.9429**. Fourteen of fifteen pairwise rankings agree. The exception is the
longer versus higher-LR run: a local advantage of 0.000598 reverses into a public
deficit of 0.005. This supports continuing to use full-fold AUC as the primary
development signal, with BCE as supporting evidence and gold-only AUC diagnostic.

The v3 protocol stratifies gold-only positive/negative indicators and gold
membership alongside the effective labels, with both classes required for every
target in each gold validation fold. Gold counts are 11, 10, 14, 12 and 11 across
folds; fold 0 still contains 893 report-derived and only eleven gold studies.
The observed agreement follows the split change, but these six selected runs
cannot establish that balancing gold labels caused the correlation. Scores are
rounded to three decimals, model submissions were selected during development,
and all runs use one fold and seed. No local-to-public score conversion or
private-leaderboard guarantee is inferred.

The later [Tonylin label comparison](2026-09-13-tonylin-label-comparison.md) is
outside this six-run, fixed-label analysis: local AUC rises to 0.892306 while
public AUC falls to 0.921. Preserve the correlation above as the measured
Steven-v4 cohort result; it does not establish reliable ranking across changes
in training-label source.

The [comparison artifact](../../../outputs/analysis/local-public-all-5fold-v3-20260913/comparison.json)
records every checkpoint, bundle and submission identity, saved local metrics,
completed API scores, matching input identities and exclusions. The
[scatter plot](../../../outputs/analysis/local-public-all-5fold-v3-20260913/local-public.png)
is also available as an
[SVG](../../../outputs/analysis/local-public-all-5fold-v3-20260913/local-public.svg).
This analysis only reads saved metrics and submission records; no training,
inference or tests were run.
