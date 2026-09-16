# Unverified runtime evidence and efficiency leads

## Unverified submission results

These are distinct configurations and dates, not a controlled scaling study.
The recorded public leaderboard scores and timings have not been independently
verified.

| Date | Setup | Score | Reported inference/submission time |
|---|---|---|---|
| 2026-08-29 | Two-model, fold-0 ensemble | Not recorded | About 4 hours; one 300M-parameter model took about 3 hours |
| 2026-09-01 | Optimized 15-model ensemble | Not stated with this timing | About 3 hours |
| 2026-09-02 | Five small-model runs | 0.938 | About 40 minutes |
| 2026-09-03 | Small-model five-fold ensemble | 0.944 | About 35 minutes |
| 2026-09-03 | Small single model | 0.938 | About 20 minutes |
| 2026-09-08 | Best single-fold model | 0.944 | About 1 hour |

Architecture identities, checkpoint versions, hardware, and inclusion of
startup/data loading are incomplete. Do not equate the unnamed models across
dates or infer linear ensemble cost from these figures.

## Runtime leads

- Separate loading/decoding cost from model execution. A loading-only notebook
  was suggested to estimate the data-access floor, but no result was reported.
- Metadata-based series/slice selection and lower-level pixel loading were
  suggested as ways to avoid decoding redundant data. Skipping over 60% of
  slices was claimed to be possible, without a supplied selection policy or
  controlled score comparison.
- Reduced resolution and window counts were discussed as speed/score tradeoffs;
  see [data preparation](../data-preparation/unverified-preprocessing-notes.md#resolution-tradeoffs) and
  [window sampling](../modeling/unverified-modeling-notes.md#window-sampling) for the actual reported settings.
- Ensemble distillation into a small model, fixed-shape compilation, parallel
  model execution, and specialized kernels were proposed. Large projected
  speedups and TPU advantages were not measured.
- Some submissions were recorded as failing after additional checks were
  introduced. Their conditions are unknown, so this does not justify
  removing correctness checks indiscriminately.

## Training throughput observations

Targeted benchmarking reportedly raised throughput on the large model from
1.8 to 3.4 studies/second. A small model improved by about 25% in a subsequent
report. An unspecified fusion change reduced GPU memory use from 19 GB to
12 GB; the exact model and changed operations were not identified.

These observations support investigating iteration cost, but they are not
reproducible optimization recipes. There is no before/after code,
batch sizes, numerical-equivalence checks, or stable hardware conditions.
Reported training durations and backbone choices are in
[modeling](../modeling/unverified-modeling-notes.md#unverified-model-structure).

## Remaining efficiency questions and historical context

The [efficiency leaderboard notebook](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)
combines score and execution time.

Earlier notes did not establish the rules or whether efficiency eligibility
depends on selected submissions. A fast model and ensemble were considered as
a way to test that hypothesis, without a definitive result.
The [official requirements above](submission-requirements-and-output-contract.md#verified-submission-requirements) supersede
that uncertainty for final prize eligibility.

TPU eligibility, hardware normalization, and medal eligibility were not resolved
by the sources reviewed. An unverified claim that another entrant achieved
0.948 in about five minutes lacks a linked run. Verify
these details on the official competition pages before choosing a strategy.
