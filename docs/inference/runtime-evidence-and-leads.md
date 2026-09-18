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
| 2026-09-05 | Single-fold single model, top of both boards | 0.949 | About 5 minutes, submit to scored |
| 2026-09-07 | CoAtNet-384 plus a five-fold DINO half, 94 windows per study | 0.937–0.938 | 9 hours, reported down to 29 minutes by 2026-09-07 |
| 2026-09-08 | Best single-fold model | 0.944 | About 1 hour |
| 2026-09-09 | Single fold | 0.941 | About 23 minutes |
| 2026-09-11 | Preprocessing moved onto four CPU threads behind a queue | Not stated | Preprocessing about 1 hour to 15 minutes |

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
by the sources reviewed. Verify these details on the official competition pages
before choosing a strategy.

The five-minute claim recurred on 2026-09-17: the same competitor repeats it for
a single-fold single model at 0.949, says read time is still 60–70% of that
submission, and now leads both the score and the efficiency boards at 0.957. It
is still a self-report without a linked run, but it is no longer isolated. Our
own 0.936 submission holds efficiency rank 16 of 3,819 scored teams.

## Techniques from the public two-T4 notebook (2026-09-18)

Downloaded source of [jiweiliu's fast 2×T4 inference notebook](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference)
(kernel `134262775`, [manifest](../research/public-notebook-sources.json)), a
re-engineering of the public 0.941 blend. Its timings are the author's, on a
dual-V100 proxy over 58- and 144-study gates, for a blend of about 30 members,
so they do not transfer to our single model; the mechanisms do.

- **Cap the cuDNN workspace before importing torch.** `CUDNN_CONV_WSCAP_DBG=1024`
  is set first, to keep eager convolution-search workspaces under the T4's
  16 GiB.
- **Scan DICOM headers once and reuse the order and spacing** across every arm
  and pass rather than re-reading them per model: 206–209 s to 179 s on one stage
  (13.5%) and 85.9 s to 78.3 s on another (8.8%), with identical predictions and
  273 reused series matching the original file order and spacing exactly.
- **Do not search the mounted competition directory for auxiliary files.** A
  recursive manifest search there walks every test DICOM.
- **Overlap decoding with scoring** (decode the next study block while the GPU
  scores the current one) and share one decoded window tensor between two arms
  that differ only by a flip, keeping both forwards.
