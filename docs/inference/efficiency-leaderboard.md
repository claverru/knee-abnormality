# Efficiency leaderboard

How the public [efficiency leaderboard](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)
picks the submission it shows for a team, what the metric prices one thousandth of AUC at, and what
its ordering reveals about hidden-test runtimes we cannot otherwise measure. The prize rules and the
official formula live in the
[submission requirements](submission-requirements-and-output-contract.md#verified-submission-requirements);
this document covers the leaderboard's measured behaviour.

Measured on 2026-09-19 against all 51 published versions of the notebook and the public leaderboard
snapshot of 2026-09-18T23:23. Versions 1–8 are empty; version 9 carries the first data
(2026-08-05). Version 51 was published 2026-09-18 22:21 and holds 3,879 teams.

## Your row is your fastest useful submission, not your best one

The leaderboard shows **one row per team, the submission minimizing the efficiency score over every
submission that team has made** — not the best public score, and not the latest. Verified:

- On version 45 our row was the 0.912 3×-images submission of 2026-09-11 16:52:22, while the 0.913
  2× submission of 16:51:58 already existed. Both predate that snapshot, so the lag below cannot
  explain it.
- Across consecutive versions, 409 teams' shown scores **decreased** and 388 shown dates moved
  **backwards**. Neither is possible if the row were the best score so far.
- In version 51, 1,130 of the 3,874 teams that join the public leaderboard show a score below their
  public best, and **none shows a score above it**.
- Replaying "argmin efficiency over the eligible submissions" against our own ledger reproduces our
  row in versions 46–51 exactly.

During the training period the leaderboard applies **no benchmark filter**: version 51 ranks scores
down to 0.447, below the 0.5 an all-constant `sample_submission.csv` earns. The eligibility
conditions in the official rules (a selected submission, above the benchmark) govern the private
prize, not this notebook.

## The data is a day old

Each version is published around 22:20 UTC and its data stops around 22:00 UTC **the previous
day** — one version per day, with 2026-09-04 missing. Version 51, published 2026-09-18 22:21, holds
no submission later than 2026-09-17 21:46:54.

A submission made after about 22:00 UTC therefore waits two nights before it can occupy the row.

## The exchange rate

`Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400`, minimized. Writing
`K = maxAUC − Benchmark` the first term is `−AUC / K`, so with the benchmark at 0.5 and the public
top at 0.957, `K = 0.457`:

| Quantity | Worth |
|---|---|
| 0.001 of AUC | 70.9 s of runtime |
| 0.003 of AUC | 212.7 s |
| One minute of runtime | 0.00085 of AUC |

Both `Benchmark` and `maxAUC` are private-leaderboard values nobody sees until the end, so the rate
is an order of magnitude, not a constant; a higher benchmark or a lower top score widens it toward
90 s.

**The leaderboard bounds `K` from below.** With every runtime inside `[0, 32400]`, a team ranked
above another can trail it in AUC by less than `K`. The deepest inversion in version 51 is rank 3665
at 0.492 standing above rank 3667 at 0.941, so `K > 0.449`: the benchmark cannot sit far above 0.5,
and the rate cannot be far below 71 s. That inversion also requires the slower of the two to be
near the nine-hour cap.

At this rate the runtime term spans `[0, 1]` while the AUC term spans about `[0, 2.1]`, and every
submission of ours has run under twelve minutes, contributing under 0.023. The board is therefore an
accuracy ranking with a runtime tiebreak — but the tiebreak is large: in version 51 **834 teams hold
a better public score than our 0.936 while our row sits at rank 18**.

## Each pick is a measurement of our hidden runtimes

Kaggle never reports `RuntimeSeconds`; the API carries no runtime field for a submission. But the
row the leaderboard picks is an inequality on the runtimes it measured. Against the
submit-to-scored times recorded in `outputs/submission/records/*.json`:

| Snapshot | Chosen | Over | Implies | Recorded wall gap |
|---|---|---|---|---|
| v45 | 0.912, 3× | 0.913, 2× | t(2×) − t(3×) > +71 s | +365 s |
| v45 | 0.912, 3× | 0.910, 1× | t(1×) − t(3×) > −142 s | −270 s |
| v48 | 0.936, 09-14 | 0.931, 09-13 | t(0.936) − t(0.931) < 355 s | +272 s |
| v49 | 0.936, 09-15 21:28 | 0.936, 09-14 12:16 | strictly faster | −484 s |
| v49 | 0.936, 09-15 21:28 | 0.936, 09-15 08:27 | strictly faster | −393 s |
| v51 | 0.936, 09-15 | 0.933, 09-17 | t(0.933) − t(0.936) > −213 s | −60 s |

Two things follow.

**The 2026-09-11 density comparison is settled.** The 3× run really did execute faster than the 2×
run on the hidden test, by at least 71 s. The submit-to-scored waits pointed that way (1,691 s
against 2,056 s) but could not establish it; see
[image count and memory](inference-image-count-and-memory.md).

**Our submit-to-scored times carry at least 128 s of queue noise.** The 1×/3× row above is the one
inequality the recorded walls violate, and it misses by 128 s. Since one thousandth of AUC is worth
71 s, those waits cannot resolve a runtime difference that matters, and a difference under roughly
two minutes between two of our submissions is not measurable from them. The polling that produces
them is on a 30 s interval, so their own resolution is finer than their noise.

## Our appearances

| Version | Data cutoff | Rank | Row shown |
|---|---|---:|---|
| v45 | 2026-09-11 21:38 | 143 | 0.912, 09-11 16:52:22 |
| v46 | 2026-09-12 21:23 | 41 | 0.925, 09-12 09:37:29 |
| v47 | 2026-09-13 21:50 | 28 | 0.931, 09-13 17:27:29 |
| v48 | 2026-09-14 21:48 | 27 | 0.936, 09-14 12:16:32 |
| v49 | 2026-09-15 22:01 | 16 | 0.936, 09-15 21:28:08 |
| v50 | 2026-09-16 21:42 | 16 | unchanged |
| v51 | 2026-09-17 21:46 | 18 | unchanged |

Our 0.939 of 2026-09-18 06:28 postdates every published cutoff, which is why the row still reads
0.936. The [prizes page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/prizes)
awards the efficiency track's first three ranks $7,000, $6,000 and $5,000.

## Reading the leaderboard's history

The notebook only prints a CSV produced by a private `ryanholbrook/rsna-knee-abnormalities-efficiency-data`
kernel, which the API refuses. Its own published output is readable per version, including versions
older than the current one:

```python
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as client:
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug = "ryanholbrook"
    request.kernel_slug = "rsna-knee-abnormalities-efficiency-lb"
    request.file_path = "full_leaderboard.csv"
    request.version_number = 51
    data = client.kernels.kernels_api_client.download_kernel_output(request).content
```

Versions beyond the current one return 404, which is how the newest is found. The columns are
`EfficiencyRank`, `TeamName`, `PublicScore` and `DateSubmitted`; joining `TeamName` against
`kaggle competitions leaderboard --download` gives each team's public best for comparison. The
competition's own page text, which the web pages do not render, comes from `ListCompetitionPages`
as described in the [submission requirements](submission-requirements-and-output-contract.md).
