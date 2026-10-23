# Efficiency leaderboard

How the public [efficiency leaderboard](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)
picks the submission it shows for a team, what the metric prices one thousandth of AUC at, and what
its ordering reveals about hidden-test runtimes we cannot otherwise measure. The prize rules and the
official formula live in the
[submission requirements](submission-requirements-and-output-contract.md#verified-submission-requirements);
this document covers the leaderboard's measured behaviour.

## Your row is your most efficient selected submission

Only **selected** submissions count: the ones a team ticks on the Submissions page (two at most),
or, when it ticks none, the ones Kaggle selects automatically. The board shows the team's
selected submission with the lowest efficiency score, computed on the public test data. The
competition's *Efficiency Prize Evaluation* page states the eligibility rule (read through
`ListCompetitionPages` on 2026-10-19; the board notebook only prints a CSV from the private
`ryanholbrook/rsna-knee-abnormalities-efficiency-data` kernel and links to that page):

> To be eligible for an Efficiency Prize, a submission: Must be among the submissions selected by
> a team for the Leaderboard Prize, or else among those submissions automatically selected under
> the conditions described in the My Submissions tab. Must be ranked on the Private Leaderboard
> higher than the `sample_submission.csv` benchmark.

The same page says the board shows public-data efficiency during the competition and is recomputed
on the private data afterwards.

**Consequence: one of the two final selections must be a fast, strong submission**, or the team has
no competitive efficiency entry. Switching both selections to slow ensembles moves the row to a
slow submission even when faster, nearly as accurate ones exist.

How the board behaves under this rule, measured 2026-10-19 on versions 76–80:

- In v80 our row jumped from S167 (0.964, 6.5 min submit-to-score) to S170 (0.965, 35.8 min), and
  our rank fell from 1 to 57. Over all submissions that is impossible: S170's extra 0.001 is worth
  about 70 s at [the exchange rate](#the-exchange-rate), not 29 minutes, and adding a submission
  can never worsen a team's best score. The selection at v80's data cutoff (2026-10-17 20:51 UTC)
  held S170 and no fast submission.
- Other teams whose row moved to a new, higher-scoring submission collapsed the same way: codexyzj
  21 → 4130 and zzc 19 → 2467 (v78 → v79), Sergey Bryansky 25 → 801. Teams whose row did not change
  kept their relative order.

**Withdrawn reading.** The 2026-09-19 version of this document concluded that the row was the
minimum "over every submission that team has made" and that selection only governed the private
prize. Its evidence (rows below a team's public best, shown scores decreasing between versions) fits
the selection rule equally well, since teams change their selections, and its replay of our own row
ran before we had selected anything manually.

During the training period the board applies **no benchmark filter**: version 51 ranked scores down
to 0.447, below the 0.5 an all-constant `sample_submission.csv` earns.

## The data is a day old

Each version is published around 22:20 UTC and its data stops around 22:00 UTC **the previous
day**: one version per day, with 2026-09-04 missing. Version 51, published 2026-09-18 22:21, holds
no submission later than 2026-09-17 21:46:54.

A submission made after about 22:00 UTC therefore waits two nights before it can occupy the row.
A change of selection is presumably subject to the same lag; v80 still showed the selection of
2026-10-17 evening on 2026-10-19 morning.

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
90 s. With the public top at 0.969 (2026-10-19), `K = 0.469` and 0.001 is worth about 69 s.

**The leaderboard bounds `K` from below.** With every runtime inside `[0, 32400]`, a team ranked
above another can trail it in AUC by less than `K`. The deepest inversion in version 51 is rank 3665
at 0.492 standing above rank 3667 at 0.941, so `K > 0.449`: the benchmark cannot sit far above 0.5,
and the rate cannot be far below 71 s. That inversion also requires the slower of the two to be
near the nine-hour cap.

At this rate the runtime term spans `[0, 1]` while the AUC term spans about `[0, 2.1]`. The board is
therefore an accuracy ranking with a runtime tiebreak, but the tiebreak is large: in version 51,
**834 teams held a better public score than our 0.936 while our row sat at rank 18**, and in v80
a 0.965 ensemble at about 36 minutes ranked 57th behind many 0.945 rows.

## Each pick is a measurement of our hidden runtimes

Kaggle never reports `RuntimeSeconds`; the API carries no runtime field for a submission. But the
row the board picks is an inequality on the runtimes it measured, **between submissions that were
both selected at that snapshot**. Before we selected anything manually, Kaggle's automatic
selection applied, which takes the best public scores; an inequality below is only valid if both
submissions were among them. Against the submit-to-scored times recorded in
`outputs/submission/records/*.json`:

| Snapshot | Chosen | Over | Implies, if both were selected | Recorded wall gap |
|---|---|---|---|---|
| v45 | 0.912, 3× | 0.913, 2× | t(2×) − t(3×) > +71 s | +365 s |
| v45 | 0.912, 3× | 0.910, 1× | t(1×) − t(3×) > −142 s | −270 s |
| v48 | 0.936, 09-14 | 0.931, 09-13 | t(0.936) − t(0.931) < 355 s | +272 s |
| v49 | 0.936, 09-15 21:28 | 0.936, 09-14 12:16 | strictly faster | −484 s |
| v49 | 0.936, 09-15 21:28 | 0.936, 09-15 08:27 | strictly faster | −393 s |
| v51 | 0.936, 09-15 | 0.933, 09-17 | t(0.933) − t(0.936) > −213 s | −60 s |

**The 2026-09-11 density comparison.** In v45 the 0.913 2× and 0.912 3× runs were our two best
public scores, so automatic selection most likely held both, and the pick means the 3× run executed
faster than the 2× run by at least 71 s. The submit-to-scored waits pointed the same way (1,691 s
against 2,056 s); see [image count and memory](inference-image-count-and-memory.md).

**Queue noise in our submit-to-scored times is unmeasured.** The earlier claim of at least 128 s
rested on the 1×/3× row, but the 0.910 1× run was not among our two best scores at v45, so it was
probably not selected and that row says nothing. Queueing is still visible: S165's
submit-to-scored time was 44.8 min against 7.1 min for S156 with the same weights. Treat
differences of a few minutes between our recorded waits as unresolved.

## Our appearances

Data cutoffs are the newest submission time each version holds, in UTC. Wall times are
submit-to-scored minutes from our records.

| Version | Data cutoff | Rank | Row shown |
|---|---|---:|---|
| v45 | 2026-09-11 21:38 | 143 | 0.912, 09-11 16:52:22 |
| v46 | 2026-09-12 21:23 | 41 | 0.925, 09-12 09:37:29 |
| v47 | 2026-09-13 21:50 | 28 | 0.931, 09-13 17:27:29 |
| v48 | 2026-09-14 21:48 | 27 | 0.936, 09-14 12:16:32 |
| v49–v51 | to 2026-09-17 21:46 | 16–18 | 0.936, 09-15 21:28:08 |
| v52–v65 | to 2026-10-02 21:40 | 4–10 | 0.940–0.950 |
| v66–v70 | to 2026-10-07 21:35 | 3–4 | 0.954, then 0.957 |
| v71–v73 | to 2026-10-10 21:10 | 1 | 0.959, 10-08 10:46:51 |
| v74 | 2026-10-11 21:13 | 1 | S143, 0.960, 6.1 min, over the 0.961 teacher ensembles S140 and S141 |
| v75 | 2026-10-12 21:12 | 1 | S148, 0.962, 9.6 min |
| v76 | 2026-10-13 20:42 | 3 | S152, 0.963, 9.6 min |
| v77–v78 | to 2026-10-15 21:14 | 1 | S156, 0.964, 7.1 min |
| v79 | 2026-10-16 21:03 | 1 | S167, 0.964, 6.5 min; S165 (TensorRT, 0.964) was already submitted but not shown |
| v80 | 2026-10-17 20:51 | 57 | S170, 0.965, 35.8 min: the selection held no fast submission |
| v81 | 2026-10-18 20:51 | 111 | S165, 0.964 (TensorRT), picked from the selection S175 + S165; its measured runtime ranks far below S156's and S167's rank 1 at the same score, consistent with the 706 MB TensorRT runtime delaying the Kaggle session start (`outputs/profiling/kaggle-startup-s165-s156-20261019`) |

The [prizes page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/prizes)
awards the efficiency track's first three ranks $7,000, $6,000 and $5,000.

## Reading the leaderboard's history

The notebook only prints a CSV produced by the private
`ryanholbrook/rsna-knee-abnormalities-efficiency-data` kernel, which the API refuses. Its own
published output is readable per version, including versions older than the current one:

```python
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as client:
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug = "ryanholbrook"
    request.kernel_slug = "rsna-knee-abnormalities-efficiency-lb"
    request.file_path = "full_leaderboard.csv"
    request.version_number = 80
    data = client.kernels.kernels_api_client.download_kernel_output(request).content
```

Versions beyond the current one return 404, which is how the newest is found. The columns are
`EfficiencyRank`, `TeamName`, `PublicScore` and `DateSubmitted`; joining `TeamName` against
`kaggle competitions leaderboard --download` gives each team's public best for comparison. The
API does not expose which submissions are selected; that is only visible on the Submissions page.
The competition's own page text, including the *Efficiency Prize Evaluation* page, comes from
`ListCompetitionPages` as described in the
[submission requirements](submission-requirements-and-output-contract.md).
