# 2026-09-22 calibrated judge: better ranking, too noisy to replace submissions

**A judge fixed in advance, built from a label source outside label construction, ranks past
submissions better than fold-0 macro AUC. Its seed noise, though, is as large as the differences
that label and weighting changes produce.** It cannot judge a label-construction campaign
without submissions. Label changes stay judged on the public leaderboard. The judge's compare mode
is recorded as a secondary, pre-registered prediction before each submission.

- **Script:** [`validate_calibrated_judge`](../../../scripts/adhoc/validate_calibrated_judge.py), CPU only,
  a few minutes; refuses an existing output directory. It prefers a run's
  `validation_predictions.csv.x1` when present, so the history is scored at its cohort's
  multiplier. Full history, then the compare mode against a reference:

  ```bash
  rtk proxy uv run --locked python -m scripts.adhoc.validate_calibrated_judge --output-dir outputs/analysis/calibrated-judge-new
  rtk proxy uv run --locked python -m scripts.adhoc.validate_calibrated_judge --output-dir outputs/analysis/calibrated-judge-compare --reference outputs/training/k_05 --candidate outputs/training/<run>
  ```
- **Artifacts:** `outputs/analysis/calibrated-judge-20260922/`.
- **Question it answers:** can a local metric evaluate training-label changes, which fold-0 macro
  AUC misranks ([class-balance exponent](2026-09-21-class-balance-exponent.md))?

## Definition

Each fold-0 cell gets an expected truth, and the judge is the fold-0 macro soft AUC against it: a
pair (i, j) counts with weight p_i(1 − p_j).

- **Gold studies:** the official label.
- **Every other cell:** the mean, over eligible sources, of P(gold positive | that source's report
  state). The probabilities are pooled across labels on the 58 gold studies.

The eligibility rules were fixed in code before any submission was scored. An eligible source:

- is not one of the nine consensus sources, Steven v4, Ramit or Pilkwang, all of which label
  construction may use;
- covers all 58 gold studies;
- has a calibrated gold AUC of at least 0.80;
- agrees with every construction source at Cohen kappa below 0.90;
- has no state holding more than 80% or less than 2% of silver cells.

Earlier in the same session, the Ramit and Pilkwang versions had been explored on the same history.
That exploration chose the form (soft AUC, pooled calibration), which is why those two are
reported here as diagnostics only.

| Source | Calibrated gold AUC | Closest construction source (kappa) | Eligible |
|---|---:|---|---|
| Hosseini JEV v2: positive 0.68, negative 0.09, unknown 0.12 | 0.819 | Pilkwang v2 (0.86) | **yes** |
| Pearse weak v2 | 0.762 | Yunus (0.60) | no: gold AUC |
| lvthanhxun Qwen state | 0.767 | Pilkwang v2 (0.51) | no: gold AUC, rare state |
| lvthanhxun rule state | 0.678 | Steven v4 (0.37) | no: gold AUC |
| Nekkon NLI, five bins | 0.530 | none | no: 89% of cells in one bin |

The frozen judge is therefore Hosseini's JEV v2 alone.
[Source screen](../../public-labels/2026-09-22-public-label-additions.md).

## Agreement with the leaderboard

"Pairs" counts submission pairs at least 0.002 apart publicly that the metric orders the same way.

| Cohort | Runs | Fold-0 macro AUC: τ, pairs | Frozen judge: τ, pairs | Pilkwang (diagnostic): τ, pairs |
|---|---:|---|---|---|
| Scanner split, combined labels, ×1 (k sweep, oblique, jitter) | 7 | −0.31, 1/11 | +0.10, 7/11 | +0.41, 9/11 |
| `all_5fold_v3`, ×1 (ordinary against balanced BCE) | 5 | −0.53, 2/7 | −0.53, 2/7 | +0.11, 4/7 |
| `all_5fold_v3`, ×2, every comparable submission | 20 | +0.44, 119/162 | +0.64, 138/162 | +0.65, 139/162 |
| The same, from the 09-13 plateau | 15 | +0.02, 38/79 | +0.36, 56/79 | +0.46, 60/79 |

The two past label changes:

- **Tonylin swap (16th of 20 publicly):** fold 0 put it 6th and the judge 14th. The judge puts it
  0.0012 below its peers' median, while fold 0 puts it 0.0011 above.
- **Combined labels against Steven v4:** on the 209 fold-0 studies both splits hold out, the judge
  puts both combined-label runs (0.7131, 0.7132) above all three Steven v4 runs (0.7101–0.7124).
  That is the leaderboard's direction, but the margins are 0.0007–0.0031.

## Noise

Across the three recorded same-recipe replicate groups (df = 5), the judge's seed SD is 0.0012,
against 0.0017 for fold-0 AUC. The judge's between-run spread is smaller still:

| Cohort | Fold-0 SD across runs | Judge SD across runs |
|---|---:|---|
| k sweep | 0.0034 | 0.0009 |
| Balanced BCE era | 0.0056 | 0.0021 |
| 20 older submissions | 0.0043 | 0.0027 |
| Plateau | 0.0014 | 0.0011 |

In the k sweep and on the plateau, the spread between runs is no larger than one seed's noise. A
paired difference has about 0.0017 of seed noise, so changes below about 0.003 on the judge cannot
be told apart from a seed. The label changes that moved the leaderboard 0.007–0.009 moved the judge
by only 0.001–0.003.

## Interpretation

- **Ranking:** the fixed judge removes fold 0's reversal on the k sweep and ranks the older and
  plateau submissions better. It does not fix the ordinary-against-balanced-BCE reversal.
- **Resolution:** it lacks the resolution to adopt or reject one label change from one fold-0 run.
  A label-construction autoresearch campaign judged by it would mostly select seed noise, so no
  such campaign is proposed.
- **Use:** before submitting a label or weighting change, run the compare mode against its
  reference and record the delta and interval. Over several submissions, that builds a
  prospective test of the judge.
- **Open (hypothesis):** for model changes, where construction sources raise no circularity
  concern, a judge on Pilkwang or Ramit states ranked history best. It was chosen after seeing that
  history, so testing it as a replacement for fold-0 AUC in model campaigns needs its own fixed
  definition and a prospective record.
