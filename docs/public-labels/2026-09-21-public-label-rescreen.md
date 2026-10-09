# 2026-09-21 public label re-screen and MCL definition

**No whole table is a better replacement for Steven v4. Two of its columns have better-supported
alternatives: Fracture and MCL.** For the other ten labels, Steven v4 is the best source or within
noise of it on the 58 radiologist-labelled studies. Stricter extractors over-correct Effusion and
PF OA, and no report-derived source recovers Synovitis.

Excluding Fracture, the best trustworthy whole-table gold gain over Steven v4 is +0.0025. That is
well inside the noise of 58 studies. The [earlier caution](2026-09-13-new-public-label-comparison.md)
still applies: a gold score alone does not justify a run.

## Per-label candidates to integrate

The comparison uses label agreement with the radiologist labels on the 58 gold studies. Seven
sources can be benchmarked this way: they neither copy nor fit the gold rows. The full grid is in
`per_label.csv`.

| Label | Steven v4 gold AUC | Alternatives | Verdict |
|---|---:|---|---|
| **Fracture** (18 positives) | 0.793, lowest of all seven | Pilkwang v2 0.871, Tonylin 0.885, JEV text 0.899, Yunus 0.875, Ramit 0.861, Josh 0.819 | **Candidate.** Every independent source orders Fracture better. Calls at 0.5 are similar (7/4 false positives/missed for Steven v4, 6/4 for Pilkwang v2), so the gain lies in the soft ordering. `flight0234/rsna-knee-hybrid-report-labels` is exactly Steven v4 with Pilkwang v2's Fracture column, and its author chose the swap before training. |
| **MCL** (9 positives) | 0.968; 7 false positives, 0 missed | Josh 0.989 (1/0), Yunus 0.984 (2/0), JEV text 0.977 (3/0) | **Candidate.** The rubric excludes sprains, and the strict sources drop Steven v4's false positives without missing any tears. See [below](#mcl-steven-v4-counts-sprains). |
| ACL | 0.987; 6/1 | Tonylin 0.998, Pilkwang 0.997; JEV text and Josh 1/1 | Hold. Stricter calls remove false positives but lower the AUC; every source scores above 0.96. |
| Medial and Lateral OA, PF OA | 0.932 / 0.833 / 0.902 | JEV text 0.924 / 0.878 / 0.912 | Hold. Strict sources trade false positives for 7–14 missed positives. |
| Effusion | 0.877; 16/0 | Strict sources 0.78–0.85, missing 12–21 | Keep Steven v4. The rubric's "moderate or large" reads stricter than the radiologists' actual calls. |
| Synovitis | 0.790; 4/14 | All 0.63–0.79, 14–20 missed | No text source helps. The positives are image findings that reports omit. |
| Menisci, Baker's, Contusion | 0.948 / 0.879 / 0.944 / 0.860 | Within about ±0.01, or worse | Keep Steven v4. |

Every figure here rests on 9–35 gold positives per label. JEV and Josh developed their prompts
against the gold set. The two candidates stand because independent sources agree, not because of
any single score.

## What was screened

On 2026-09-21, 23 label datasets were downloaded with the Kaggle CLI: every table listed since the
09-13 audit, plus earlier ones never scored. The API no longer exposes version numbers, so each
dataset's latest release is pinned by file hash. Snapshots are under
`data/labels/public/<owner>/<slug>/downloaded-20260921/`.

Script: [`screen_public_label_tables.py`](../../scripts/adhoc/screen_public_label_tables.py).
Output: `outputs/analysis/p/public-label-screen-20260921/`, containing `screen.csv`,
`per_label.csv`, `strict_mcl.json`, `strict_mcl_majority.csv` and `manifest.json` with every input hash.
It runs on CPU from the saved tables and the fold-0 predictions of control B and learned strength,
and refuses an existing output directory:

```bash
rtk proxy uv run --locked python -m scripts.adhoc.screen_public_label_tables --output-dir outputs/analysis/public-label-screen-new
```

Gold AUC is the label table's macro ROC-AUC on the official 58 studies. Steven v4 is rescored on
the same rows each time.

## Whole-table comparison

| Table | Gold macro | Steven, same rows | Without Fracture | Steven without Fracture | Note |
|---|---:|---:|---:|---:|---|
| `tsuyu122/rsna-knee-mri-balanced-labels-public` | 0.9124 | 0.8927 | 0.9109 | 0.9018 | Provenance unclear; per-label swings of −0.055 to +0.135 |
| `riadmohamed42/jev-knee-labels` (HYBRID) | 0.9045 | 0.8932 | 0.9045 | 0.9025 | Tuned on gold (see below) |
| `tonylin1026` soft (already trained on) | 0.9027 | 0.8927 | 0.9043 | 0.9018 | Cost 0.008 publicly when trained on |
| `flight0234` and `mtitoadhip` v4hybrid | 0.8991 | 0.8927 | 0.9018 | 0.9018 | Steven v4 with only Fracture changed |
| `yunusgmsoy/rsna-knee-llm-report-labels` | 0.8798 | 0.8927 | 0.8802 | 0.9018 | Strict MCL |
| JEV text ensemble (FINAL2) | 0.8745 | 0.8927 | 0.8723 | 0.9018 | Strict MCL |
| `zhen931` v8 | 0.8700 | 0.8932 | 0.8699 | 0.9025 | Scores identical to Pilkwang v2 on gold |
| `ramitnandi/rsna-knee-labels-gpt-luna` | 0.8569 | 0.8927 | 0.8565 | 0.9018 | Strict MCL |
| `joshuaziel` gpt-5.6-luna | 0.8297 | 0.8927 | 0.8307 | 0.9018 | Strict MCL; binary annotation × confidence |

About the sources in this table:

- **JEV HYBRID** takes each whole column from Steven v4 or Pilkwang v2, whichever scored higher on
  gold, and blends in a text ensemble weighted on 67 hand-checked studies. Its gold score is
  therefore optimistic.
- **tsuyu** merges MRNet and OAI studies with the RSNA ones. Its RSNA values look like model
  outputs, not report reads.
- **Without Fracture, no candidate leads Steven v4 by more than 0.009,** and that best case is
  tsuyu, whose provenance is unclear. The next best leads are +0.0025 (Tonylin) and +0.0020 (JEV
  HYBRID). The public analysis puts the paired standard deviation on the
  58 at about 0.0125 ([uncertainty](../research/public-research.md#validation-and-ensembling)).

## Tables that cannot be benchmarked

Nine tables copy the official labels into all 58 gold rows: Barun, Jezreal, Saidul, Niklita,
Noisyislands, Shingo, Yunus two- and three-source merges, and Zaid. Their gold AUC is exactly 1.
Copying gold does not disqualify them for training. Their silver columns, however, are merges
of Steven v4 and Pilkwang v2 with near-identical positive rates, so they add no new source.

Three tables omit the gold studies entirely: Dreaddevelopment, Freshtime and Vmohitrao.
Freshtime's silver MCL rate of 91% suggests a different score meaning.

Giovanni's probabilities score 0.992 on gold, but its own cross-validation is 0.615, so the gold
rows are in-sample fits.

## MCL: Steven v4 counts sprains

The [official rubric](../metadata/official-label-criteria.md) grades only high-grade acute tears as
positive, and "low-grade sprains and chronic or remote stress changes" as negative.

The gold set has 9 MCL positives. Calls at 0.5:

| Table | Called positive | False positives | Missed | Silver positive rate |
|---|---:|---:|---:|---:|
| Steven v4 | 16 | 7 | 0 | 15.2% |
| Pilkwang v2 and JEV HYBRID | 18 | 9 | 0 | 17.2% |
| Yunus LLM | 11 | 2 | 0 | 6.5% |
| JEV text | 12 | 3 | 0 | 5.3% |
| Josh gpt-5.6-luna | 10 | 1 | 0 | 4.3% |
| Ramit gpt-luna | 9 | 2 | 2 | 4.9% |
| Dreaddevelopment (no gold rows) | — | — | — | 4.9% |

The five strict sources support this without using labels:

- They were built separately, yet each one's silver positives almost all fall inside Steven v4's:
  182/187, 212/212, 226/231, 271/284 and 211/214.
- Pairwise Jaccard overlap between them is 0.52–0.74.
- Their majority vote marks **223 of 4,349** silver studies positive, against Steven v4's 663. For
  364 of Steven v4's positives, none of the five agrees.

With the gold studies set aside, fold 0 has 132 MCL positives under our split labels and 42 under
the strict majority. Scored against each, our saved predictions give:

| Run | MCL AUC, split labels | MCL AUC, strict majority |
|---|---:|---:|
| Control B | 0.8623 | 0.9026 |
| Learned strength | 0.8524 | 0.9020 |

This shows our local MCL metric, not label quality: the local metric counts about two thirds
of its positives on studies the strict extractors call sprains or negatives. Two cautions apply.
The gold confusion counts rest on 9 positives. At least two strict sources, JEV and the Josh
deathmatch, developed their prompts against the gold set.

Steven v4 also over-calls other labels on gold. Precision is 0.53 for Lateral OA and 0.57 for
Contusion, and it calls 16 false Effusions. No stricter consensus was checked for those labels.

## Implications

- An outside MCL above 0.9 is consistent with validating on rubric-strict labels. It is not
  evidence of an MCL model failure.
- The follow-up analysis picked four targets, not two, and replaced them with a consensus of
  independent sources rather than single-source swaps. The result is the
  [combined labels](combined-labels.md), in force since 2026-09-21.
