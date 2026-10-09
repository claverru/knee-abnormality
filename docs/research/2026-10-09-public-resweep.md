# 2026-10-09 public forum and notebook re-sweep

Checked on **2026-10-09**. This sweep records only what is new since the
[2026-09-28 re-sweep](2026-09-28-public-resweep.md); the label tables published in the same
period are in [2026-10-09 public label additions](../public-labels/2026-10-09-public-label-additions.md).
The method is in [provenance](provenance.md). Every score below is an author's report or a public
leaderboard (LB) reading, not a reproduction.

**Nothing here changes the recipe.** No new labels, masks or regions of interest worth using were
published; the public notebooks moved to 0.951 only through one checkpoint trained with external
OAI data, whose eligibility the host has not ruled on; the host has answered no question since
09-28.

## Coverage

- Listed all 173 forum topics and read the 58 active since 2026-09-28: new threads in full, older
  threads only for comments from 09-28 onwards.
- Listed 760 notebooks by public score and by run date; a subagent pulled 43 and read them without
  running anything, decoding embedded scripts. The five cited below are pinned in the
  [notebook manifest](public-notebook-sources.json).
- Listed datasets updated since 2026-10-03.
- Pulled the public leaderboard at 10:24 UTC: 5,570 teams.

## The board

| Public LB at or above | Teams |
|---|---:|
| 0.964 (top) | 2 |
| 0.960 | 25 |
| 0.959 (our best, S126, rank 29) | 34 |
| 0.957 | 53 |
| 0.955 | 82 |
| 0.953 | 126 |
| 0.951 (best public notebook) | 230 |
| 0.950 | 675 |
| 0.945 | 995 |
| 0.943 | 1,871 |

The top moved from 0.960 to **0.964** and the public notebooks from 0.945 to **0.951**. The 445
teams at exactly 0.950 are forks of the notebooks below. The team-merge deadline is 2026-10-15 and
final selections close on 2026-10-22 (dates quoted by participants).

## Public 0.949–0.951: one checkpoint trained with OAI

**Inspected code and dataset metadata.** Every public notebook at 0.948 or above runs
`nartaa/rsna-knee-publication-swa-weights-20261007`, a CoAtNet whose README states it was trained
with 2,399 OAI knees as external supervision.

- **[nartaa 0949 anatomical mirror](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror)**,
  0.949 as one model, with a
  [write-up](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792).
  Author's LB table, each row one submission, rows not isolated:

  | Step | Public LB |
  |---|---:|
  | All-data CoAtNet, 80 slices, 78 windows | 0.939 |
  | 96 slices, native 384 px cache, 94 windows | 0.940 |
  | Blend of public labels and JEV labels | 0.942 |
  | OAI knees added (PF OA, Lateral OA and Synovitis supervised, other cells masked) at 224 px | 0.945 |
  | Same at 384 px, full field | 0.948 |
  | Mirror averaging at inference | 0.949 |

  Their negative results: 94 windows cut to 24, 0.940 → 0.934; width flip on every plane,
  0.940 → 0.939; a native crop of 304 or 288 px instead of 320, 0.948 → 0.947 and 0.946; one
  epoch instead of the three-epoch average, 0.948 → 0.947; a model retrained on the 320 px crop,
  0.949 → 0.946 despite better gold.
- **The mirror** uses no laterality: sagittal three-slice windows reverse their slice order,
  coronal and axial windows flip width, and the two probability vectors are averaged. Ours already
  averages a reflection ([current model](../modeling/current-model.md)).
- **Blends on top of it:**
  [heliosli blend-gold](https://www.kaggle.com/code/heliosli/rsna-knee-blend-gold), **0.951**, is
  `0.7 · rank(nartaa) + 0.3 · rank(own 4-fold ConvNeXt-Small)`;
  [versia-5](https://www.kaggle.com/code/evgendvorkin/rsna-versia-5) and its forks, 0.950, blend
  it per label with three public ConvNeXt-Tiny folds, weights hand-set from gold AUCs. A global
  60/40 blend with the old 0.943 stack scored 0.949
  ([goodpjw2008](https://www.kaggle.com/code/goodpjw2008/rsna-knee-0-949-coatnet-stack-blend-lb-0-949)).
  heliosli's own model also loads an SKM-TEA segmenter to centre its views on the joint.

**Decision relevance:** the [public-stack blend](../backlog/endgame.md#ensembling) entry would now
mean blending OAI-trained weights. It stays unusable until the host rules; the OAI-free public
stack is still at 0.943–0.944, below our single models.

## External data: still no ruling

- No host or admin comment has been posted on any thread since 2026-09-28. Open and repeatedly
  bumped: which external datasets are allowed
  ([743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416),
  80 votes), whether public weights trained on OAI may be used
  ([747455](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747455),
  [747660](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747660)),
  prize eligibility with other participants' public weights
  ([744056](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744056)),
  and the RadImageNet licence, CC BY-NC-SA on Kaggle against MIT on Hugging Face
  ([745283](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745283),
  [735121](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735121)).
  The last one concerns our backbone.
- **OAI access:** participants posted replies from the OAI team and NDA saying the data may be
  used; registration needs only an e-mail. Participants in China report being unable to
  register, which bears on the host's "accessible to all" test. Several strong participants
  (hengck23, Cody_Null) are downloading it; others (YYama, Tucker Arrants) stay with competition
  data. KneeCoT remains the only dataset explicitly prohibited.
- Whether every hidden test study has all three planes and a fluid-sensitive series is asked and
  unanswered
  ([744519](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744519)).

## Single-model reports

Self-reported in the
[best single-model thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304),
without recipes:

- colum2131: ConvNeXt-Tiny 2.5D **0.953**, DINOv2 ViT-S 2.5D 0.951, the two blended 0.955; both
  trained on pseudo-labels, no external data.
- Tanuki_boosting: one model, 5 folds, **0.955**. CoolinLai's 0.954 five-fold ResNet at 224 px is
  a ResNet-50. Musa Tur Farazi: 0.952 at 224 px. Tucker Arrants: most models in the low 0.95s
  with a plain Qwen extraction.
- SpeedSci: changing only the labels moved a DINOv2 from 0.931 to 0.942.

Our ResNet-50 at 0.959 is above every single-model figure reported in public.

## Labels

- **Effusion and Synovitis on gold (colum2131,
  [745949](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745949)):**
  Synovitis not mentioned in the report, 11 of 36 gold-positive; Effusion reported absent, 10 of
  26 positive; Effusion reported small or equivocal, 14 of 18 positive. An LLM given report and
  images raised gold Effusion AUC from 0.835 to about 0.92 and Synovitis to about 0.85; an image
  model trained on those pseudo-labels scored 0.986 on Effusion and 0.859 on Synovitis. This
  matches [gold versus silver](../public-labels/gold-versus-silver.md); the small-effusion rate is
  higher than our 15 of 28.
- **Label quality is not separating teams:** a bootstrap on gold finds no public set clearly
  better than the median on 11 of 12 findings
  ([747540](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747540));
  Tucker Arrants says the same from LB experience. A gold report contradicts its gold label
  (report "no knee effusion", label Effusion 1), posted by hengck23.
- **Ideas floated without results:** show a multimodal LLM the report and the images and ask for
  the region of interest (hengck23), atomic question-and-answer targets from a reading rubric
  (hengck23), report-state auxiliary classes (in heliosli's code). The last two are what our
  [report attribute extraction](../ongoing/report-attribute-extraction.md) already does. No such
  labels were published.

## Other notes

- **bfloat16 on T4 or P100:** autocast to bf16 is emulated and about 50× slower; a notebook with a
  time guard then filled half the test with 0.5 and scored 0.710 instead of 0.851
  ([744230](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744230)).
  Ours uses FP16.
- **Kaggle faults:** `ERRORED_MOUNTING_DATASET` failures on notebooks and submissions since about
  10-04; a failed submission of this kind does not use up an attempt
  ([746542](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746542)).
  See the [submission workflow](../inference/kaggle-submission-workflow.md).
- **Slice order in a shared baseline:** sorting by file name instead of position costs 0.028 on
  one fold
  ([745759](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745759)).
  Ours sorts by position.
- **Speed reference:** [hengck23 MaViT](https://www.kaggle.com/code/hengck23/mavit-288-baseline-01-1-5sec-study-1gpu),
  0.935 at 1.5 s per study on one GPU; nartaa's 224 px one-view model, 0.945.
