# 2026-09-28 public forum and notebook re-sweep

Checked on **2026-09-28**. This sweep records only what is new since the
[2026-09-20 re-sweep](public-research.md#the-2026-09-20-re-sweep) and the
[2026-09-27 label investigation](../../public-labels/audits/2026-09-27-public-label-investigation.md);
items that repeat existing notes are left out. The method is in [provenance](../provenance.md).
Every score below is an author's report or a public leaderboard (LB) reading, not a reproduction.

## Coverage

- Listed all 139 forum topics and read the 23 active since 2026-09-20. For older threads, only
  comments posted from 2026-09-20 onwards were read.
- Listed notebooks by date, by votes and by public score. Downloaded 18; the four cited below
  are pinned in the [notebook manifest](public-notebook-sources.json).
- Listed datasets updated since 2026-09-26. The only new label table,
  `narisettichaitanya/rsna-knee-teacher-soft-labels`, is already screened in the
  [label rebuild](../../public-labels/label-rebuild/INDEX.md).
- Pulled the public leaderboard: 4,520 teams.

## The board

| Public LB at or above | Teams |
|---|---:|
| 0.960 | 1 |
| 0.955 | 25 |
| 0.950 | 78 |
| 0.948 (our best, S67) | 102 |
| 0.945 | 184 |
| 0.943 | 747 |
| 0.941 | 1,258 |

The top moved from 0.957 on 09-20 to **0.960**. The public notebooks reached **0.945**, up from
0.942–0.943 on 09-20. Notebook titles quoting "0.957" report a full-set score on the 58 gold
studies, recorded as `local_fullset_diagnostic_macro_auc` in the notebook metadata; they score
0.943 on the LB.

## Public 0.945: the public stack rank-blended with a private model

**Inspected code; LB scores from Kaggle's score-sorted notebook listing.** The two top public
notebooks are the 0.943 public stack with one private model blended in percentile-rank space:

- **[pjmathematician d4-lite](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-lite)**
  scored **0.945**. Its final cells run a private student ("S_dist", 36 windows) from
  unlisted datasets. The blend is `0.55 · rank(public) + 0.45 · rank(private)`, then ranked again,
  with one global weight. The code comments put the per-label rank correlation between the private
  models and the public stack at 0.75–0.89. A derivative notebook names its parent,
  `pjmathematician/rsna-knee-d4-blend`, at 0.946.
- **[aastikrajan15 knee-s75-w50](https://www.kaggle.com/code/aastikrajan15/knee-s75-w50)** scored
  **0.945** at private weight 0.5, and its w60 sibling also scored 0.945 at 0.6. Its private leg is
  ten ConvNeXt-Tiny checkpoints, two per fold, at 320 px over 24 slices.
- **Forum:** Prateek Grover reports a CoAtNet-2 at 352 px scoring 0.940 as a single fold and 0.942
  across four folds, and **0.945** blended with the public stack
  ([thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)).

In all three, a +0.002 LB gain over the stack came from blending an independently trained model at
roughly equal weight, with the private model scoring at or below the stack on its own. Our best
single model (0.948) sits above the stack. The endgame entry is
[public-stack blend](../../backlog/untried-ideas.md#ensembles).

**Open rules question:** a thread asks whether a submission that uses publicly shared weights trained
on the competition data by other participants is prize-eligible, given that winners must deliver
training code ([744056](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744056)).
It had no host answer on 2026-09-28.

## Measurements embedded in the s75 private model

**Inspected code comments** in the base64-embedded inference script of `knee-s75-w50`. These are the
author's out-of-fold measurements on their own ConvNeXt pipeline; none is replicated.

- **Per-target ensemble weights:** the protocol fits on half the studies and scores the other half
  (50 repeats × 2 directions × 5 folds, 84 out-of-fold files). A flat mean scored 0.8988; per-target
  top-3 members scored **0.9022**, with an estimated optimism of +0.0041; per-target greedy selection
  scored 0.9024 (+0.0078). The gain was positive in 5/5 folds and in 493/500 split-half units. Most
  of it came from MCL (+0.0163), Synovitis (+0.0068) and ACL (+0.0057). This disagrees with our
  [loss-ensemble simulation](../../validation/results/labels-and-losses/2026-09-21-loss-ensemble-simulation.md), where
  subset search did not beat the plain average once selection optimism was removed; their member
  pool is larger.
- **Per-target pooling over test-time offset passes** (n=2,646, 3 folds): max over passes helped
  Lateral Meniscus (+0.0010) and Fracture (+0.0008); top-2 helped ACL and MCL (+0.0005 each); max
  hurt Contusion (−0.0010) and Baker's (−0.0009). It applies only to a model scored over several
  offset passes, which ours is not: ×4 is one denser pass.

## Single-model reports

Self-reported, without recipes or linked runs
([thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)):

- **Qwen3.5-2B vision-language model fine-tuned on the images:** LoRA on the language model trained
  jointly with the vision encoder, 384 px, **0.950** from a single fold, about 3.5 h of training.
  It is the first report of this model family on the images here.
- **Five-fold ResNet at 224 px: 0.954** (a participant ranked 34th).
- Chris Deotte: with pseudo-labels "every single model can match our best ensemble CV", so
  single-model claims may hide an ensemble through the labels.

## Pseudo-labels and out-of-fold label blending

Forum reports that bear on the
[out-of-fold self-distillation](../../validation/results/labels-and-losses/2026-10-04-two-family-oof-distillation.md) entry. The
evidence is mixed and anecdotal:

- **For:** Archit Konde (ranked 46th) says only truly out-of-fold predictions help, since in-fold
  ones "parrot the labels back", and that going heavier than 50/50 toward the model added more.
  Raymond Yuen (0.947) calls pseudo-labels his main lever. He trains teachers on several weak-label
  sources (public, API and local), blends 50/50 and says single and multiple teachers both work.
  He reports that expensive and cheap LLMs differ little (under $5 of API spend in total).
- **Against:** Nicolai Karcher saw a clear lift on gold and on local CV at 50/50 but none on the
  LB. Myo Min Htet saw the same with the same encoder for teacher and student. Our S50 was also a
  public tie.

## RadImageNet input format

Ziad Ahmed, fold 0, out-of-fold, n=1,102, weak-label macro AUC
([thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696)):

| Backbone | Input channels | AUC |
|---|---|---:|
| ImageNet ResNet-34 | 3 adjacent slices | 0.8152 |
| RadImageNet ResNet-50 | 3 adjacent slices | 0.8339 |
| RadImageNet ResNet-50 | centre slice repeated ×3 | **0.8542** |

The input format is worth +0.020 on its own. The author's explanation is that RadImageNet was
pretrained on single-channel images copied across RGB, so an adjacent-slice stack is out of
distribution for its first convolution. This is specific to RadImageNet; the
[unverified notes](../../modeling/unverified-modeling-notes.md) record repeated slices as the same or
worse on ImageNet backbones. It is a design detail for the
[RadImageNet backbone notes](../../modeling/backbones/resnet-radimagenet-backbones.md).

## Rules and logistics

- **OAI:** the host says external data is allowed only if it is accessible without institutional
  review or sign-off and is free
  ([741819](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819),
  2026-09-21); OAI's data use certification makes that a no in practice. A request for a yes/no on
  MRNet, SKM-TEA, fastMRI(+), OAI, KneeMRI and KneeXNet-2.5D
  ([743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416))
  was unanswered on 2026-09-28. We use no external image data.
- **Final selections:** both selected submissions count for both the main and efficiency tracks.
  This is a participant's answer (Chris Deotte), not the host's
  ([743356](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743356)).
- **Kaggle T4 queue:** several participants report GPU kernels stuck in `QUEUED` for hours since
  about 2026-09-20, including inference notebooks that need saving. One reports that re-pushing
  after 10–15 minutes works, while waiting does not
  ([743746](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743746)).
  See the [submission workflow](../../inference/kaggle-submission-workflow.md).
