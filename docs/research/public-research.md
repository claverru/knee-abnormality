# Public discussion and notebook findings

Investigated on **2026-09-09**. The review recommended a small pretrained image
model with physically ordered adjacent-slice inputs, report-derived supervision
and evaluation isolated from label repair. That baseline now exists; geometry,
label handling and complementary models remain controlled comparisons. Public
evidence does **not** establish one universally best encoder, resolution, loss
or ensemble size.

For evidence from earlier RSNA and other MRI competitions, see the
[broader research collection](INDEX.md), reviewed on 2026-09-10.
It adds source-linked comparisons, inspected historical notebooks, external-data
compatibility notes, and experiment proposals. This document remains the
canonical review of public evidence from the current knee competition.

## Coverage and evidence strength

- Screened all 106 discussion titles returned by the competition forum and
  retrieved 26 relevant threads; reviewed experiment reports, corrections,
  host answers, and troubleshooting posts.
- **Re-swept on 2026-09-18** through the forum API and the Kaggle CLI: all 114
  topics screened, the 27 threads active since 2026-09-08 read in full, the
  notebook and dataset listings sorted by date and by votes, both leaderboards
  pulled, and one inference notebook downloaded. The findings are collected in
  [the 2026-09-18 section](#slice-density-and-the-board-2026-09-18-re-sweep).
- Inspected downloaded source for six public Kaggle code entries: five
  notebooks and one training script. Selection covered the popular baseline,
  higher-resolution DINO, large ensembles, and a different backbone family.
- Read the official data, evaluation, code requirements, efficiency, and rules
  pages through Kaggle's public page API. Direct browser extraction often
  returned empty pages; search-index extracts were supplemented with forum API
  text. Discussion attachments were not used for numerical conclusions.
- **Reported experiment** means an author's measurement, not our reproduction.
  **Inspected code** establishes an implementation, not its advertised score.
  **Project implication** means our inference to test. No training, inference,
  or submission was run, and attached datasets/checkpoints were not audited.

Scores below identify their endpoint: label extraction against the 58 supplied
reference studies, image-model validation against reference or inferred labels,
or public leaderboard (LB). These are different measurements. Related public
forks also share models and preprocessing, so their agreement is not independent
replication. Earlier unverified working notes remain clearly distinguished in
the [topic notes](../INDEX.md).

## Report-derived supervision

| Change or observation | Reported result | Interpretation and limits |
|---|---|---|
| [Steven Lee's label extraction and missing-label experiments](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932) | Extraction macro AUC against 58 gold studies: regex **0.8136**, LLM **0.8780**. About **25.4%** of target cells were undecided; Synovitis was **83.7%** undecided. | Preserve uncertainty separately from explicit absence. These are extractor scores, not image-model LB results. |
| Same experiment: fill only undecided Synovitis cells using Effusion | Synovitis AUC **0.678 → 0.790**; overall extraction AUC **0.8780 → 0.8873**. | A targeted repair candidate; explicit Synovitis statements were retained. The small reused gold set limits confidence in the gain. |
| Same experiment: learned imputation across all twelve targets | Overall **0.8805**, below the targeted **0.8873**. Baker's and Contusion worsened. | Missing mentions carry different information by target. Blanket correlation-based imputation did not outperform the targeted repair. The author also reports that improved labels initially failed to improve the image model. |
| [Comparison of public label tables](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737454) | Gold extraction AUC: Steven's v4 blend **0.893**, v2 **0.887**, Pilkwang `report_labels_v2` **0.870**. | Useful sources to compare, not a trustworthy ranking of small differences on 58 studies. Version names are those in the post, not a claim about today's latest files. |
| [Paired HARD versus SOFT replication](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105) | DINOv2, ten epochs, seeds **11/22/33**: SOFT won on gold in all three seeds; reported rank-ensemble gain **+0.0143**, paired 95% interval **[-0.0041, +0.0330]**. HARD won by **0.0132** on a separate 430-study binarized-lexicon endpoint. | Gold was excluded from training and checkpoint selection; duplicate reports were grouped. The gold interval is inconclusive, while the surrogate rewards reproducing HARD labels. The proposed explanation for SOFT's gain failed its pre-specified check. Earlier, larger ResNet-18 gains did not transfer quantitatively. |

Concrete public resources are
[Steven's report-label dataset](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)
and the labels referenced by [Pilkwang's baseline](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1).
Before training, record the exact dataset version/file, target order, coverage,
uncertainty encoding, and whether gold rows have been replaced with reference
labels. Published extraction AUC can otherwise be mistaken for independent
validation. The external label tables were not downloaded in this investigation;
the [local competition-file audit](../metadata/INDEX.md) covers official CSVs separately.

**Subsequent local verification, 2026-09-10:** the
[public-label audit](../public-labels/INDEX.md) downloads and pins these tables,
reproduces Steven's scores, and finds that Pilkwang's file covers **57** of the
58 reference studies. Its 0.870 therefore has a different denominator from
Steven's full-reference scores. The audit also reconstructs v2's Synovitis
repair and v4's blend, and documents why their values cannot share one generic
uncertainty rule. This is new local evidence beyond the investigation above.

The [bottleneck discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735826)
also reports report/image disagreements even for apparently clear severity
terms. Matching the official rubric is a better starting point than keyword
presence, but a confident interpretation of a report is not necessarily a
confident prediction of an independent image annotation. An image student can
outperform its report-label teacher; teacher AUC is not a mathematical ceiling.
See [the official label criteria and service clarification](../modeling/supervision.md#official-labeling-context).

## Geometry, sampling, encoders, and losses

| Experiment | Reported result | What to take from it |
|---|---|---|
| [EfficientNet-B0 ablations](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737597): four slots, weighted BCE, scanner-aware folds | OOF **0.7695 → 0.7777 → 0.7815** while jointly changing crop/resolution/sampling. Separately, nine central adjacent slices beat nine spread-out slices by **0.018** on fold 0. | The first progression confounds several variables. The isolated sampling result supports testing local slice continuity before adding more slices. The OOF label provenance is not fully specified in this post. |
| Same B0 experiment: EMA, Mixup, longer training, asymmetric loss | Fold-0 baseline **0.7914**; EMA **0.7940**, Mixup **0.7937**, 60 epochs **0.7922**, asymmetric loss **0.6526**. | Small gains were judged noise by the author; the loss change was strongly negative in this implementation. Repeated-seed evidence and the full asymmetric-loss configuration were not supplied. Do not generalize to every backbone or loss implementation. |
| Same B0 post: laterality | Side-dependent targets had a smaller left/right AUC gap than the other targets. | This is a subgroup diagnostic, not an experiment comparing orientation normalization on/off. It does not settle whether anatomically appropriate normalization or flipping helps. |
| [DINOv2 Small → Base](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154) | On 2,652 report-labeled validation studies across three folds at 224 px: **0.7931 → 0.7942**, versus reported seed variation **0.0020**, for **3.9×** step compute. | No resolved aggregate benefit in that setup. It does not prove encoder capacity never helps on reference labels or at another resolution. |
| Same encoder study: apparent RAD-DINO failure | The losing run used ImageNet normalization instead of the checkpoint's grayscale statistics; a corrected rerun was pending in the retrieved post. | This negative result cannot establish that medical pretraining fails. Log the loaded checkpoint and its preprocessing; the post also found an unintended ResNet-18 fallback when weights were missing. |
| [Synovitis and medical pretraining](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737566) | One participant reports Synovitis AUC **0.78** with RadImageNet versus **0.62–0.72** across ImageNet models. | A complementary-model lead, not a controlled general ranking: checkpoint/split/seed details are missing. RadImageNet and RAD-DINO are distinct experiments. |
| [Recent negative-results comment](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740162#3522386) | A commenter reports DINOv3 underperforming DINOv2 and report-based auxiliary loss hurting despite weight tuning; no deltas/configurations provided. | Low-confidence negative evidence. Do not assume newer pretraining or an auxiliary report objective must help. |

Multiple posters identify filename-based slice ordering as a concrete bug.
The [ResNet-34 report](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738339)
describes replacing UID order with physical-position projection and fixing
severity interpretation. The projection is
`dot(ImagePositionPatient, cross(row_direction, column_direction))`, with
directions from `ImageOrientationPatient`. Adjacent triplets require consistent
series orientation and grouping; merely sorting filenames does not provide it.

**Project implication:** compare 224/288/336-or-384 px while holding physical
field of view and sampled anatomy fixed. Separately compare the number and
placement of triplets. Public sources use roughly 130–150 mm crops, but these
are candidates, not an established optimum. Keep missing-slot masks and verify
what a thin structure looks like after the complete transform. Current
[preparation notes](../data-preparation/INDEX.md) and [modeling notes](../modeling/INDEX.md) retain
the alternative unverified recipe.

## Slice density and the board (2026-09-18 re-sweep)

Public LB top is **0.957**, 21 teams stand above 0.950 and 124 above 0.941 — the
plateau of the forked public notebook. Our best is 0.939, and our 0.936
submission holds efficiency rank 16 of the 3,819 scored teams.

### Slice density, not span

The Raptor author published each corpus' geometry with matched public scores
([thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696)).
One architecture (CoAtNet-384), one label set and one windowing code throughout;
the corpora are percentage-span resamplings of every series, at a 140 mm crop
stored at 336 px.

| Corpus | Public |
|---|---:|
| 44 slices over 6–94% of each series | 0.926 |
| 44 slices over 2–98% | 0.917 |
| 64 slices over 2–98% | 0.928 |
| 80 slices over 2–98% (22 sagittal-fluid, 18 sagittal, 15 coronal-fluid, 10 coronal, 15 axial) | 0.932 |

- **Widening the reach at a fixed slice count cost 0.009**, and restoring density
  by adding slices recovered it and more. The author's summary is that density is
  the lever rather than span, with **1.20% of the stack per slice** the finest and
  the best measured.
- **Evaluation windows are a separate axis.** The same weights on the 64-slice
  corpus read **0.924 at 42 windows and 0.927 at 62**. The 80-slice corpus needs
  78 windows for full coverage and its 0.932 is not available at 62, where a fifth
  of the volume goes unread; a checkpoint swap and a coverage change then stop
  being separable.
- **SWA was a null** for the same author: 0.9167 against 0.9150 on the 58 gold
  studies, nothing on average over six pairs, and 0.938 against 0.937 on the board.
- **Limits:** unreplicated single submissions 0.001–0.009 apart on a board that
  displays three decimals, and window counts are not held constant across every
  row. The geometry is not ours either: these corpora resample a fixed percentage
  of each series, while our sampler places triplets by physical distance from the
  study centre. The transferable claim is the direction — more samples at the same
  reach beat the same samples spread wider — not the numbers.
- **Project implication:** it agrees with our own measurements (20% fewer slices
  −0.0041 and 35% fewer −0.0051, [runs](../validation/results/2026-09-16-restored-code-hyperparameter-runs.md);
  sigma 60, 80 and 120 all worse than 40) and points at the one untested
  direction, denser per-type budgets at the current reach, filed in the
  [overnight candidates](../backlog/overnight.md#candidates-for-later-queues). The
  window-count axis is our evaluation multiplier, open above ×4 in
  [endgame](../backlog/endgame.md#test-time-augmentation).

### Single-model scores at the top of the board

- A **single-fold CoAtNet at 224 px reached 0.950**. The author attributes it to
  label work begun after noticing that their OOF predictions beat the extracted
  training labels on the 58 gold studies.
- The team now leading both boards reports **0.949 from one single-fold model
  scoring in about five minutes**, labels extracted locally with Gemma, about
  0.930 on gold ([thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)).
- Other reports: CoAtNet-384 single fold 0.937, CoAtNet-288 full-train 0.938,
  CoAtNet-224 0.930. Two participants above 0.94 state that resolution is not a
  major driver at 224–288 px, and one puts the **field-of-view sweet spot at
  140–160 mm** against 100 mm. Our recipe sits inside that window (224 px at
  0.625 mm/px is 140 mm).
- The most-upvoted advice in the [plateau thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740610)
  is to rank changes on report-label CV rather than on gold or the board, with a
  **0.003 threshold** for a genuine improvement — the same floor our replicate
  spread gives ([state](../ongoing/state.md)).
- That thread also reports two more nulls from a 20-checkpoint DINOv2 pipeline at
  0.903: explicit 2×2 spatial region tokens appended to the label-query attention
  (+0.000003 over two folds) and swapping in another public label table.
- These are unreplicated self-reports with incomplete recipes. What they establish
  is that single models in the 0.94–0.95 range exist on several backbones, not how
  they were obtained.

## Validation and ensembling

| Evidence | Result or warning | Project implication |
|---|---|---|
| [Metadata-only shortcut probe](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517) | HistGradientBoosting against report-derived labels: about **0.652** random-fold macro AUC versus **0.598** with 265 scanner/software fingerprints grouped. Series composition alone scored **0.5954**. | Metadata did not explain high image-model LB scores. Grouped evaluation is a useful robustness check, but scanner fingerprints are not hospital identities, and part of the signal can be reporting style. |
| [58-study uncertainty analysis](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733876) | Simulation estimated paired macro-AUC standard deviation **0.0125**, strongly dependent on correlation between models. | This is model-based uncertainty, not a universal significance threshold. It argues against ranking tiny gold-set differences. The post's recommendation to rank on weak labels alone is qualified by the HARD/SOFT endpoint reversal above. |
| [OOF versus LB experiences](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736635) | Several participants see LB above local scores; others report gold validation tracking LB better than weak-label validation. | There is no established conversion from local AUC to LB. The [site-split question](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734681) had no host answer in the retrieved thread. |
| [Single-model score thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304) | Reports include **0.936** for a small single-fold ResNet at 224 px and **0.942**, or **0.943** with TTA, at 288 px. Another participant explicitly corrected **0.947** to a five-fold ensemble after previously reporting a single-fold result. | These are unreplicated claims with incomplete recipes, but they undermine the assumption that a 20-model ensemble is required. “Single model” sometimes means one architecture across several folds; preserve the distinction. |
| [Blend-weight sweep](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740439) | Public source **0.936**, own CoAtNet ensemble **0.934**, blend **0.937** across weights 0.50–0.75. Several blend operators and adaptive weights still displayed **0.937**. | A displayed tie does not imply identical predictions or exactly equal AUC. Many public-LB trials cannot establish which small variation generalizes; prioritize an independent local comparison. |
| [One-fold versus five-fold anecdote](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740375) | ConvNeXt-Tiny: **0.939** for one model's last epoch versus **0.938** for a five-fold ensemble selected by validation AUC. | Fold count and checkpoint selection both changed. This does not demonstrate that ensembling hurts. |

For this project, preserve fixed study/report groups, an untouched reference
evaluation endpoint, and paired predictions with uncertainty estimates. Use
inferred-label validation for training diagnostics and a scanner-grouped stress
test alongside the primary split. Neither substitutes for reference labels.
Do not repair the targets used to evaluate the repair model. Keep checkpoint
selection separate from final comparison, and evaluate ensemble additions by
their incremental benefit and runtime, not standalone rank. These are proposed
controls. The [implemented protocol](../validation/project-validation-protocol.md)
uses five all-study folds with normalized-report groups and frozen mixed
gold/report-derived targets. A separate untouched reference endpoint and
scanner-grouped stress evaluation remain proposals beyond that protocol.

## Public code worth inspecting

Cell numbers below are **zero-based source-cell indices** in the downloaded
snapshots, not execution counters. The [source manifest](public-notebook-sources.json)
records full SHA-256 hashes and Kaggle kernel IDs. The API metadata did not
provide immutable source-version IDs; a future pull of the same URL may differ.

| Source | Useful implementation | Limit before reuse |
|---|---|---|
| [Pilkwang: RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | Cells 13, 19, 25–26: seed 2026, 130-mm crop, 336-px cache, three-slice groups; DINOv2 CLS/patch features and diagnosis-specific slot attention. Cell 32 discusses duplicate-report leakage and a fixed hash holdout. | Gold evaluation is limited to held-out gold rows; other gold rows can be trained on. Reconstruct actual membership rather than treating every gold score as an untouched 58-study test. Attached labels/weights remain dependencies. |
| [Will: DINOv2 at meniscus resolution](https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution) | Cells 2, 31–32: DINOv2-S/14, 336 px over 150 mm; five plane/contrast slots; mean/max ridge probe plus twelve-query attention, fixed 0.5 rank blend; GroupKFold by duplicate report text. | Author explicitly separates inferred-label OOF from the small gold check and claims no LB score. Useful as a readable baseline/validation reference, not evidence of a winning score. |
| [Mattia Angeli: Bend the Knee to DINOv3 (ensembled)](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-dinov3-ensembled) | Cells 1, 5, 13, 15: 20 DINOv2 members plus a fine-tuned DINOv3 arm; 130-mm crop, 336 px, central slice band; masked slots, per-target window pooling, rank aggregation and TTA. | Cell 0 warns about public-LB overfitting. Many inherited components are combined; a score change cannot isolate DINOv3's contribution. Model count has substantial inference cost. |
| [Roman Tamrazov: DINOsaur V4 train](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4-train) | Cells 2, 4, 6–7: six plane/contrast slots, 16 sampled slices, 130 mm/336 px, 1st–99th percentile normalization; frozen DINOv3 feature-cache head and several pooling alternatives. Code excludes held-out gold from pseudo-label training and contains nested specialist-weight selection. | Gold repetition, specialist fitting, OOF tables, caches and attached checkpoints require careful lineage checks. Source inspection does not validate the supplied OOF artifacts or prove any one pooling alternative better. |
| [Dread Development: training the twelve-finding model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model) | Script lines 4–69, 174–267, 368–420: CoAtNet, random adjacent-triplet bags, per-finding attention, denser evaluation, optional fold files and top-epoch weight averaging. | Requires precomputed slice stacks and an explicitly **unprovided label parquet**. Best-epoch selection uses the 58 gold studies, so that score is selection evidence. The header's **0.924** refers to the author's inference notebook, not a result reproduced here. |
| [prvsiyan: Head and shoulders, knees and toes](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes) | Cells 1, 12–14, 30–32 combine inherited DINO ensembles, DINOv3 folds, CoAtNet/Raptor views and residual arms through rank fusion. | Useful for studying ensemble plumbing and missing-slot handling; weak evidence for causal improvements. Numerous weight/cache inputs are needed. Cell 37 leaves final score unset until a real competition submission, and the source warns about inherited LB tuning. |

Start by reproducing one understandable branch. Pin its preprocessing, label
file, split, seed, checkpoint, and model identifier before borrowing more
components. Inspect the actual active configuration: public notebooks contain
alternative paths, inherited comments and later overrides. A title, source
comment, or best historical score does not establish the behavior of today's
downloaded source.

## Data access and runtime lessons

The [encoder/cache report](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154)
stores selected inputs as a memory-mapped `uint8` array:
`4407 × 6 × 9 × 224 × 224`, approximately **11.12 GiB**. This is a lossy,
model-specific selection of the corpus, not lossless compression of the original
DICOM dataset. Reusing decoded pixels removes repeated preparation work; it
does not remove the cost of decoding the hidden test set.

Public reports also identify truncated pixel payloads, missing codec support,
GPU/library incompatibility, and changing input mount layouts. Follow the
[DICOM handling evidence](../data-preparation/public-data-contract-and-decoding.md)
and [submission requirements](../inference/submission-requirements-and-output-contract.md#verified-submission-requirements)
before using a notebook's successful example run as evidence of submission readiness.
The three visible example test studies are replaced during scoring; their
displayed notebook runtime is not a hidden-test inference benchmark.

## Experiments to prioritize

The durable idea queue is the [backlog](../backlog/INDEX.md). This review
supplies evidence for those proposals rather than maintaining a second queue.

Revisit this review when hosts clarify unresolved scoring/site-split questions,
source notebooks or label tables change, or private-leaderboard solutions become
available. Current public results cannot establish the eventual winning recipe.
