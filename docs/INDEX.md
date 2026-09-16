# Documentation index

Working knowledge for the [RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

| Read when | Document |
|---|---|
| Checking active work or tasks planned for the short term: the next fold-0 runs and inference optimization | [Ongoing work](ongoing.md) |
| Recording, classifying or picking ideas not yet started: the next overnight hyperparameter queue, small code changes (attention pooling, EMA/SWA, supervision and augmentation tweaks, per-plane spacing), big bets (position and knee side, medial/lateral transfer, study centers) and end-of-competition work (ensembling, bigger images, TTA, distillation) | [Backlog](backlog/INDEX.md) |
| Deciding how far to trust unverified working notes and how to iterate from the baseline | [How to use these notes](#how-to-use-these-notes) |
| Deciding where code belongs, setting up Python and matching dependencies or Lightning to Kaggle's GPU image, or following the geometry contract, data flow, config and experiment-artifact rules | [Architecture](architecture/INDEX.md) |
| Finding a reusable script, locating an investigation, or adding and indexing a command | [Script index](../scripts/INDEX.md) |
| Downloading small competition files or checking CSV schemas, report normalization, series profiles, image counts or knee-side metadata | [Competition metadata](metadata/INDEX.md) |
| Downloading or preparing MRI images, DICOM reading, preprocessing profiles, loader threading, augmentation previews, reflection geometry, or the implemented physical sampler, its verification and metadata analysis | [Data preparation](data-preparation/INDEX.md) |
| Choosing public label files or checking their coverage, uncertainty semantics, reference agreement and newer label comparisons | [Public label audit](public-labels/INDEX.md) |
| Choosing or swapping a backbone, following tensor shapes, checking reference-label coverage and supervision (inferred labels, Tonylin setup), or understanding the current full-slice spatial-token model, its recipe and reproduction, inputs, schedule or reflection | [Modeling](modeling/INDEX.md) |
| Using fixed folds, scoring OOF predictions, comparing fold runs, diagnosing encoding use or finding a dated experiment result | [Validation and experiments](validation/INDEX.md) |
| Preparing or submitting a Kaggle notebook, running dual-GPU inference, finding dated submission records and scores, or investigating runtime and the efficiency prize | [Inference and efficiency](inference/INDEX.md) |
| Checking where evidence came from, comparing public notebooks and discussion findings, or applying lessons from past RSNA and other MRI competitions, papers and public code, or position, knee-side, optimizer and backbone research | [Research and evidence sources](research/INDEX.md) |
| Following the model lineage from baseline1 to the champion and the current recipe, retired variants such as crop selection, the token graph and attention prior, baseline1's results and code defects, the retired 1A–1C sharing experiments (generalization investigation, OA/meniscus disagreement audit) or the dropped crop-selection investigation, or recovering archived sources and plans for retired crop-selection, regional, sharing, knee-side and token-graph work | [History](history/INDEX.md) |
| Recovering the pinned public-label dataset versions and hashes used by training (a config input in `configs/labels/`) | [Label-source manifest](../configs/labels/public-label-sources.json) |
| Recovering the releases, column mappings and hashes pinned by the September 13 label comparison | [Label-comparison manifest](../configs/labels/public-label-comparison-sources.json) |
| Recovering the prepared Tonylin soft-label method and input identities | [Tonylin label manifest](../configs/labels/tonylin-label-sources.json) |

## How to use these notes

Use the unverified working notes ([provenance](research/provenance.md)) to reach a useful baseline
quickly and prioritize experiments. Specific settings are candidates to reproduce and improve, not
project requirements. Apparent successes and failures can change with the backbone, labels,
sampling, or implementation.

1. Start from the agreed direction and open decisions in the
   [current model](modeling/current-model.md), using the reported [preprocessing](data-preparation/unverified-preprocessing-notes.md#intensity-and-geometry)
   and [modeling evidence](modeling/INDEX.md) as references when evaluating choices.
2. Use the fixed [validation protocol](validation/project-validation-protocol.md)
   and iterate on fold 0, retaining measured throughput and inference cost.
   Additional folds, ensembling and bigger images are deferred until
   [near the competition end](backlog/endgame.md).
3. Test promising changes such as [series-dropout ablations](modeling/unverified-modeling-notes.md#augmentation-and-complementary-views)
   and [resolution choices](data-preparation/unverified-preprocessing-notes.md#resolution-tradeoffs) against
   that baseline. Preserve useful gains without assuming the recorded choices
   are optimal for this project.
4. Use unsuccessful attempts to avoid repeating expensive searches blindly.
   Revisit them when there is a concrete reason the outcome could differ.

As project experiments produce evidence, update each canonical topic with the
validated setup and distinguish project results from unverified working notes.
