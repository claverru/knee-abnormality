# Documentation index

Working knowledge for the [RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

| Read when | Canonical document |
|---|---|
| Deciding where code belongs or how to keep competition experiments fast and reproducible | [Project architecture](architecture.md) |
| Finding a reusable script, locating an investigation, or adding and indexing a command | [Script index](../scripts/INDEX.md) |
| Checking active work or tasks planned for the short term | [Ongoing work](ongoing.md) |
| Reviewing reflection's best public score of 0.931 and its validation comparison with 3dsampling | [Reflection validation comparison](validation.md#2026-09-13-reflection-validation-comparison), [submission](inference-efficiency.md#reflection-submission) |
| Using study-level knee reflection, reviewing geometry eligibility, previews and augmentation-only CUDA costs | [Reflection modeling](modeling.md#study-level-knee-reflection), [geometry and evidence](data-preparation.md#reflection-geometry-and-eligibility) |
| Reviewing 3dsampling's 0.930 public result and validation comparison with the previous champion | [3dsampling validation comparison](validation.md#2026-09-13-3dsampling-validation-comparison) |
| Reviewing the completed center experiment, its rejected sampler and the frozen control for fusion | [Accurate-centers result](validation.md#2026-09-12-accurate-centers-result) |
| Training the spatial-token model with five embeddings and study-level 3D dropout | [Active spatial token model](modeling.md#active-spatial-token-model) |
| Reviewing the completed spatial-token run, its target gains and paired validation uncertainty | [Spatial-token validation comparison](validation.md#2026-09-13-spatial-token-validation-comparison) |
| Checking how local validation tracks public scores since the gold-balanced split | [Local and public agreement](validation.md#2026-09-13-local-and-public-agreement-after-the-split-change) |
| Reproducing the training-image geometry measurement behind the fixed 18-mm coverage radius | [Per-plane coverage radius](modeling.md#per-plane-coverage-radius) |
| Understanding why per-plane fusion resembles crop140mm and measuring its regional attention and omission effect | [Per-plane fusion usage diagnosis](validation.md#2026-09-13-per-plane-fusion-usage-diagnosis) |
| Comparing the completed max-pooling run with average pooling and interpreting its timing difference | [Max-pooling result](validation.md#2026-09-13-max-pooling-validation-comparison) |
| Following tensor shapes from triplet images through spatial tokens and final attention pooling | [Model shape diagram](../schema.md) |
| Measuring a trained spatial-token checkpoint's reliance on its encodings | [Manual encoding diagnosis](validation.md#manual-encoding-usage-diagnosis) |
| Finding removed regional investigation sources and original reproduction bundles | [Historical regional sources](history/regional-sources/README.md) |
| Reviewing the earlier spatial-fusion implementation and failed deployment runtime gate | [Spatial-fusion engineering result](validation.md#2026-09-12-spatial-fusion-engineering-result) |
| Checking the native DICOM reader's layout limits, preprocessing package, equivalence evidence and performance gates | [Production DICOM integration](data-preparation.md#production-dicom-integration) |
| Understanding the implemented physical sampler, overlap, fallback and its metadata evidence | [Physical sampling analysis](physical-sampling.md) |
| Recording or reviewing ideas to explore, including baseline, physical-coverage and resolution experiments | [Future work](future-work.md) |
| Comparing public notebook implementations and reported successes, failures, and uncertainties | [Public discussion and notebook findings](public-research.md) |
| Applying lessons from past RSNA and other MRI competitions, papers, and public code | [Research across competitions](research/README.md) |
| Setting up Python or matching dependencies to Kaggle's GPU image | [Environment and dependency versions](environment.md) |
| Downloading small competition files or checking schemas, report normalization, series profiles, image counts and median slice limits | [Competition metadata](metadata.md) |
| Downloading/preparing MRI images, previewing augmentations, or evaluating loader threading | [Data preparation](data-preparation.md) |
| Checking reference-label coverage, choosing inferred labels, or refining supervision | [Supervision](supervision.md) |
| Reproducing the completed Tonylin experiment with our split and frozen validation targets | [Training with Tonylin](supervision.md#training-with-tonylin-soft-labels) |
| Reviewing why Steven v4 was retained after the Tonylin public-score regression | [Tonylin label result](validation.md#2026-09-13-tonylin-label-comparison) |
| Choosing public label files or checking their coverage, uncertainty semantics, provenance, and measured reference agreement | [Public label audit](public-label-audit.md) |
| Comparing newer public labels with Steven v4 and checking published gold-score claims | [September 13 label comparison](public-label-audit.md#2026-09-13-new-public-label-comparison) |
| Choosing a backbone, sampling windows, augmentation/losses, or diagnosing batch memory and profiled runtime settings | [Modeling](modeling.md) |
| Using fixed folds, scoring OOF predictions, or interpreting local results | [Validation and experiments](validation.md) |
| Reviewing baseline1's measured results, confirmed code defects, or historical limitations | [Baseline1 review](baseline-review.md) |
| Preparing or submitting a Kaggle notebook from the local CLI, reviewing inference gaps, or investigating runtime and the efficiency prize | [Inference and efficiency](inference-efficiency.md) |
| Matching Kaggle's Lightning package locally | [Environment and dependency versions](environment.md) |
| Running training-to-submission recipes or preparing submissions quickly | [Justfile recipes](inference-efficiency.md#justfile-recipes) |

## Provenance

Some competition strategy sections contain unverified working notes gathered
during early project planning. They retain useful hypotheses, unsuccessful
attempts, and changing conclusions, but lack the complete configurations,
checkpoints, and run artifacts needed for independent reproduction. Treat their
measurements as experiment leads rather than project results.

The [public-source investigation](public-research.md), checked on 2026-09-09,
has separate provenance: direct Kaggle discussion text, official pages, and
downloaded public notebook source. It distinguishes reported measurements from
inspected implementations and project proposals. It did not reproduce training
or submission scores. Notebook snapshot identities are recorded in its
[source manifest](public-notebook-sources.json).

The [public-label audit](public-label-audit.md), completed on 2026-09-10,
downloads and fingerprints the Steven Lee and Pilkwang label datasets plus
Steven v4's lixin73 dependency. Its local Polars audit measures coverage,
source transformations, uncertainty and reference agreement, separately from
the earlier public experiment reports. It does not reproduce label generation
or establish downstream training results. Dataset versions and file hashes
are pinned in [the label-source manifest](public-label-sources.json).

The [September 13 label comparison](public-label-audit.md#2026-09-13-new-public-label-comparison)
adds six candidate publishers, reproduces available reference-score claims and
distinguishes copied gold targets from scoreable predictions. Its separate
[comparison manifest](public-label-comparison-sources.json) pins the inspected
releases. Its initial Tonylin recommendation was followed by a separate
[training comparison](validation.md#2026-09-13-tonylin-label-comparison):
public AUC fell from 0.929 to 0.921 despite higher local AUC, and the baseline
returned to Steven v4. Label-table agreement and image-model results remain
separate evidence.

The [broader competition research](research/README.md), reviewed on 2026-09-10,
has separate primary-source evidence from past RSNA and MRI competitions,
original papers, and author repositories. Its topic documents distinguish
reported results, inspected implementations, and proposed transfers to this
task. Four notebooks and four Python files were inspected without execution;
the [code manifest](research/source-manifest.json) pins their revisions and hashes.
This research does not establish new project experiment results or approve
external datasets for the current competition.

Sections explicitly marked as official or verified in the topic notes now
incorporate the 2026-09-09 public-source investigation. External links without
their own verification record remain research leads. Recheck volatile competition
requirements when preparing submissions or adding external data.

The environment document has separate provenance: direct host checks and
published Kaggle image metadata.

The metadata and reference-label sections also have independent evidence:
a Polars audit of five official competition CSVs downloaded on 2026-09-09.
[Competition metadata](metadata.md) records file fingerprints, reproduction
instructions, and limits; pixel and experiment claims were not verified by
that audit.

## How to use these notes

Use the unverified working notes to reach a useful baseline quickly and
prioritize experiments. Specific settings are candidates to reproduce and
improve, not project requirements. Apparent successes and failures can change
with the backbone, labels, sampling, or implementation.

1. Start from the agreed direction and open decisions in the
   [implemented baseline](modeling.md#implemented-baseline), using the reported [preprocessing](data-preparation.md#intensity-and-geometry)
   and [modeling evidence](modeling.md) as references when evaluating choices.
2. Use the fixed [validation protocol](validation.md#project-validation-protocol)
   and iterate on fold 0, retaining measured throughput and inference cost.
   Additional folds and ensembling are deferred until near the competition end.
3. Test promising changes such as [series-dropout ablations](modeling.md#augmentation-and-complementary-views)
   and [resolution choices](data-preparation.md#resolution-tradeoffs) against
   that baseline. Preserve useful gains without assuming the recorded choices
   are optimal for this project.
4. Use unsuccessful attempts to avoid repeating expensive searches blindly.
   Revisit them when there is a concrete reason the outcome could differ.

As project experiments produce evidence, update each canonical topic with the
validated setup and distinguish project results from unverified working notes.
