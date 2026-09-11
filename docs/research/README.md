# Research across RSNA and MRI competitions

Use the implemented pretrained slice/window baseline and fixed validation
protocol when testing ideas from this review. The strongest additions to
investigate are better anatomy coverage, controlled missing-series robustness
and modest adaptation of pretrained features. Anatomy localization is promising,
but its benefit in previous competitions often depended on annotations
unavailable here.

These are research conclusions and experiment proposals, **not measured results
in this repository**. The current competition's
[public discussion and notebook review](../public-research.md) remains the
canonical account of knee-specific public evidence.

## Reading guide

| Read when | Document |
|---|---|
| Comparing the closest RSNA MRI tasks and their successful and unsuccessful approaches | [RSNA MRI competitions](rsna-mri.md) |
| Borrowing sequence modeling, crop supervision, or feature aggregation from volumetric CT competitions | [Transfer from RSNA CT](rsna-ct.md) |
| Understanding knee MRI classification, MRI domain shift, segmentation, and reconstruction challenges | [Other MRI benchmarks](mri-benchmarks.md) |
| Choosing external knee data or evaluating pretrained model ideas | [Pretraining and external data](pretraining-and-external-data.md) |
| Finding inspectable implementations and avoiding misleading notebook defaults | [Public code and notebooks](code-and-notebooks.md) |
| Choosing affordable, controlled experiments for this project | [Experiment priorities](experiment-priorities.md) |
| Recovering exact reviewed code versions | [Code source manifest](source-manifest.json) |

## What transfers, and what changes

| Evidence family | Relevant similarity | Main mismatch with this competition |
|---|---|---|
| RSNA lumbar spine, 2024 | Multiple MRI series, small structures, location-specific targets | Coordinate supervision and ordinal, weighted log-loss scoring |
| RSNA brain radiogenomics, 2021 | Several MRI contrasts, few subjects, unstable validation | Molecular status is a different prediction problem from visible structural abnormality |
| RSNA aneurysm, 2025 | Small findings, anatomical targets, heterogeneous MRI inputs | Also includes CTA; vessel masks and lesion points support a substantial 3D pipeline |
| MRNet, 2018–2019 | Knee MRI, report-derived training labels, expert reference evaluation | Three targets with different definitions; patient-separated splits and more reference cases |
| BraTS segmentation, 2021 | Multi-contrast MRI and reproducible architecture ablations | Dense masks and aligned volumes; segmentation metrics |
| M&Ms, 2020 | MRI scanner/protocol variation | Cardiac segmentation, rather than weakly supervised classification |
| fastMRI, 2019–2020 | Knee/brain MRI, preservation of small pathology | Reconstruction from raw k-space; image quality is not abnormality AUC |
| SKI10, 2010–2018 | Knee cartilage and bone anatomy | Segmentation endpoint; original data distribution has closed |
| RSNA hemorrhage, 2019; abdominal trauma, 2023 | Slice-to-volume aggregation and limited informative regions | CT intensities and stronger localization/slice supervision |

Sources and qualifications for each row are in the linked topic documents.
The review also includes fastMRI+ and SKM-TEA as related knee datasets, and a
2026 paper on adapting 2D foundation models to volumetric classification.
They are explicitly distinguished from competition-winning solutions.

For the current task, the [official-file audit](../metadata.md) establishes
4,407 training studies, only 58 with twelve reference labels. Reports are
training-only, and the [label rubric](../supervision.md#official-labeling-context)
matters as much as a disease name. A method that needs trustworthy masks, lesion
coordinates, or many clean labels has an additional cost here. Published scores
from the table above cannot be numerically compared with knee macro AUC.

## Evidence and coverage

The review was completed on **2026-09-10**, covering selected primary solution
discussions/write-ups, author repositories, public notebooks, organizer pages,
and original papers. It is a targeted review of transferable methods, not a
complete enumeration of every RSNA competition, forum post, or MRI paper.
Radiograph-only competitions received less attention because the volumetric and
multi-series problems are better matched by the sources above.

- **Author report:** a solution author's claimed score, ablation, or experience.
  It was not reproduced here. A winning ensemble does not isolate the value of
  each component, and multiple solutions from the same authors are not
  independent replications.
- **Paper result:** the authors' experiment on its stated dataset and endpoint.
  External evaluation and controlled ablations are stronger evidence than a
  leaderboard anecdote, but neither establishes transfer to this knee task.
- **Inspected implementation:** downloaded source establishes what the reviewed
  code contains. It does not authenticate attached weights, OOF arrays, reported
  scores, or the exact code used for a historical submission.
- **Project proposal:** our synthesis, to be evaluated locally before adoption.

Some Kaggle pages returned empty browser bodies. Indexed first-party text and
author-maintained repository versions supplied the cited content; partial
coverage is identified where it limits a claim. The public code review covers
four notebook files and four Python files, plus solution prose. Notebook source
cells are numbered from zero. Git commit IDs and downloaded-byte hashes are in
the manifest; external datasets and model weights were not downloaded or audited.

The current competition overview, data, and rules URLs were consulted but did
not expose readable bodies in this review. Current-task requirements therefore
retain their explicit **2026-09-09 verification date** in
[supervision](../supervision.md#official-labeling-context) and
[inference requirements](../inference-efficiency.md#verified-submission-requirements).
This review does not newly authorize any external dataset. Revisit source
availability and competition rulings before incorporating one.

## How this changes the research direction

The prior knee review already motivates good geometry, soft-label comparisons,
and efficient encoders. This broader review adds a stronger reason to compare
simple pooling with series-aware sequence aggregation, a concrete path toward
anatomy-guided pooling, and an MRI-specific robustness study. It also identifies
reusable public code and the annotation mismatches that could invalidate reuse.

The [experiment document](experiment-priorities.md) defines the comparisons and
their stopping conditions. [Future work](../future-work.md) is the canonical
idea queue; [ongoing work](../ongoing.md) tracks tasks actually underway.
