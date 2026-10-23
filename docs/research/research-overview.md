# Research across RSNA and MRI competitions

The review (completed **2026-09-10**) found that better anatomy coverage,
controlled missing-series robustness and modest adaptation of pretrained
features were the strongest additions to test on the pretrained slice/window
baseline. Anatomy localization looked promising, but its benefit in previous
competitions often depended on annotations unavailable here.

These are research conclusions and transfer proposals, **not measured results
in this repository**; the project's own results are in the validation results.
The current competition's
[public discussion and notebook review](competition-and-prior-art/public-research.md) remains the
canonical account of knee-specific public evidence.

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

For the current task, the [official-file audit](../metadata/INDEX.md) establishes
4,407 training studies, only 58 with twelve reference labels. Reports are
training-only, and the [label rubric](../modeling/supervision.md#official-labeling-context)
matters as much as a disease name. A method that needs trustworthy masks, lesion
coordinates, or many clean labels has an additional cost here. Published scores
from the table above cannot be numerically compared with knee macro AUC.

## Direction the review gave

The prior knee review motivated good geometry, soft-label comparisons and
efficient encoders. The broader review added a reason to compare simple pooling
with series-aware sequence aggregation, a path toward anatomy-guided pooling and
an MRI-specific robustness study. It also identified reusable public code and
the annotation mismatches that could invalidate reuse. The
[backlog](../backlog/INDEX.md) holds the idea queue.

## Evidence types and coverage

The review covered selected primary solution discussions and write-ups, author
repositories, public notebooks, organizer pages and original papers. It is a
targeted review of transferable methods, not a complete enumeration of every
RSNA competition, forum post or MRI paper. Radiograph-only competitions received
less attention because the volumetric and multi-series problems are better
matched by the sources above.

- **Author report:** a solution author's claimed score, ablation or experience,
  not reproduced here. A winning ensemble does not isolate the value of each
  component, and multiple solutions from the same authors are not independent
  replications.
- **Paper result:** the authors' experiment on its stated dataset and endpoint.
  External evaluation and controlled ablations are stronger than a leaderboard
  anecdote, but neither establishes transfer to this knee task.
- **Inspected implementation:** downloaded source establishes what the reviewed
  code contains. It does not authenticate attached weights, OOF arrays, reported
  scores or the exact code used for a historical submission.
- **Project proposal:** our synthesis, evaluated locally before adoption.

Some Kaggle pages returned empty browser bodies; indexed first-party text and
author-maintained repository versions supplied the cited content, and partial
coverage is identified where it limits a claim. The public code review covers
four notebook files and four Python files plus solution prose. Notebook source
cells are numbered from zero. Git commit IDs and downloaded-byte hashes are in
the [code manifest](competition-and-prior-art/source-manifest.json); external
datasets and model weights were not downloaded or audited.

The competition overview, data and rules URLs did not expose readable bodies in
this review, so requirements keep their explicit **2026-09-09 verification date**
in [supervision](../modeling/supervision.md#official-labeling-context) and
[inference requirements](../inference/submission-requirements-and-output-contract.md#verified-submission-requirements).
This review did not authorize any external dataset.

The knee-specific public evidence (discussion threads, notebooks, re-sweeps) is
in [public discussion and notebook findings](competition-and-prior-art/public-research.md);
source lineage and how to read unverified notes are in [provenance](provenance.md).
