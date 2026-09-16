# Lessons from RSNA MRI competitions

The closest precedent is lumbar spine classification: combine complementary MRI
series, retain enough slices to see small structures, and aggregate at the unit
that receives a label. Aneurysm detection strengthens the case for anatomical
guidance when localization supervision exists. Brain radiogenomics supplies a
counterexample to treating leaderboard rank as proof of a reliable method.

Evidence conventions and review scope are in the [research index](INDEX.md).
All reported measurements below belong to their source task.

## Lumbar spine degenerative classification, 2024

This task predicted three severities for five conditions across five disc
levels. Its metric combined sample-weighted log losses with an additional
any-severe-spinal component; normal/mild, moderate, and severe received weights
1, 2, and 4. This differs materially from twelve independent knee targets
evaluated by macro AUC. [1](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/overview/abstract)

**First-place approach.** The reviewed portion of the Avengers write-up separates
slice selection, coordinate prediction, and severity prediction. Sagittal slice
localization used 3D ConvNeXt with regression and classification objectives;
coordinate models used 2D encoders and normalized coordinates. The author
reports a small benefit from coordinate-task pretraining over ImageNet for that
stage. This is evidence for task-aligned localization pretraining, not a
comparison of MRI versus natural-image pretraining for knee classification.
Only the indexed localization sections were available; do not infer unreviewed
training or validation details. [2](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution)

**Sixth-place approach.** Theo Viel's author-maintained account is more directly
reusable: a 2D coordinate model locates discs, CoAtNet processes cropped slices,
RNN heads combine their features, and an MLP aggregates predictions. Five frames
were insufficient for some targets; additional frames and separate left,
right, and central heads helped. Combining sagittal modalities also helped.
The final MLP worked best with each target's own features rather than unrestricted
cross-target interactions. Theo's sagittal branch reports private loss **0.420**;
the team ensemble reports **0.401**, with CV **0.382** and public loss **0.355**.
These are author-reported composite losses, not AUCs. [3](https://github.com/TheoViel/kaggle_rsna_lumbar_spine/blob/0e4dbea8ce16249792ec64cefa4cd5510eda7cdb/README.md)

The team write-up lists unsuccessful attempts including medical-specific
SpineNet models, external-data pseudo-labeling, denoising, auxiliary injury
localization, and a bi-encoder axial/sagittal architecture. Configurations and
paired uncertainty are incomplete, so these are reasons to require an ablation,
not permanent exclusions. Its four-fold setup and final MLP were designed for
the lumbar task. [4](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/nvspine-6th-place-solution)

**Seventh-place alternative.** Li Haowei's solution used predicted 3D points and
small ROI stacks; three slices worked best in one branch. The author notes
difficulties with multiple axial series and that selecting the highest-confidence
detection need not select the best one. This qualifies the sixth-place
experience: the useful slice count depends on crop coverage, targets, and
architecture. [5](https://github.com/lhwcv/solution-rsna-2024-lumbar-spine)

**Transfer proposal.** Start with study-level multiple-instance learning (MIL):
the label belongs to the study, while the model selects evidence among its
windows. Compare a shared encoder with target-specific pooling against a small
RNN within each physical series. Preserve plane/contrast identity when combining
series. Do not concatenate the end of one series and the beginning of another
as neighboring anatomy.

Test broader coverage before concluding that more slices are wasteful. A fixed
number of central sagittal windows may be adequate for one ligament but miss
peripheral or compartment-specific findings. Keep a global branch when adding
small crops so that crop errors cannot remove all evidence.

The lumbar coordinate annotations are an enabling resource absent from this
knee dataset. A learned knee localizer would require a separately justified
annotation source or a carefully evaluated weak localization method. See the
[crop notebook inspection](code-and-notebooks.md#lumbar-coordinate-crops) before
reusing its data flow.

## Brain tumor radiogenomic classification, 2021

The Kaggle task predicted MGMT promoter methylation from multiparametric brain
MRI. It was separate from the BraTS tumor-segmentation task, despite shared
imaging context. The organizer lists their results separately.
[6](https://www.rsna.org/artificial-intelligence/ai-image-challenge/brain-tumor-ai-challenge-2021)

First-place entrant Firas Baba describes a simple early baseline winning, with
unstable CV and little benefit from ensembling, large networks, metadata, or
several pretraining attempts. Second-place entrant Minh Phan used an
EfficientNet–LSTM approach with all four MRI sequences and five-fold averaging,
and also cautioned about the result's interpretation. The opposing architecture
choices undermine any claim that this competition settled 2D versus 3D.
[7](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/writeups/tunisia-ai-i-love-this-competition-1st-place-solut),
[8](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/writeups/minh-phan-2nd-place-solution)

A subsequent original study by Kim, Lee, and colleagues performed **420**
experiments varying datasets, architectures, sequence combinations, and seeds.
It reports external AUROC **0.562** and accuracy **0.548** for the first-place
solution on SNUH data. The abstract reports that **60.0%** of their models did
not differ significantly from chance in test AUROC; **80.2%** refers to accuracy,
not AUROC. These are the authors' statistical conclusions, not a fresh analysis.
The study provides strong evidence that earlier internal success did not
reliably generalize in these experiments; it does not prove MRI-based MGMT
prediction mathematically impossible. [9](https://doi.org/10.3390/cancers14194827)

**Transfer proposal.** The knee targets are structural findings with a different
signal, so do not import the pessimistic conclusion about MGMT. Import the
experimental discipline: preserve a small understandable baseline, compare
paired predictions, record seeds, and inspect target-level results. A small
reference set cannot reliably rank a long sequence of architecture, threshold,
and blend searches.

The inspected winning code also contains a train/inference slice-selection
asymmetry and filename-based sorting. Those details belong to a particular
released implementation, not a recommended DICOM contract. See
[the code review](code-and-notebooks.md#brain-tumor-input-selection).

## Intracranial aneurysm detection, 2025

This challenge covered CTA, MRA, and conventional MRI, with aneurysm labels at
13 anatomical locations. The organizer confirms tomoon33 as the winner.
[10](https://www.rsna.org/artificial-intelligence/ai-image-challenge/intracranial-aneurysm-detection-ai-challenge)

The winner's refined write-up describes coarse vessel localization, fine vessel
segmentation, and an ROI classifier initialized from segmentation training.
Regional pooling shares one backbone pass across anatomical heads. A prior
design needing fourteen passes was too expensive. Its reduced-resolution
ablations used folds 0–2 without retuning every variant:

| Variant | Reported aggregate score |
|---|---:|
| Reduced-resolution final model | 0.902 |
| Without backbone pretraining | 0.794 |
| Without auxiliary segmentation loss | 0.876 |
| Without location-aware transformer | 0.896 |

These are local ablation scores, not private leaderboard scores. The useful
signal is the dependence on segmentation supervision, stronger than the
reported transformer contribution in this configuration. On two T4 GPUs over
100 samples, mean stage times were **4.03 s** preprocessing, **10.65 s** vessel
segmentation, and **3.33 s** classification per series. Times are source-specific,
not knee runtime estimates. [11](https://github.com/uchiyama33/rsna2025_1st_place/blob/e1dcdf0058e1e0d0044d8053e92243b4b4794555/solution_v2_1.md)

**Transfer proposal.** If reliable knee masks become available, share image
features across twelve target heads and pool them with anatomy-specific weights.
A cartilage/meniscus/bone mask is an anatomical prior; it is not itself an
abnormality label. Keep global features and quantify failure coverage as well
as average AUC. The inspected
[pooling and submission code](code-and-notebooks.md#aneurysm-regional-pooling-and-submission)
offers concrete implementation references.

The pretraining ablation concerns a task-aligned segmentation encoder. It does
not establish that an arbitrary medical checkpoint beats ImageNet or DINO.
Likewise, lesion-point auxiliary losses cannot be recreated from study labels
alone. Prioritize this direction after a no-localizer baseline establishes both
accuracy and full inference cost.

## Sources

1. Kaggle/RSNA. *RSNA 2024 Lumbar Spine Degenerative Classification*, evaluation section, 2024; indexed official text.
2. Avengers. *1st place solution*, Kaggle, October 28, 2024; indexed localization sections only.
3. Theo Viel. *6th Place Solution … Theo's part*, author repository, competition 2024; pinned revision linked above.
4. NVSpine. *6th Place Solution*, Kaggle, October 24, 2024; indexed first-party write-up.
5. Li Haowei (lhwcv). *Kaggle Competition Solution (7th)*, author repository, competition 2024.
6. RSNA. *Brain Tumor AI Challenge (2021)*, official task and winner inventory.
7. Firas Baba. *1st place solution with very simple code*, Kaggle, 2021; indexed first-party write-up.
8. Minh Phan. *2nd place solution*, Kaggle, October 21, 2021; indexed first-party write-up.
9. Byung-Hoon Kim, Hyeonhoon Lee, et al. *Validation of MRI-Based Models to Predict MGMT Promoter Methylation in Gliomas: BraTS 2021 Radiogenomics Challenge*. Cancers 14(19), 4827, October 3, 2022. Full text inspected through the Europe PMC mirror, PMCID PMC9562637.
10. RSNA. *Intracranial Aneurysm Detection AI Challenge (2025)*, official task and winner inventory.
11. tomoon33 / uchiyama33. *1st place solution*, refined author version `solution_v2_1.md`, competition 2025; pinned revision linked above.

All linked sources were consulted on 2026-09-10. Source code provenance is
recorded separately in the [manifest](source-manifest.json).
