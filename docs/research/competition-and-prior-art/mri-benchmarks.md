# Lessons from other MRI benchmarks

MRNet is the most direct task precedent. M&Ms supplies evidence about scanner
shift, while fastMRI and BraTS help distinguish image-processing quality from
the endpoint actually needed here. Their lessons complement the
[RSNA MRI review](rsna-mri.md).

## MRNet: knee MRI classification

The original MRNet system applies a pretrained 2D image encoder to slices,
spatially averages each slice's features, then max-pools across slices. Separate
series predictions are combined with logistic regression. This offers a useful
low-complexity comparator for today's attention-based window models.
[1](https://stanfordmlgroup.github.io/projects/mrnet/)

The organizer's dataset page describes **1,370 exams**, with report-derived
labels and patient-separated train/tuning/hidden-test sets of **1,130/120/120**
exams. The hidden test used expert reference labels. The paper calls that hidden
test its internal validation set, a terminology difference that can otherwise
mislead comparisons. The competition is now closed.
[2](https://stanfordmlgroup.github.io/competitions/mrnet/)

The original paper reports internal AUCs **0.937** for abnormality, **0.965** for
ACL, and **0.847** for meniscus. A Stanford-trained sagittal model obtained
external ACL AUC **0.824** on a different institution's protocol; training on
that external dataset produced **0.911**. These are different training and
evaluation settings, not an isolated normalization ablation. The paper retained
artifact-affected cases and shows that a correct abnormality prediction can
attend to a different finding from the lesion of interest.
[3](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699)

**Transfer proposal.** Compare max/mean pooling before adding a complex
aggregator. Keep the validation unit at study/patient level, and distinguish
report-derived development targets from expert reference targets. A heatmap can
help identify a failure hypothesis; it does not prove correct localization.
Recompute performance by target and series availability rather than assume one
plane dominates every finding.

The [external-data document](../backbones-and-pretraining/pretraining-and-external-data.md) explains why
MRNet's labels cannot simply become twelve-target knee labels.

## M&Ms: scanner and protocol robustness

The 2020 Multi-Centre, Multi-Vendor and Multi-Disease Cardiac Segmentation
challenge included **375** studies from four vendors and six hospitals. Its
organizer paper compares fourteen submitted approaches and highlights
intensity-driven augmentation. More complex domain adaptation did not establish
a general advantage over well-augmented nnU-Net systems; unseen domains remained
difficult. These are cardiac segmentation results, not evidence that a specific
augmentation improves knee AUC.
[4](https://www.iris.unina.it/retrieve/dca6a55d-d93a-4603-8089-eeb67be8f37b/Multi-Centre_Multi-Vendor_and_Multi-Disease_Cardiac_Segmentation_The_MampMs_Challenge.pdf)

Full et al.'s first-place study evaluated domain transfer on fixed holdouts.
Its final recipe combined stronger spatial/intensity augmentation with batch
normalization. The effect depended on the source and target vendors; stronger
augmentation alone did not consistently close the gap. The authors evaluated
the combinations rather than attributing everything to one transform.
[5](https://arxiv.org/abs/2011.07592v1)

**Transfer proposal.** Add a modest MRI intensity-variation experiment:
contrast/gamma, noise, bias-field-like intensity variation, or simulated lower
resolution. Start with one family at a time. Apply spatial transforms
consistently across the slices in a window and preserve anatomical label
meaning. Inspect thin-structure visibility after transformation.

Use scanner/sequence groups for a supplementary robustness analysis when
metadata supports them. Vendor is not hospital identity, and changing the
split also changes label prevalence and reporting style. Preserve the primary
fixed reference folds. A normalization-layer change should be evaluated with
the actual batch size and pretrained model; this cardiac result is not a
general instruction to replace layer or instance normalization with batch norm.

## BraTS: useful ablations, different supervision

BraTS segmentation offers aligned multi-contrast brain volumes and dense tumor
labels; the 2021 benchmark separately defines the radiogenomic classification
task. Channel-level fusion assumes corresponding voxels. Independently acquired
knee planes or sequences are not automatically registered volumes.
[6](https://arxiv.org/abs/2107.02314)

Futrega et al.'s *Optimized U-Net* won the BraTS 2021 validation phase and placed
third on the final test phase. In five-fold ablations, deep supervision changed
mean Dice **0.9130 → 0.9149**; decoder attention, residual connections, DropBlock,
and focal loss had much smaller changes. A VAE variant cost roughly three times
the training time without a better baseline score. The endpoint is tumor
segmentation Dice, not classification AUC.
[7](https://arxiv.org/abs/2110.03352v2)

**Transfer proposal.** Adopt the ablation method: compare a clear baseline,
change one component, and count time as well as score. Auxiliary tasks should
have an identifiable supervision source and a measured downstream benefit.
Do not assume a reconstruction or segmentation objective helps simply because
it is more anatomically detailed.

## fastMRI: preserving pathology through preprocessing

The 2019 fastMRI challenge concerned accelerated reconstruction of knee MRI
from raw k-space. It is a relevant source of knee imaging experience, but its
reconstruction leaderboard does not rank abnormality classifiers.
[8](https://arxiv.org/abs/2001.02518)

The 2020 brain challenge added a scanner-transfer track and emphasized pathology
depiction during radiologist evaluation. Its report describes hallucinated
features and abnormalities altered toward normal-looking anatomy even when
SSIM was high. That is a concrete counterexample to treating image similarity
as sufficient evidence of diagnostic fidelity.
[9](https://arxiv.org/abs/2012.06318v3)

**Transfer proposal.** Evaluate denoising, aggressive resampling, lossy caches,
and learned compression using downstream reference-label AUC and review of
small abnormalities. Favor the simplest processing that meets the required
throughput. A sharper-looking knee or smaller reconstruction error does not
establish preserved meniscus, cartilage, or ligament evidence.

Raw k-space reconstruction networks need inputs this competition does not
provide. Reconstructed-image encoders or anatomy supervision are different reuse
routes. The [fastMRI+ annotation notebook](code-and-notebooks.md#fastmri-annotation-overlay)
is particularly useful for understanding coordinate conventions and checking
that transformations preserve a lesion's location.

## SKI10: historical knee localization evidence

SKI10 compared knee bone and cartilage segmentation. Its evaluation included
surface distances for bones and overlap/volume errors for cartilage, rather
than abnormality labels. The organizer states that data sharing ended in 2019
and that downloads and submissions are closed.
[10](https://ski10.grand-challenge.org/),
[11](https://ski10.grand-challenge.org/FAQ/)

**Transfer proposal.** Use the challenge as historical evidence for anatomical
localization and physical-unit evaluation. It is not a currently available
dataset recommendation. Even an accurate cartilage outline does not directly
provide this competition's compartment-specific, high-grade OA labels.

## Sources

1. Stanford ML Group / Bien et al. *MRNet: Deep-learning-assisted diagnosis for knee magnetic resonance imaging*, project page, 2018.
2. Stanford ML Group. *A Knee MRI Dataset and Competition*, organizer page, 2018–2019; current closure notice reviewed.
3. Nicholas Bien, Pranav Rajpurkar, et al. *Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet*. PLOS Medicine 15(11), e1002699, November 27, 2018.
4. Víctor M. Campello et al. *Multi-Centre, Multi-Vendor and Multi-Disease Cardiac Segmentation: The M&Ms Challenge*. IEEE Transactions on Medical Imaging 40(12), 2021; author-institution PDF.
5. Peter M. Full, Fabian Isensee, Paul F. Jäger, Klaus Maier-Hein. *Studying Robustness of Semantic Segmentation under Domain Shift in cardiac MRI*. STACOM/MICCAI 2020, arXiv v1, November 15, 2020.
6. Ujjwal Baid et al. *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification*. arXiv:2107.02314, 2021.
7. Michał Futrega, Alexandre Milesi, Michal Marcinkiewicz, Pablo Ribalta. *Optimized U-Net for Brain Tumor Segmentation*. arXiv:2110.03352v2, December 24, 2021; PDF ablations reviewed.
8. Florian Knoll et al. *Advancing machine learning for MR image reconstruction with an open competition: Overview of the 2019 fastMRI challenge*. arXiv:2001.02518, 2020; abstract and task description reviewed.
9. Matthew J. Muckley, Bruno Riemenschneider, et al. *Results of the 2020 fastMRI Challenge for Machine Learning MR Image Reconstruction*. IEEE Transactions on Medical Imaging, 2021; arXiv v3, May 3, 2021, full PDF reviewed.
10. SKI10 organizers. *Segmentation of Knee Images 2010*, current official home page.
11. SKI10 organizers. *Frequently Asked Questions*, official evaluation definitions.

Reviewed on 2026-09-10. Dataset availability is distinct from permission to use
data or weights in the current Kaggle competition.
