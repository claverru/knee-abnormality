# Pretraining and external knee data

The strongest immediate pretraining experiment is to adapt an existing image
encoder more effectively while holding labels and sampling fixed. External knee
data is a later option for representation learning or localization, with explicit
label mapping. None of the resources below has been approved or incorporated
into this repository by this review.

## Adapting 2D foundation models to volumes

Liu et al.'s *AnyMC3D* combines a frozen 2D foundation backbone with trainable
LoRA adapters and query-based slice pooling. Its multi-view extension uses
view-specific adapters and pooling before combining view features. The paper
evaluates twelve tasks, including private shoulder MRI datasets; it does not
evaluate this RSNA knee task. In its pancreatic CT slice-fusion ablation,
query pooling obtained AUC **0.962**, mean pooling **0.958**, and LSTM **0.903**.
This is one task-specific comparison, not a universal ranking. Its DINOv2/v3
comparison reports no consistent winner.
[1](https://arxiv.org/html/2512.12887v2)

**Transfer proposal.** Compare frozen-feature pooling, a lightweight adapter,
and full fine-tuning with the same input coverage and seed. Try target-specific
queries without requiring a large sequence model. Preserve a simple mean/max
baseline: the older [RSNA solutions](rsna-mri.md) and this paper support different
aggregation choices in different settings.

Parameter-efficient adaptation reduces trainable state, but a LoRA-equipped
encoder still changes its features during training. A feature cache made before
those updates cannot replace its image forward pass. Similarly, swapping ten
adapters over one loaded backbone saves weight storage but ordinarily still
requires ten adapted forward passes. Measure actual throughput and memory;
trainable-parameter count is not an inference-time estimate.

The [aneurysm ablation](rsna-mri.md#intracranial-aneurysm-detection-2025) supports
segmentation-task pretraining when suitable labels exist. It does not rank all
medical pretraining above general visual pretraining. The knee review's
[RAD-DINO preprocessing failure](public-research.md#geometry-sampling-encoders-and-losses)
is an additional reason to pin the exact checkpoint and input normalization.

## External resources and label compatibility

| Resource | Potential use here | Main obstacle |
|---|---|---|
| MRNet | Knee representation pretraining; broad ACL/meniscus auxiliary tasks | Three labels, different severity definitions, and no medial/lateral meniscus split |
| fastMRI+ | Pathology-aware crop learning or sparse localization | Limited views and single-reader annotations; label coverage must be explicit |
| SKM-TEA | Cartilage/meniscus anatomy pretraining; pathology localization | Small qDESS cohort and substantial contrast/geometry differences |
| SKI10 | Historical segmentation methods | Official data download has closed |
| KneeCoT | No current adoption candidate | Existing host ruling disallows its access arrangement |

The first three are research leads with **current-competition eligibility
unverified**. Check the actual image, annotation, and checkpoint terms separately
before use. The [host guidance](../modeling/supervision.md#official-labeling-context)
already distinguishes routine registration from restrictive institutional
access; this review does not require another approval step for ordinary research.

### MRNet

MRNet's reference ACL label includes low-grade partial tears, whereas this
competition's [rubric](../modeling/supervision.md#official-labeling-context) targets
high-grade tears. Its meniscal target merges menisci, and its abnormality target
combines heterogeneous findings. The same disease name therefore does not imply
the same binary label.
[2](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699)

**Proposed mapping:** use a separate auxiliary head or pretraining task; do not
convert general abnormality into twelve positives or a combined meniscus label
into both compartment labels. Refit the competition heads using its rubric.
Keep patient identity and possible cross-dataset overlap under review if any
external images are later acquired.

### fastMRI+

Zhao et al. provide pathology annotations linked to fastMRI image files and
slices. The repository states that one radiologist labeled each image without
the other views available in a complete examination. A reviewed-file list is
separate from the annotation CSV: an unlisted finding row alone cannot identify
whether an image was reviewed. The source explicitly distinguishes reviewed
files without findings from unreviewed files.
[3](https://arxiv.org/abs/2109.03812),
[4](https://github.com/microsoft/fastmri-plus)

**Proposed mapping:** keep an annotation-observation mask and use bounding boxes
as sparse positive localization signals. Do not label every other pixel, slice,
or competition target negative. Even a reviewed scan with no recorded finding
is not automatically an adjudicated twelve-target negative under the RSNA
rubric. The [example notebook](code-and-notebooks.md#fastmri-annotation-overlay)
also exposes an important orientation convention.

### SKM-TEA

SKM-TEA's paper describes **155 patients**, qDESS knee MRI, four tissue
segmentations, and bounding boxes for sixteen pathologies. Its official
repository separates raw-data and DICOM analysis tracks; they have different
image products and corresponding segmentation masks. The dataset documentation
describes approximately **900 GB compressed / 1.6 TB expanded** and explains
that its raw DICOM directory is for visualization, with processed image files
for the DICOM analysis track. These are documented distribution sizes, not a
download measurement from this review.
[5](https://arxiv.org/abs/2203.06823),
[6](https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md)

**Proposed mapping:** consider anatomy pretraining for meniscus/cartilage crops,
using matched image/mask tracks. A mask of all cartilage does not assign medial,
lateral, or patellofemoral OA severity. Treat qDESS-to-routine-MRI transfer as an
experiment and evaluate the localizer on representative competition images
before allowing it to discard pixels.

## Admission criteria for an external-data experiment

Record the source/version, data and weight permissions, subjects and overlaps,
sequence types, physical spacing, annotation granularity, disease definitions,
and exactly which target entries are observed. Pin a label-mapping table before
training. Fit source-dependent preprocessing using training data only.

Compare the resulting model with a no-external-data baseline under the existing
reference protocol. A pretraining gain on the source dataset or a better
segmentation score does not answer whether knee reference AUC improves. Defer
large acquisitions until a concrete missing capability justifies them.

## Sources

1. Han Liu et al. *Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification*. CVPR 2026; reviewed arXiv:2512.12887v2, March 27, 2026, especially methodology and Appendix F. Some MRI evaluation data are private; no public implementation was audited here.
2. Nicholas Bien, Pranav Rajpurkar, et al. *Deep-learning-assisted diagnosis for knee magnetic resonance imaging*. PLOS Medicine, November 27, 2018; reference-label definitions.
3. Ruiyang Zhao et al. *fastMRI+: Clinical Pathology Annotations for Knee and Brain Fully Sampled Multi-Coil MRI Data*. arXiv:2109.03812, 2021; published dataset paper, 2022.
4. Microsoft / fastMRI+ authors. *Clinical pathology annotations for the fastMRI dataset*, official repository README and annotation conventions.
5. Arjun D. Desai et al. *SKM-TEA: A Dataset for Accelerated MRI Reconstruction with Dense Image Labels for Quantitative Clinical Evaluation*. NeurIPS 2021 Datasets and Benchmarks; arXiv posting March 14, 2022.
6. StanfordMIMI. *Stanford Knee MRI Multi-Task Evaluation Dataset*, `DATASET.md`, documented dataset version v1.0.0.

Reviewed on 2026-09-10. [SKI10 closure](mri-benchmarks.md#ski10-historical-knee-localization-evidence)
and the [KneeCoT ruling](../modeling/supervision.md#official-labeling-context) have separate
primary-source references.
