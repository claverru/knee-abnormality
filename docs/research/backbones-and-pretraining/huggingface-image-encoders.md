# Reusing Hugging Face image and MRI encoders

Reviewed 2026-09-20. **Several released encoders are technically reusable, but none of the new
candidates is a config-only swap in this repository.** Keep timm where it already supports the
checkpoint. Start with DINOv3 ConvNeXt-T as the practical general-image candidate and RadioDINO-S
as the small medical candidate. OrthoFoundation is the most directly relevant released 2D knee
encoder found, but its large backbone and loading details make it a separate investigation.
These are research priorities, not measured improvements or adopted implementation decisions.

Scope: primary model cards, Hub API file/config metadata, author repositories and selected source
files, compared with current repository code and prior results. No weights were downloaded or
executed, no training or tests were run, and runtime/memory are unmeasured. Searches covered
general visual encoders, radiology pretraining, knee/MRI models, volumetric encoders and
reconstruction models; this is a shortlist, not an exhaustive Hub inventory. Proposed work has
one owner in [Big bets](../../validation/results/backbones/INDEX.md).

## Hugging Face does not require replacing timm

The Hub distributes weights for multiple libraries. A timm-formatted repository can be loaded
with `timm.create_model("hf_hub:owner/model", pretrained=True)`; Transformers, OpenCLIP and custom
PyTorch checkpoints have different loading contracts. Hugging Face hosting alone does not imply
`AutoModel` compatibility. [timm documentation](https://huggingface.co/docs/timm/hf_hub)

The current model in [model.py](../../../src/knee_abnormality/modeling/model.py) takes three adjacent
MRI slices as three channels, creates a timm backbone without its classifier, and requires
`forward_features` to return `[N, C, Hf, Wf]` with `C == num_features`. It projects the spatial
features to 256 dimensions and uses label-specific attention over study tokens. The pinned timm
version is 1.0.26; Transformers and OpenCLIP are not direct project dependencies.

[spatial_tokens.py](../../../src/knee_abnormality/modeling/spatial_tokens.py) also requires a
registered, verified mapping from feature cells to input pixels. Its five accepted names cover
ReXNet, CoAtNet and EfficientNet variants. A Hub identifier alone therefore fails even when timm
can construct the architecture. [Backbone swap requirements](../../modeling/backbones/backbone-swaps.md#code-requirements)

## Practical 2D shortlist

| Checkpoint | Pretraining and representation | Reuse assessment |
|---|---|---|
| [timm/convnext_tiny.dinov3_lvd1689m](https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m) | General images; about 28M parameters, 768-channel final map, 224-pixel config | Lowest integration cost on this shortlist: retain timm and the existing spatial head, add the correct ConvNeXt mapping. This is a general-pretraining control, not a medical model. |
| [Snarcy/RadioDino-s16](https://huggingface.co/Snarcy/RadioDino-s16) | DINO ViT-S/16 trained on RadImageNet: the card reports 1.35M CT/MRI/ultrasound images; 384-dimensional features, 224 pixels | Best small medical candidate to investigate first. Published timm config; needs patch-token adaptation and geometry. Card reports CC BY 4.0 and links its paper. Reported medical benchmarks do not establish knee-task gains. |
| [Snarcy/OmniRad-small](https://huggingface.co/Snarcy/OmniRad-small) | DINOv2 ViT-S/14; card reports about 1.2M RadImageNet images; 21.6M parameters, 384 dimensions, 224 pixels | Another practical medical candidate with a timm config. Resolve conflicting license metadata before adoption. The base variant uses 768 dimensions and adds cost without local evidence. |
| [OrthoFoundation-L](https://github.com/ytrsk/OrthoFoundation) | DINOv3-L continued pretraining on knee radiographs and MRI slices; released config uses ViT-L/16 | Strongest direct anatomical match among the released 2D candidates found. Official weights are on GitHub, not a verified Hub release. Large model; custom checkpoint loading and patch-token integration remain to be established. |
| [MRI-CORE](https://github.com/mazurowski-lab/mri_foundation) | More than 6M slices from over 110,000 MRI volumes across 18 body locations; released SAM-style ViT-B encoder | MRI-specific lead, but heavier integration and preprocessing work. The author code exposes a 256-channel NCHW encoder map; examples use 1024 pixels. Official checkpoint link is Google Drive, not a verified Hub model. [Paper](https://arxiv.org/abs/2506.12186v2) |

### RadioDINO and OmniRad

RadioDINO-S/16 was trained and tuned on 2026-10-15 and not adopted (×4 0.929641, below the
ResNet-50 and MambaVision-T and lowering their blends): [result](../../validation/results/backbones/2026-10-15-radiodino-s16.md).

Every RadImageNet-pretrained release found, including the S/8 and B/16 variants and the
supervised CNNs, is catalogued in [RadImageNet weights](radimagenet-weights.md).


Both expose patch features compatible in principle with the current MIL head after reshaping.
At 224 pixels, RadioDINO's patch size 16 gives 196 spatial tokens and OmniRad's patch size 14
gives 256, versus the current ReXNet's 49. These are derived counts, respectively 4× and about
5.2× as many tokens; they are not measured runtime ratios. ViT attention operates within each
triplet, while our study-level MIL head remains a separate cost.

Use `forward_features`, remove non-spatial prefix tokens according to the architecture, and
reshape only patch tokens. A pooled CLS embedding cannot replace our spatial map without
changing the pooling experiment. Match each checkpoint against the same architecture with
general-image pretraining to distinguish domain adaptation from architecture and token density.

**OmniRad has a concrete metadata discrepancy:** both small and base model cards state
CC BY 4.0, while their `pretrained_cfg.license` fields state `cc-by-nc-4.0`.
The source framework separately reports Apache 2.0. This review cannot determine whether the
config field is stale; do not assume the most permissive declaration resolves weight terms.
[Small config](https://huggingface.co/Snarcy/OmniRad-small/blob/4650430aa7361d159a5fa75923fb9e2793993eb7/config.json),
[source framework](https://github.com/unica-visual-intelligence-lab/OmniRad)

### OrthoFoundation

The author repository currently reports 893,985 MRI slices and 357,670 radiographs, combining
OAI, fastMRI and a private multicenter knee cohort. It evaluates ACL, PCL, MCL, meniscus and
cartilage tasks after fine-tuning. These are unusually relevant source tasks, but their metrics
are not comparable to our twelve-label competition AUC. The published GitHub file
`OrthoFoudation-L.pth` is about 1.21 GB according to the GitHub contents API.
[Author release](https://github.com/ytrsk/OrthoFoundation)

The code loads a local DINOv3 implementation and weight path through `torch.hub`; it does not
provide a ready-made adapter for our model. Its wrapper manually traverses the transformer
blocks. Establish the released state-dict layout and intended DINOv3 forward semantics before
porting it; do not infer complete compatibility from the model name. The augmentation source
uses ImageNet mean/std and 224-pixel global crops. Repository MIT metadata does not by itself
settle the terms inherited from the DINOv3 base weights.
[Wrapper](https://github.com/ytrsk/OrthoFoundation/blob/4ae0a0aa1a5eedf578c1ec8db3e82bd9fac68470/nets/dinov3_vit.py),
[augmentation](https://github.com/ytrsk/OrthoFoundation/blob/4ae0a0aa1a5eedf578c1ec8db3e82bd9fac68470/datasets/augmentation.py)

### MRI-CORE

The released encoder has a patch-16 embedding and a convolutional neck returning NCHW features.
Its source includes positional-embedding resizing for other input sizes. That makes a smaller
input technically plausible, but does not establish that the checkpoint retains its usefulness
at 224 pixels. The author examples specify per-slice normalization to `[0, 1]`; audit the exact
encoder call and normalization rather than inserting our ImageNet preprocessing automatically.
The README states Apache 2.0 for code and models. Weight retrieval and loading remain unverified.
[Encoder](https://github.com/mazurowski-lab/mri_foundation/blob/0f4f452614013873f8bb58a1fd613dbe36b58aba/models/sam/modeling/image_encoder.py),
[builder](https://github.com/mazurowski-lab/mri_foundation/blob/0f4f452614013873f8bb58a1fd613dbe36b58aba/models/sam/build_sam.py)

## Knee-specific 3D lead

[lanstat0123/orthodiffusion](https://huggingface.co/lanstat0123/orthodiffusion) contains public
`axial_model.pt`, `coronal_model.pt` and `sagittal_model.pt` files and declares MIT. The
[paper](https://arxiv.org/abs/2602.20752) reports pretraining three orientation-specific 3D
diffusion backbones on 15,948 unlabeled knee MRI scans, then transferring to eight disease
targets and eleven anatomical structures.

This is more relevant than a generic image-generation checkpoint: the
[official implementation](https://github.com/lt-0123/OrthoDiffusion) explicitly extracts
intermediate denoising features for classification and supports frozen-feature probing.
Its linear-probe config uses `[1, 16, 256, 256]` volumes; extraction selects a diffusion timestep
and feature block. It does not require generating an MRI through the full sampling trajectory
to obtain features. [Config](https://github.com/lt-0123/OrthoDiffusion/blob/main/configs/config_linear.yaml),
[extraction](https://github.com/lt-0123/OrthoDiffusion/blob/main/linear_classifier.py)

Reusing it requires a volume-building policy, orientation-specific inputs, different intensity
normalization, 3D feature aggregation and a new twelve-target head. Our three image channels
are adjacent slices, not a 3D tensor. Neither its input nor its five-dimensional feature maps
satisfies our current contract. It is a substantial separate experiment; file presence does not
establish checkpoint loading, competition eligibility or a score benefit.

## Lower-priority and unsuitable shortcuts

| Model/family | Why it is not the first experiment |
|---|---|
| [microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino) | Reusable Transformers DINOv2-B/14 encoder, but trained on chest X-rays. It is not an MRI-pretrained model. Its processor uses 518 pixels and mean 0.5307/std 0.2583 in every channel, unlike ImageNet. [Pinned processor](https://huggingface.co/microsoft/rad-dino/blob/110cbc18d5133582e320b43d53bf5c44e410c936/preprocessor_config.json) |
| [MedDINOv3-ViTB-16-CT-3M](https://huggingface.co/ricklisz123/MedDINOv3-ViTB-16-CT-3M) | Public custom `.pth` weights; the card describes 3.87M CT slices, not MRI pretraining. Already present in the historical screen, with no encouraging result there. |
| [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) and RadImageNet CNNs | Technically reusable, but already represented in the historical screen. BiomedCLIP's biomedical figure/text training is not equivalent to knee MRI pretraining. See [prior MRI evidence](backbone-mri-evidence.md#this-dataset). |
| [google/medsiglip-448](https://huggingface.co/google/medsiglip-448) | Its training includes CT/MRI slices as well as other medical and natural images. The vision encoder alone is about 400M parameters at 448 pixels. Auto-gated under Google's Health AI Developer Foundations terms; technically useful, but a costly first choice for many triplets per study. |
| [NeuroVFM](https://huggingface.co/mlinslab/neurovfm-encoder), [Dinov3D-Neuro](https://huggingface.co/huggingbrain/Dinov3d-Neuro), brain-MRI encoders | MRI modality match does not remove the brain-to-knee anatomy gap or the 3D integration cost. NeuroVFM's inspected Hub metadata is manually gated and CC BY-NC-SA 4.0. |
| [Decipher-MR](https://pmc.ncbi.nlm.nih.gov/articles/PMC13230676/) | Relevant multianatomy 3D MRI research, but the paper's code-availability statement says pretrained weights might not be shared. No directly reusable public checkpoint was established. |
| [microsoft/mri-autoencoder-v0.1](https://huggingface.co/microsoft/mri-autoencoder-v0.1) and fastMRI reconstruction models | Knee/brain MRI pretraining is real, but the former targets complex-valued MRI compression/reconstruction and is evaluated by PSNR/SSIM. Neither that objective nor a reconstruction leaderboard result establishes useful abnormality features for our magnitude DICOM inputs. |

Large language/report-generation models and narrow brain-tumor classifiers were not promoted
merely because they carry an MRI tag. Their full task heads are not our intended reusable unit.

## Adaptation and evaluation requirements

1. **Preserve spatial information.** CNNs need verified feature-map origins/strides. ViTs need
   the actual patch grid, prefix-token removal and coordinates: an unpadded patch-16 projection
   has pixel-center origin 7.5 and stride 16; patch-14 has origin 6.5 and stride 14. Trace the
   instantiated architecture before registering a mapping. Keep the existing affine/reflection
   validity handling and head when measuring a backbone change.
2. **Match input semantics.** Medical 2D encoders commonly see one grayscale slice repeated
   across RGB channels; our channels contain three different slices. Shape compatibility does
   not make these distributions equivalent. Keep the triplet input in the first practical
   fine-tuning comparison; a repeated-center-slice diagnostic is a separate change, especially
   relevant when judging frozen features.
3. **Pin preprocessing and weights.** Record the exact revision, weight file/hash, channel
   convention, mean/std, resize/crop and interpolation. Preserve physical coverage deliberately;
   an automatic image processor's center crop can change it. Package the selected encoder and
   dependencies for offline submission; the Lightning training framework can remain.
4. **Use current evidence standards.** Compare against a fresh control with the current
   balanced-BCE recipe, label source, split, preprocessing identity, sampling, image size and
   study coverage. Record seed, per-label/macro AUC, runtime and memory. A promising one-seed
   difference around 0.001 needs replication under the
   [measurement rules](../../final-solution.md), not immediate adoption. Evaluate ensemble value
   as well as standalone AUC; the B0/ReXNet experience shows these can disagree.
5. **Separate feasibility from eligibility.** The official
   [competition rules URL](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
   returned no readable body in this review, so eligibility is not newly verified. Before
   adoption, check the exact weight terms, public availability and source/competition overlap;
   a public file or a private pretraining corpus alone does not settle eligibility. Existing
   [external-data guidance](pretraining-and-external-data.md#admission-criteria-for-an-external-data-experiment)
   remains the reference. No access terms were accepted by this review.

The [older backbone screen](../../modeling/backbones/backbone-search.md#finished-model-comparison) ranked
DINOv3 ConvNeXt-T second, RadImageNet ResNet-50 nineteenth, MedDINOv3 twenty-fifth and BiomedCLIP
thirty-fifth. It used selected windows and checkpoint selection on 58 gold studies, and lacks
configuration/split/seed/artifact identities. It supports prioritization only; it does not prove
medical pretraining fails or predict rankings under the current full-slice balanced-BCE recipe.

## Inspected artifact revisions

The Hub API supplied these revisions on 2026-09-20. File inventories and small configs were
inspected; weights were not. Pin the final weight files separately if an experiment starts.

| Repository | Revision |
|---|---|
| `timm/convnext_tiny.dinov3_lvd1689m` | `39c7bb32f83c11bbb13cb23058f8852a4dac88ed` |
| `Snarcy/RadioDino-s16` | `773d28d873455153495d39179fc42ea816aa1212` |
| `Snarcy/OmniRad-small` | `4650430aa7361d159a5fa75923fb9e2793993eb7` |
| `Snarcy/OmniRad-base` | `3e42381bcf0e6981a8b438e335c807c70b857e9a` |
| `microsoft/rad-dino` | `110cbc18d5133582e320b43d53bf5c44e410c936` |
| `ricklisz123/MedDINOv3-ViTB-16-CT-3M` | `20d233b0ce1b1f93c46550787cf4b409a6f4a95e` |
| `google/medsiglip-448` | `9cea28a1a1195f665105faa6e8544c112fd960a4` |
| `lanstat0123/orthodiffusion` | `d6920d49e7d1b3dd0774eed6b8e47e386a54dd81` |
| `mlinslab/neurovfm-encoder` | `d5194fc70a162185f8ef062e362bd522a35312a9` |
| GitHub `ytrsk/OrthoFoundation` | `4ae0a0aa1a5eedf578c1ec8db3e82bd9fac68470` |
| GitHub `mazurowski-lab/mri_foundation` | `0f4f452614013873f8bb58a1fd613dbe36b58aba` |
