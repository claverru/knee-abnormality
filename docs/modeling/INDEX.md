# Modeling

| Read when | Document |
|---|---|
| Understanding and reproducing the current model recipe, and when each part was adopted | [Current model](current-model.md) |
| Tracing model tensor shapes | [Model shape diagram](model-shapes.md) |
| Recovering the removed CoAtNet backbone and its spatial anchors (code removed 2026-10-14) | [CoAtNet 224 spatial backbone](coatnet-224-spatial-backbone.md) |
| Understanding triplet channels, normalization and sampling | [Triplet inputs and sampling](triplet-inputs-and-sampling.md) |
| Checking reference labels, class-balanced BCE, masking and Tonylin supervision | [Supervision](supervision.md) |
| Recovering the removed training-only report-attribute auxiliary heads (code removed 2026-10-14; JevK5 question targets, parent gating, soft cross-entropy, KL diagnostics) | [Report-attribute heads](report-attribute-heads.md) |
| Recovering the removed training-only scanner confusion head (code removed 2026-10-14; detached scanner-group head, bounded KL confusion loss, site debiasing, why gradient reversal was rejected) | [Scanner confusion head](scanner-confusion-head.md) |
| Diagnosing GPU batches, host RAM, prefetch and validation loader waves | [Training runtime and memory](training-runtime-and-memory.md) |
| Checking training schedules and augmentation, and recovering the removed weight averaging | [Training schedule and augmentation](schedule-and-augmentation.md) |
| Recovering retired side embeddings and classifier-sharing variants | [Retired side and compartment variants](retired-side-and-compartment-variants.md) |
| Reviewing the historical regional fusion model | [Historical regional fusion](historical-regional-fusion.md) |
| Changing backbones and checking spatial mapping | [Swapping a backbone](backbone-swaps.md) |
| Training MambaVision-T or ConvNeXt-Tiny beside the ResNet-50: stage adapters per backbone family, the MambaVision wrapper and its `mamba-ssm` import, the release and Kaggle wheels, new-batch-size cost and padding, training time of six backbones on the 3090, the weight-scale recipe | [MambaVision-T and stem-staged backbones](mambavision-backbone.md) |
| Training FocalNet-T SRF or EfficientViT-B2: their configs, stage adapters, FocalNet's pointwise convolutions as linear layers, EfficientViT's real-row BatchNorm padding, stochastic depth, stage-0 checkpointing and float32 attention, token mappings | [FocalNet-T and EfficientViT-B2 backbones](focalnet-and-efficientvit-backbones.md) |
| Comparing backbone compute, per-label AUC and bootstrap uncertainty | [Fast backbone search](backbone-search.md) |
| Finding unverified model, augmentation, loss and ensemble experiments | [Unverified modeling notes](unverified-modeling-notes.md) |

## Common lookups

| Read when | Section |
|---|---|
| Checking current backbone, input and training settings | [Model recipe](current-model.md#recipe) |
| Checking where the ResNet's tokens sit in the 224 px crop and why 225 or 257 px was not adopted | [Token grid alignment](../validation/results/2026-10-08-token-grid-alignment.md#the-224-px-grid) |
| Finding when a recipe change was adopted, its paired evidence and the control retrained after it (seeds, folds) | [Adoption history](current-model.md#adoption-history) |
| Reproducing the current model run | [Model reproduction](current-model.md#reproduce) |
| Checking per-label positive and negative BCE weights | [Class-balanced BCE](supervision.md#fixed-class-balanced-bce) |
| Checking reflection augmentation across imaging planes | [Study-level knee reflection](schedule-and-augmentation.md#study-level-knee-reflection) |
