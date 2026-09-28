# Modeling

| Read when | Document |
|---|---|
| Understanding and reproducing the current model recipe, and when each part was adopted | [Current model](current-model.md) |
| Tracing model tensor shapes | [Model shape diagram](model-shapes.md) |
| Using the CoAtNet backbone and its spatial anchors | [CoAtNet 224 spatial backbone](coatnet-224-spatial-backbone.md) |
| Understanding triplet channels, normalization and sampling | [Triplet inputs and sampling](triplet-inputs-and-sampling.md) |
| Checking reference labels, class-balanced BCE, masking and Tonylin supervision | [Supervision](supervision.md) |
| Enabling or interpreting the training-only report-attribute auxiliary heads (JevK5 question targets, parent gating, soft cross-entropy, KL diagnostics) | [Report-attribute heads](report-attribute-heads.md) |
| Understanding the training-only scanner confusion head (detached scanner-group head, bounded KL confusion loss, site debiasing, why gradient reversal was rejected) | [Scanner confusion head](scanner-confusion-head.md) |
| Diagnosing GPU batches, host RAM, prefetch and validation loader waves | [Training runtime and memory](training-runtime-and-memory.md) |
| Checking training schedules, augmentation and weight averaging | [Training schedule and augmentation](schedule-and-augmentation.md) |
| Recovering retired side embeddings and classifier-sharing variants | [Retired side and compartment variants](retired-side-and-compartment-variants.md) |
| Reviewing the historical regional fusion model | [Historical regional fusion](historical-regional-fusion.md) |
| Changing backbones and checking spatial mapping | [Swapping a backbone](backbone-swaps.md) |
| Comparing backbone compute, per-label AUC and bootstrap uncertainty | [Fast backbone search](backbone-search.md) |
| Finding unverified model, augmentation, loss and ensemble experiments | [Unverified modeling notes](unverified-modeling-notes.md) |

## Common lookups

| Read when | Section |
|---|---|
| Checking current backbone, input and training settings | [Model recipe](current-model.md#recipe) |
| Finding when a recipe change was adopted, its paired evidence and the control retrained after it (seeds, folds) | [Adoption history](current-model.md#adoption-history) |
| Reproducing the current model run | [Model reproduction](current-model.md#reproduce) |
| Checking per-label positive and negative BCE weights | [Class-balanced BCE](supervision.md#fixed-class-balanced-bce) |
| Checking reflection augmentation across imaging planes | [Study-level knee reflection](schedule-and-augmentation.md#study-level-knee-reflection) |
