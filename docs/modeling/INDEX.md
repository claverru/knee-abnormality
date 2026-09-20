# Modeling

| Read when | Document |
|---|---|
| Understanding and reproducing the current model recipe | [Current model](current-model.md) |
| Tracing model tensor shapes | [Model shape diagram](model-shapes.md) |
| Using the CoAtNet backbone and its spatial anchors | [CoAtNet 224 spatial backbone](coatnet-224-spatial-backbone.md) |
| Understanding triplet channels, normalization and sampling | [Triplet inputs and sampling](triplet-inputs-and-sampling.md) |
| Checking reference labels, class-balanced BCE, masking and Tonylin supervision | [Supervision](supervision.md) |
| Diagnosing GPU batches, host RAM, prefetch and validation loader waves | [Training runtime and memory](training-runtime-and-memory.md) |
| Checking learning-rate schedules and augmentation | [Training schedule and augmentation](schedule-and-augmentation.md) |
| Recovering retired side embeddings and classifier-sharing variants | [Retired side and compartment variants](retired-side-and-compartment-variants.md) |
| Reviewing the historical regional fusion model | [Historical regional fusion](historical-regional-fusion.md) |
| Changing backbones and checking spatial mapping | [Swapping a backbone](backbone-swaps.md) |
| Comparing backbone compute, per-label AUC and bootstrap uncertainty | [Fast backbone search](backbone-search.md) |
| Finding unverified model, augmentation, loss and ensemble experiments | [Unverified modeling notes](unverified-modeling-notes.md) |

## Common lookups

| Read when | Section |
|---|---|
| Checking current backbone, input and training settings | [Model recipe](current-model.md#recipe) |
| Reproducing the current model run | [Model reproduction](current-model.md#reproduce) |
| Checking per-label positive and negative BCE weights | [Class-balanced BCE](supervision.md#fixed-class-balanced-bce) |
| Checking reflection augmentation across imaging planes | [Study-level knee reflection](schedule-and-augmentation.md#study-level-knee-reflection) |
