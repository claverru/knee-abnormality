# Modeling

| Read when | Document |
|---|---|
| Understanding the shared final model (posed views, spatial tokens, routed attention pooling, soft-tissue read, training-only maps), its common settings and when each part was adopted | [Current model](current-model.md) |
| Reproducing how the final ensemble members were trained: fastMRI and OAI pseudo-labels, pretraining, teacher share, generations, per-family splits and seeds | [External data and distillation](external-data-and-distillation/INDEX.md) |
| Recovering the side-routed shared heads adopted 2026-10-07 (run 24): shared meniscus and OA classifiers routed to medial, lateral and patellofemoral targets by a knee frame from the meniscus attention spots and the knee side, the side head, the sector-ring-height region table, position prior, what they replaced, and how the design evolved | [Side-routed shared heads](side-routed-heads.md) |
| Tracing model tensor shapes | [Model shape diagram](model-shapes.md) |
| Choosing, swapping or training a backbone: ConvNeXt-Tiny, RadImageNet ResNets, NextViT, FocalNet, EfficientViT, MambaVision, EfficientNet, CoAtNet, swap mechanics and the early search | [Backbones](backbones/INDEX.md) |
| Understanding triplet channels, normalization and sampling | [Triplet inputs and sampling](triplet-inputs-and-sampling.md) |
| Checking reference labels, class-balanced BCE, masking and Tonylin supervision | [Supervision](supervision.md) |
| Recovering the removed training-only report-attribute auxiliary heads (code removed 2026-10-14; JevK5 question targets, parent gating, soft cross-entropy, KL diagnostics) | [Report-attribute heads](report-attribute-heads.md) |
| Reading the site-bias evidence and every debiasing variant tried (scanner confusion targets, per-site biases, post-hoc centering) and why none beat the board | [Site bias](site-bias.md) |
| Recovering the removed training-only scanner confusion head (code removed 2026-10-14; detached scanner-group head, bounded KL confusion loss, site debiasing, why gradient reversal was rejected) | [Scanner confusion head](scanner-confusion-head.md) |
| Diagnosing GPU batches, host RAM, prefetch and validation loader waves | [Training runtime and memory](training-runtime-and-memory.md) |
| Checking training schedules and augmentation, and recovering the removed weight averaging | [Training schedule and augmentation](schedule-and-augmentation.md) |
| Recovering retired side embeddings and classifier-sharing variants | [Retired side and compartment variants](retired-side-and-compartment-variants.md) |
| Reviewing the historical regional fusion model | [Historical regional fusion](historical-regional-fusion.md) |
| Checking another participant's early model notes (window MIL, repeated slices, series dropout, flips, sagittal share) against what this project measured | [Unverified modeling notes](unverified-modeling-notes.md) |

## Common lookups

| Read when | Section |
|---|---|
| Checking the settings every family shares | [Shared settings](current-model.md#shared-settings) |
| Checking the per-family backbone settings of the final ensemble | [Per-family recipes](../final-solution.md#model-and-per-family-recipes) |
| Checking where the ResNet's tokens sit in the 224 px crop and why 225 or 257 px was not adopted | [Token grid alignment](../validation/results/crops-and-views/2026-10-08-token-grid-alignment.md#the-224-px-grid) |
| Finding when a recipe change was adopted, its paired evidence and the control retrained after it (seeds, folds) | [Adoption history](current-model.md#adoption-history) |
| Reproducing a family run or a chain stage | [Model reproduction](current-model.md#reproduce) |
| Choosing the teacher share | [Teacher share](external-data-and-distillation/teacher-student-chain.md#choosing-the-teacher-share) |
| Checking per-label positive and negative BCE weights | [Class-balanced BCE](supervision.md#fixed-class-balanced-bce) |
| Checking reflection augmentation across imaging planes | [Study-level knee reflection](schedule-and-augmentation.md#study-level-knee-reflection) |
