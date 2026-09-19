# Modeling

| Read when | Document |
|---|---|
| Understanding the current full-slice spatial-token model (the `00-baseline` recipe in `configs/training.yaml`): recipe, training mode, embeddings, attention MIL, result, restoration checks and how to reproduce it | [Current model](current-model.md) |
| Following tensor shapes from prepared triplets through the backbone, spatial-token embeddings and target-specific attention pooling | [Model shape diagram](model-shapes.md) |
| Using the 224-pixel CoAtNet backbone, its spatial anchors, pretrained normalization and the ReXNet noninterference review | [CoAtNet 224 spatial backbone](coatnet-224-spatial-backbone.md) |
| Checking three-channel triplet input semantics, normalization and per-plane channel-appearance limits | [Triplet inputs and sampling](triplet-inputs-and-sampling.md) |
| Checking reference-label coverage, official labeling context, choosing inferred labels, fixed class-balanced BCE, training-label masking, refining supervision or reproducing the Tonylin training setup | [Supervision](supervision.md) |
| Diagnosing training batch memory, host RAM and loader prefetch with larger inputs, validation loader waves, dynamic-shape runtime, or the sampler settings recorded with the ReXNet runtime | [Training runtime and memory](training-runtime-and-memory.md) |
| Checking the ten-epoch learning-rate schedule, augmentation strengths or study-level knee reflection | [Training schedule and augmentation](schedule-and-augmentation.md) |
| Recovering the retired knee-side embedding, meniscus classifier coupling or 1A–1C disease-sharing heads | [Retired side and compartment variants](retired-side-and-compartment-variants.md) |
| Reviewing the replaced regional fusion model or reproducing its 18-mm per-plane coverage radius | [Historical regional fusion](historical-regional-fusion.md) |
| Swapping the backbone: parameter and compute budget, the 280 px cost, delta formulas, code requirements and spatial-mapping pitfalls | [Swapping a backbone](backbone-swaps.md) |
| Choosing a backbone from the fast window-based search, the unverified finished-model comparison, or its per-target AUCs, target leaders and bootstrap uncertainty | [Fast backbone search](backbone-search.md) |
| Reviewing unverified model-structure, window-sampling, augmentation, loss/ensemble and exploratory notes | [Unverified modeling notes](unverified-modeling-notes.md) |
