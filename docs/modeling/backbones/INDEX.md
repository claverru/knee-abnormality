# Backbones

| Read when | Document |
|---|---|
| Training or reproducing the ConvNeXt-Tiny family of the final ensemble: its config against the ResNet-50's, per-stage rate decay, token centres at 15.5 and 7.5 px, image counts padded to multiples of 16 | [ConvNeXt-Tiny backbone](convnext-tiny-backbone.md) |
| Training or reproducing the Refined-RadImageNet ResNet-50 family, or tracing how RadImageNet weights replaced ReXNet: original RadImageNet ResNet-50 runs, weight pin, normalization and weight-scale learning rate, Refined ResNet-18 and ResNet-10t, average-pool shortcuts and rejected in-block modules | [RadImageNet ResNet backbones](resnet-radimagenet-backbones.md) |
| Training FocalNet-T SRF, EfficientViT-B2 or NextViT-S (the third final family): their configs, stage adapters, FocalNet's pointwise convolutions as linear layers, EfficientViT's real-row BatchNorm padding, NextViT's wrapper and token mapping | [FocalNet-T, EfficientViT-B2 and NextViT-S backbones](focalnet-and-efficientvit-backbones.md) |
| Training MambaVision-T: stage adapters per backbone family, the MambaVision wrapper and its `mamba-ssm` import, release and Kaggle wheels, new-batch-size cost, training time of six backbones on the 3090, the weight-scale recipe | [MambaVision-T and stem-staged backbones](mambavision-backbone.md) |
| Checking the EfficientNet-B0/B3 runs against ReXNet-200 (September 2026) and what was never tested | [EfficientNet backbones](efficientnet-backbones.md) |
| Recovering the removed CoAtNet backbone and its spatial anchors (code removed 2026-10-14) | [CoAtNet 224 spatial backbone](coatnet-224-spatial-backbone.md) |
| Adding a backbone: stage adapter and mapping requirements, mapping pitfalls per timm family, checkpoint weight scale and learning rate, the ReXNet-era compute budget and same-cost candidates | [Swapping a backbone](backbone-swaps.md) |
| Comparing the early 40-backbone search on the 58 gold studies: per-label AUC, target leaders and bootstrap uncertainty against ReXNet-200 | [Fast backbone search](backbone-search.md) |
