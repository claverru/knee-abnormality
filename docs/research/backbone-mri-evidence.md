# Backbone MRI evidence

Which 2D backbone families have held up on MRI, for the main backbone and for the scout of the
dropped [crop-selection design](../history/crop-selection-investigation.md). A web review on
2026-09-16 focused on lightweight families for the scout; classifier backbones were recorded
as encountered, not searched exhaustively. Sources were read but not
pinned to commits; evidence types follow [evidence and coverage](evidence-and-coverage.md).
Swap costs and code requirements are in [swapping a backbone](../modeling/backbone-swaps.md).

## Answer

- **Main backbone:** the strongest evidence is from this dataset. ReXNet-200 led an
  unverified 40-model comparison, ahead of ConvNeXt-T (DINOv3), CoAtNet and RegNetY-160.
  Outside it, gold-medal MRI classifiers mostly used ConvNeXt (nano to base), EfficientNetV2,
  MaxViT, CAFormer, Swin and CoAtNet.
- **Scout:** no backbone of scout size (≤ 5M parameters) has consistent MRI evidence.
  EfficientNet is the only family that does, from other MRI tasks. Gold-medal localizers were
  mostly large; only `regnety_002` appeared as a small one.
- **ImageNet accuracy is a weak guide:** [CheXtransfer](https://arxiv.org/abs/2101.06871)
  (16 architectures, chest X-ray) found no relationship between ImageNet and downstream
  performance. It also found that pretraining helps small models more, and that trimming final
  blocks cut parameters about 3.25× without a significant loss.
- **Bigger is not reliably better:** 1st place in lumbar spine found ConvNeXt-large < base <
  small, and 5th place found no gain from bigger backbones.

## This dataset

The [fast backbone search](../modeling/backbone-search.md#finished-model-comparison) compared
full-size models in the crop role. Its provenance is unverified, and selection used the 58
gold studies.

- **Top ranks:** ReXNet-200 1st of 40, DINOv3 ConvNeXt-T 2nd, CoAtNet 3rd, RegNetY-160 4th and
  ReXNet-300 7th.
- **Low ranks:** EfficientNetV2-S 38th.
- **Medical pretraining did not help:** RadImageNet ResNet-50 ranked 19th, MedDINOv3 25th,
  BiomedCLIP 35th, RadImageNet DenseNet-121 39th and RadiologyNet ResNet-50 40th.

## Lumbar spine, 2024

Author reports and repositories from the
[RSNA 2024 lumbar spine](rsna-mri.md#lumbar-spine-degenerative-classification-2024) gold
solutions.

**Localizers (the scout's role).** Mostly large models:

- 1st: ConvNeXt-base and EfficientNetV2-L.
- 2nd: `hgnetv2_b5` at 256 px ([repo](https://github.com/brendanartley/RSNA-2024-Competition)).
- 3rd: EfficientNet-B3, B4 and B6.
- 6th: `efficientnetv2_rw_t`
  ([repo](https://github.com/darraghdog/kaggle-rsna-2024-6th-place-solution-model2)).
- 11th: EfficientNet-B3 and B5.
- 5th is the only small localizer: a 2D UNet+LSTM on `regnety_002`, ensembled with
  `efficientnet_b5`, which the team reported as its best
  ([repo](https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification)).

**Classifiers (the main backbone's role).**

- **Common families:** ConvNeXt small, tiny and nano, EfficientNetV2, MaxViT, CAFormer and
  Swin.
- **Individual teams:**
  - 4th: CAFormer-S18, ConvNeXt-T, SwinV2-T and ResNet-RS50.
  - 6th: CoAtNet on crops.
  - 7th: `rexnetr_200` and `convnext_nano`
    ([repo](https://github.com/lhwcv/solution-rsna-2024-lumbar-spine)).
  - 11th: kept a `tf_efficientnet_b0_ns` MIL+GRU model in its final ensemble
    ([write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/yynk-11th-place-solution)).
- **3rd place's comparison:** one pipeline, 5-fold CV weighted log loss, lower is better
  ([repo](https://github.com/Moyasii/Kaggle-2024-RSNA-Pub), `src/predict.py`):

  | Backbone | Spinal | Side |
  |---|---:|---:|
  | `convnext_nano` | 0.3968 | 0.4044 |
  | `convnext_tiny` | 0.3970 | – |
  | `efficientnetv2_rw_t` | 0.3979 | – |
  | `maxvit_rmlp_nano` | – | 0.4018 |
  | `resnet18` | 0.4014 | 0.4040 |
  | `mnasnet_small` | 0.4033 | 0.4062 |

  `mnasnet_small` still joined the final ensemble.

## Other sources

- **UW-Madison GI tract (MRI segmentation):** 1st place used EfficientNet-B4 to B7 UNets plus
  an empty-slice classifier
  ([write-up](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/writeups/nccsm-1st-place-solution-for-2-5d-parts)).
- **Brain tumor radiogenomics, 2021:** says nothing about backbones, since most models were at
  chance ([Saeed et al.](https://arxiv.org/abs/2304.00774)); see
  [RSNA MRI](rsna-mri.md#brain-tumor-radiogenomic-classification-2021).
- **Knee MRI papers:** MRNet comparisons use one 120-exam validation set
  ([Azcona 2020](https://arxiv.org/abs/2010.01947) favoured VGG16 over EfficientNet for ACL).
  [MeniMV](https://arxiv.org/html/2512.18437) trained from scratch, which does not transfer to
  fine-tuning.
- **Lightweight benchmarks:** none is rigorous.
  - A 45-model brain-tumor study
    ([Cluster Computing 2024](https://link.springer.com/article/10.1007/s10586-024-04532-1))
    reported 99.54% for FastViT-s12 and RepGhostNet-200, from one split of a Kaggle set with no
    repeats.
  - [MV2SwimNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC12385416/) reported 99.94% on MRNet
    against a 34% EfficientNet-B2 baseline.

## Verdict by family

| Family | MRI evidence |
|---|---|
| ReXNet | Strong on this dataset at full size; `rexnetr_200` in lumbar 7th; small sizes untested |
| ConvNeXt | Strong: lumbar classifiers (nano best in 3rd's comparison) and DINOv3 ConvNeXt-T 2nd here; atto to zepto untested |
| EfficientNet (B0–B7, V2) | Consistent on other MRI tasks (lumbar localizers and ensembles, UW-Madison 1st); V2-S 38th here |
| CoAtNet, MaxViT, CAFormer, Swin | Lumbar classifiers; CoAtNet 3rd here, MaxViT-T 37th |
| RegNetY/X | Some: `regnety_002` lumbar 5th localizer; RegNetY-160 4th here |
| HGNetV2 | B5 only (lumbar 2nd localizer) |
| MnasNet | Some: lumbar 3rd ensemble, behind ConvNeXt-nano |
| MobileNetV3, GhostNet family, MobileViT, FastViT | Low-quality papers only |
| MobileNetV4, EfficientNet-Lite, EdgeNeXt, EfficientViT, LeViT, EfficientFormer, RepViT, TinyNet, LCNet | None found |
