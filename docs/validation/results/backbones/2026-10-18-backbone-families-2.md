# 2026-10-18 Backbone families 2

Backbone families trained on fold 2 of `all_5fold_scanner_v2` (seed 1110484336) on the FocalNet-T recipe
with settings from their own weight scale, in the
[backbone-families-2 campaign](../../../../autoresearch/campaigns/backbone-families-2/INDEX.md)
(2026-10-17 13:55 to 2026-10-18 15:02 CEST, Codex, 25 trainings). Reference: FocalNet-T SRF
`focalnet-silver-fold-2`, ×4 0.938816, ×1 0.933241. NextViT-S became the incumbent and is merged as
`configs/nextvit.yaml` ([code](../../../modeling/backbones/focalnet-and-efficientvit-backbones.md#nextvit-s)). Run
details: the campaign's [run notes](../../../../autoresearch/campaigns/backbone-families-2/run-notes.md).

## Families on fold 2

×4 macro AUC alone, and the equal logit blend with the ResNet-50 `resnet50-silver-fold2` (×4 0.937820
alone) as gain over the ResNet-50; best run per family.

| Family (timm) | Pretraining | Epochs | ×4 | Blend gain |
|---|---|---:|---:|---:|
| NextViT-S (`nextvit_small.bd_ssld_6m_in1k`) | SSLD 6M | 10 | 0.938851 | +0.003267 |
| TinyViT-21M (`tiny_vit_21m_224.dist_in22k_ft_in1k`) | IN-22k distillation | 10 | 0.938808 | +0.003223 |
| FocalNet-T SRF (reference) | IN-1k | 12 | 0.938816 | +0.003179 |
| FocalNet-T LRF (`focalnet_tiny_lrf.ms_in1k`) | IN-1k | 12 | 0.937903 | +0.002933 |
| EfficientNet-B3 (`tf_efficientnet_b3.ns_jft_in1k`) | JFT-300M noisy student | 10 | 0.935737 | +0.002233 |
| FlashInternImage-T (DCNv4) | IN-1k | 7 | 0.935955 | +0.001833 |
| InceptionNeXt-T (`inception_next_tiny.sail_in1k`) | IN-1k | 10 | 0.936240 | +0.001829 |
| LingBot-Vision ViT-S/16 | web images, boundary SSL | 10 | 0.933450 | +0.001229 |
| VSSD-Tiny (MESA) | IN-1k | 5 | 0.933081 | +0.000315 |
| InceptionResNetV2 (original RadImageNet) | RadImageNet | 10 | 0.931412 | −0.000634 |
| TransNeXt-Micro | IN-1k | 4 | 0.928007 | −0.001120 |
| DenseNet-121 (original RadImageNet) | RadImageNet | 10 | 0.928967 | −0.001425 |
| InceptionV3 (original RadImageNet) | RadImageNet | 10 | 0.927303 | −0.001935 |
| MedSAM2 Hiera-T | CT, MRI, PET segmentation | 10 | 0.917773 | −0.002900 |

NextViT-S was adopted at ×4 +0.000034 (80% interval −0.0017 to +0.0018) and ×1 +0.001337: level
at ×4, ahead at ×1, one run. Per label against FocalNet-T SRF: PF OA +0.0075, Contusion +0.0050,
Medial OA +0.0036, Effusion +0.0030, ACL, Medial Meniscus and Baker's level, Synovitis, Lateral OA,
Lateral Meniscus and MCL −0.0033 to −0.0035, Fracture −0.0058.

OverLoCK-XT never trained: ten epochs projected to 14,736 s (its training-form large-kernel
depthwise convolutions). HGNetV2-B4, MambaOut-T, TResNet-M, CS3DarkNet-L, GhostNetV2-160,
Twins-SVT-S, Res2Net50d and BiT ResNetV2-50x1 were measured and seeded but not run.

## What it changes

- **Pretraining decides what a family adds to the ResNet-50.** Every RadImageNet-pretrained family
  lowered the blend (four DenseNet-121 settings, InceptionV3, InceptionResNetV2), the ResNet-50
  itself being RadImageNet-pretrained; SSLD, ImageNet-22k distillation, JFT and ImageNet-1k
  FocalNets added +0.002 to +0.0033.
- **Optimising a family that already ran moved it by about 0.001**, a new family by about 0.01; the
  user stopped settings follow-ups on 2026-10-18.
- **Measure before seeding:** published throughput missed training speed by 1.5 to 3.8 times; the
  GPU timing of 64 images (ten epochs ≈ 10 × (138 + 1,954 × s); peak at 256 images ≈ 4.7 × the
  64-image GiB) predicted every second-batch run within about 10%.
- **Campaign time bound:** up to 120% of the reference's training time (the user's setting of
  2026-10-18).

## Five folds of NextViT-S

`configs/nextvit.yaml` on the other folds with one seed per fold (`outputs/training/nextvit-silver-fold-{k}`):

| Fold | Seed | ×1 | ×4 | Training |
|---:|---:|---:|---:|---:|
| 0 | 451509333 | 0.93316 | 0.93819 | 3,082 s |
| 1 | 1514023685 | 0.92515 | 0.93127 | 3,096 s |
| 2 | 1110484336 | 0.93458 | 0.93885 | 3,094 s |
| 3 | 350704958 | 0.93032 | 0.93469 | 3,087 s |
| 4 | 1277371158 | training | | |
