# Backbone families 2: new candidates of 2026-10-18

Measured by the keeper on 2026-10-18 at the user's request ("search new backbone families, add more seeds"), with
the GPU idle: 20 timm backbones with pretrained weights, downloaded to the torch hub cache. No training yet.

## Why these

The campaign's blends with the ResNet-50 `resnet50-silver-fold2` (equal logit, fold 2, x4) separate the
families by pretraining, not by score alone: FocalNet-T SRF +0.003179, FlashInternImage-T +0.0017 to +0.0018,
LingBot ViT-S/16 up to +0.0012, VSSD-Tiny +0.0003, and every RadImageNet-pretrained backbone negative
(InceptionResNetV2 -0.0006 to DenseNet-121 stage decay -0.0024), the RadImageNet images being the ResNet-50's own
pretraining (guess: shared errors). So the candidates favour pretraining other than RadImageNet: ImageNet-22k,
semi-supervised SSLD, JFT noisy student, distillation, and new designs on ImageNet-1k.

## Measurements

Forward and backward of `forward_features`, 64 images at 224 px, bf16, channels last, cudnn benchmark, random
weights (time does not depend on them). Ten epochs projected as 10 × (138 + 1,954 × s), the fit of the FocalNet-T
and ResNet-50 runs (prior-work-radimagenet.md). Peak at 256 images about 4.7 × the 64-image figure (DenseNet-121:
4.1 → 19.2 GiB measured; FocalNet-T SRF: 4.5 → 21.5 GiB), so above about 4.8 GiB per 64 a family needs stage
checkpointing for the 22.5 GiB cap. Weight RMS: matrices and kernels excluding the head, median per fifth of the
parameter list from stem to last stage.

| Backbone (timm) | Pretraining | Params | s / 64 | GiB / 64 | Ten epochs | Weight RMS by depth | BN / LN / depthwise | Strides 8, 16, 32 channels |
|---|---|---:|---:|---:|---:|---|---|---|
| `focalnet_tiny_lrf.ms_in1k` | IN-1k | 27.9M | 0.112 | 4.87 | 3,566 s | 0.054 → 0.050 flat | 0 / 29 / 36 | 192, 384, 768 |
| `nextvit_small.bd_ssld_6m_in1k` | SSLD 6M semi-supervised | 30.7M | 0.091 | 3.59 | 3,153 s | 0.053 → 0.053 flat | 61 / 0 / 0 | 256, 512, 1024 |
| `tf_efficientnet_b3.ns_jft_in1k` | JFT-300M noisy student | 10.7M | 0.084 | 5.34 | 3,020 s | 0.200 → 0.070 falling | 78 / 0 / 26 | 48, 136, 384 (1536 after conv_head) |
| `tiny_vit_21m_224.dist_in22k_ft_in1k` | IN-22k distillation | 20.6M | 0.090 | 5.03 | 3,129 s | 0.205 → 0.208 flat | 27 / 21 / 15 | 192, 384, 576 |
| `inception_next_tiny.sail_in1k` | IN-1k | 25.8M | 0.079 | 3.72 | 2,918 s | 0.101 → 0.094 flat | 22 / 1 / 54 | 192, 384, 768 |
| `hgnetv2_b4.ssld_stage2_ft_in1k` | SSLD | 17.7M | 0.051 | 2.63 | 2,374 s | 0.058 → 0.020 falling | 80 / 0 / 27 | 512, 1024, 2048 |
| `mambaout_tiny.in1k` | IN-1k | 24.2M | 0.095 | 5.07 | 3,239 s | 0.090 → 0.095 flat | 0 / 25 / 18 | 192, 384, 576 |
| `hiera_tiny_224.mae_in1k_ft_in1k` | MAE IN-1k | 27.1M | 0.100 | 4.44 | 3,329 s | 0.033 → 0.052 rising | 0 / 25 / 0 | 192, 384, 768 |
| `eva02_tiny_patch14_224.mim_in22k` | MIM IN-22k | 5.5M | 0.052 | 1.87 | 2,391 s | 0.043 → 0.048 | 0 / 25 / 0 | plain ViT, 192 at 14 |
| `xcit_small_12_p16_224.fb_dist_in1k` | IN-1k distillation | 25.9M | 0.111 | 4.33 | 3,550 s | 0.043 → 0.046 | 16 / 41 / 24 | plain, 384 at 16 |
| `regnety_032.ra_in1k` | IN-1k | 17.9M | 0.123 | 3.25 | 3,777 s | 0.140 → 0.056 falling | 68 / 0 / 0 | 216, 576, 1512 |
| `resnest50d.in1k` | IN-1k | 25.4M | 0.125 | 4.11 | 3,824 s | 0.041 → 0.013 falling | 71 / 0 / 0 | 512, 1024, 2048 |
| `coat_lite_small.in1k` | IN-1k | 19.3M | 0.125 | 5.98 | 3,822 s | 0.070 → 0.061 | 0 / 37 / 16 | no features_only |
| `fastvit_sa24.apple_dist_in1k` | IN-1k distillation | 20.5M | 0.134 | 5.47 | 3,996 s | 0.070 → 0.050 | 128 / 0 / 75 | 128, 256, 512 |
| `repvit_m2_3.dist_450e_in1k` | IN-1k distillation | 22.4M | 0.136 | 7.62 | 4,034 s | 0.093 → 0.089 | 218 / 0 / 105 | 160, 320, 640 |
| `caformer_s18.sail_in22k_ft_in1k` | IN-22k | 23.2M | 0.147 | 7.93 | 4,250 s | 0.043 flat | 0 / 41 / 6 | 128, 320, 512 |
| `convformer_s18.sail_in22k_ft_in1k` | IN-22k | 23.7M | 0.155 | 8.69 | 4,398 s | 0.052 → 0.049 | 0 / 41 / 18 | 128, 320, 512 |
| `convnextv2_tiny.fcmae_ft_in22k_in1k` | FCMAE + IN-22k | 27.9M | 0.175 | 6.96 | 4,790 s | 0.048 → 0.069 rising | 0 / 23 / 18 | 192, 384, 768 |
| `gcvit_tiny.in1k` | IN-1k | 27.7M | 0.181 | 8.54 | 4,921 s | 0.127 → 0.107 | 0 / 71 / 11 | 128, 256, 512 |
| `efficientformerv2_l.snap_dist_in1k` | IN-1k distillation | 25.6M | 0.249 | 7.05 | 6,242 s | 0.112 flat | 182 / 0 / 55 | 80, 192, 384 |

The bound is 4,881 s (1.20 of the reference's 4,067 s). The last six need checkpointing on top of times near or
over the bound and are not seeded. CAFormer and ConvFormer (IN-22k) would fit in time only without the
checkpointing their memory requires.

## Seeded (seed-ideas-12.yaml)

Seven families, chosen for pretraining diversity and fit: FocalNet-T LRF (the large-receptive-field sibling of the
best blend partner), NextViT-S (SSLD 6M), EfficientNet-B3 (JFT noisy student), TinyViT-21M (IN-22k distillation),
InceptionNeXt-T, HGNetV2-B4 (SSLD) and MambaOut-T. Plain ViTs (EVA02-Tiny, XCiT-S12) are left out after LingBot and
RadioDINO trailed with the same adapter; Hiera-T after MedSAM2 and SAM2.1; RegNetY and ResNeSt as ResNet-like
ImageNet-1k CNNs closest to the blend partner.

Starting settings by the campaign's weight-scale rule ([prior work](prior-work.md#starting-settings)): flat RMS
0.05 to 0.107 takes the recipe's rate 2e-4 and anchors 0.001 and 0.00001; other scales move the rate with the RMS
and the anchors with its inverse square against FocalNet-T's 0.047. Ten epochs, ImageNet mean and std (every
candidate's pretrained_cfg).

## Second batch (2026-10-18, after the user asked for exploration over optimisation)

Same measurements. Dropped before timing: Sequencer2D-S, PiT-S and NesT-T (no stride-32 map), Visformer-S (no
feature stages), DeiT3-S (plain ViT, after LingBot and RadioDINO).

| Backbone (timm) | Pretraining | s / 64 | GiB / 64 | Ten epochs | Weight RMS by depth | BN / LN-GN / depthwise | Strides 8, 16, 32 channels |
|---|---|---:|---:|---:|---|---|---|
| `tresnet_m.miil_in21k_ft_in1k` | ImageNet-21k (MIIL) | 0.071 | 2.37 | 2,760 s | 0.073 → 0.061 flat | 60 / 0 / 0 | 128, 1024, 2048 |
| `cs3darknet_l.c2ns_in1k` | IN-1k | 0.049 | 2.68 | 2,333 s | 0.107 → 0.034 falling (median 0.048) | 56 / 0 / 0 | 256, 512, 1024 |
| `ghostnetv2_160.in1k` | IN-1k | 0.072 | 3.39 | 2,785 s | 0.324 → 0.081 falling (median 0.125) | 122 / 0 / 69 | 64, 128, 960 after the last block |
| `twins_svt_small.in1k` | IN-1k | 0.087 | 3.56 | 3,076 s | 0.070 → 0.066 flat | 0 / 48 / 4 | 128, 256, 512 |
| `res2net50d.in1k` | IN-1k | 0.095 | 3.91 | 3,228 s | 0.040 → 0.012 falling (median 0.018) | 87 / 0 / 0 | 512, 1024, 2048 |
| `resnetv2_50x1_bit.goog_in21k_ft_in1k` | ImageNet-21k (BiT) | 0.111 | 6.17 | 3,540 s | 0.034 → 0.014 (weight-standardised) | 0 / 49 GN / 0 | 512, 1024, 2048 |
| `efficientnetv2_rw_s.ra2_in1k` | IN-1k | 0.086 | 5.14 | 3,056 s | 0.114 → 0.057 | 112 / 0 / 30 | not seeded: EfficientNet family already ran |
| `rexnetr_200.sw_in12k_ft_in1k` | ImageNet-12k | 0.077 | 5.83 | 2,885 s | 0.019 flat | 62 / 0 / 16 | not seeded: ReXNet was the old control |
| `regnetz_d8.ra3_in1k` | IN-1k | 0.111 | 7.46 | 3,545 s | 0.111 → 0.046 | 76 / 0 / 0 | not seeded: needs heavy checkpointing |
| `mvitv2_tiny.fb_in1k` | IN-1k | 0.153 | 5.93 | 4,376 s | 0.065 → 0.072 | 0 / 51 / 30 | not seeded: over the bound with checkpointing |
| `mobilevitv2_200.cvnets_in22k_ft_in1k` | IN-22k | 0.133 | 8.25 | 3,979 s | 0.065 → 0.034 | 25 / 21 / 9 | not seeded: memory |
| `poolformerv2_s24.sail_in1k` | IN-1k | 0.157 | 8.60 | 4,450 s | 0.086 → 0.093 | 0 / 53 / 0 | not seeded: memory and time |
| `halonet50ts.a1h_in1k` | IN-1k | – | – | – | – | – | not seeded: failed to build at 224 px |

Seeded (seed-ideas-16.yaml, seed-ideas-17.yaml): TResNet-M, CS3DarkNet-L, GhostNetV2-160, Twins-SVT-S, Res2Net50d and
BiT ResNetV2-50x1, with settings by the same weight-scale rule (falling profiles by their median RMS).
