# Fast backbone search

Outside MRI evidence is in [backbone MRI evidence](../../research/backbones-and-pretraining/backbone-mri-evidence.md); cost and
code requirements for a swap are in [swapping a backbone](backbone-swaps.md).

An unverified experiment used three models' out-of-fold predictions with
leave-one-out inference to identify an important three-slice window.
A reduced dataset mapping one selected window to study targets let candidate
backbones train in about 5–10 minutes. This reportedly helped discover the
`rexnetr300` candidate and transferred to the full task.

A proposed extension would select one important window per target, about
12 windows per study. That extension was not reported as completed. Selection
details, ranking aggregation across models, and validation isolation are missing.

## Finished model comparison

The following unverified completed-run summary was recorded on 2026-09-11 as
source `target-specialists-20260911-v1`. Best AUC is the unweighted mean of the
twelve target AUCs on the 58 gold studies of the
[validation protocol](../../validation/validation-protocol.md), and each
run's checkpoint was selected by that mean. Training configuration, split, seed
and artifact identities are unavailable, so these values rank that experiment
set but do not establish a controlled comparison with repository baseline3.
ReXNet-200 has the highest reported AUC and is the next maintained
baseline-config backbone.

| # | Finished model | ID | Best AUC | Selected epoch | Epochs completed | Cap | Run time | Stop reason |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | ReXNet-200 | `rexnet200` | *0.908347* | 5 | 11 | 20 | 13m 46s | Patience 6 |
| 2 | DINOv3 ConvNeXt Tiny | `dinov3_convnext_tiny` | 0.904386 | 7 | 13 | 20 | 11m 33s | Patience 6 |
| 3 | CoAtNet | `coatnet` | 0.904336 | 5 | 11 | 20 | 40m 17s | Patience 6 |
| 4 | RegNetY-160 IN12k | `regnety_160_in12k` | 0.901071 | 14 | 20 | 30 | 46m 52s | Patience 6 |
| 5 | ReXNet-200 ordered map mixer | `rexnet200_ordered_map_mixer` | 0.897025 | 6 | 12 | 30 | 14m 51s | Patience 6 |
| 6 | DINOv2 Base reg4 | `dinov2_base_reg4` | 0.895418 | 5 | 11 | 30 | 22m 53s | Patience 6 |
| 7 | ReXNet-300 | `rexnet300` | 0.894535 | 8 | 14 | 20 | 24m 40s | Patience 6 |
| 8 | RegNetY-120 IN12k | `regnety_120_in12k` | 0.894422 | 6 | 12 | 30 | 23m 46s | Patience 6 |
| 9 | DINOv3 ConvNeXt Small | `dinov3_convnext_small` | 0.890972 | 8 | 14 | 30 | 19m 04s | Patience 6 |
| 10 | RegNetY-064 IN1k | `regnety_064_in1k` | 0.890054 | 12 | 18 | 30 | 27m 41s | Patience 6 |
| 11 | ReXNet-200 ordered window mixer | `rexnet200_ordered_window_mixer` | 0.888753 | 9 | 15 | 30 | 18m 04s | Patience 6 |
| 12 | DINOv3 Small gated, cap 30 | `dinov3_small_gated_epochs30` | 0.888279 | 12 | 18 | 30 | 13m 56s | Patience 6 |
| 13 | DINOv3 Small mean/max, cap 30 | `dinov3_small_meanmax_epochs30` | 0.887999 | 18 | 24 | 30 | 17m 40s | Patience 6 |
| 14 | ConvNeXt Tiny supervised | `convnext_tiny_supervised` | 0.886954 | 9 | 15 | 30 | 13m 14s | Patience 6 |
| 15 | DINOv3 Small gated | `dinov3_small_gated` | 0.886726 | 19 | 20 | 20 | 15m 16s | Epoch cap |
| 16 | DINOv3 Base | `dinov3_base` | 0.885491 | 5 | 11 | 30 | 20m 00s | Patience 6 |
| 17 | DINOv3 Small gated acquisition | `dinov3_small_gated_acquisition` | 0.884663 | 22 | 28 | 30 | 21m 27s | Patience 6 |
| 18 | ReXNet-200 spatial queries | `rexnet200_spatial_queries` | 0.883935 | 10 | 16 | 30 | 20m 46s | Patience 6 |
| 19 | RadImageNet ResNet-50 | `radimagenet_resnet50` | 0.883363 | 29 | 30 | 30 | 25m 13s | Epoch cap |
| 20 | DINOv3 Small gated axis, cap 30 | `dinov3_small_gated_axis_epochs30` | 0.882248 | 19 | 25 | 30 | 19m 33s | Patience 6 |
| 21 | DINOv3 Small gated axis | `dinov3_small_gated_axis` | 0.880551 | 19 | 20 | 20 | 15m 20s | Epoch cap |
| 22 | DINOv3 Small mean/max | `dinov3_small_meanmax` | 0.879972 | 19 | 20 | 20 | 14m 29s | Epoch cap |
| 23 | DINOv3 Small Plus | `dinov3_small_plus` | 0.878961 | 7 | 13 | 30 | 11m 26s | Patience 6 |
| 24 | CAFormer-S18 IN22k | `caformer_s18_in22k` | 0.878179 | 15 | 21 | 30 | 36m 16s | Patience 6 |
| 25 | MedDINOv3 Base | `meddinov3_base` | 0.875170 | 5 | 11 | 30 | 20m 04s | Patience 6 |
| 26 | DINOv2 Small reg4 | `dinov2_small_reg4` | 0.874858 | 8 | 14 | 20 | 10m 43s | Patience 6 |
| 27 | DINOv3 Small | `dinov3_small` | 0.874582 | 19 | 20 | 20 | 14m 44s | Epoch cap |
| 28 | InceptionNeXt Tiny | `inceptionnext_tiny` | 0.872661 | 7 | 13 | 30 | 12m 08s | Patience 6 |
| 29 | DINOv3 Small last6 | `dinov3_small_last6` | 0.871846 | 10 | 16 | 30 | 7m 55s | Patience 6 |
| 30 | ConvNeXt-V2 Tiny FCMAE | `convnextv2_tiny_fcmae` | 0.871493 | 9 | 15 | 30 | 33m 00s | Patience 6 |
| 31 | ImageNet DenseNet-121 | `imagenet_densenet121` | 0.869147 | 8 | 14 | 30 | 16m 19s | Patience 6 |
| 32 | DINOv3 Small, cap 30 | `dinov3_small_epochs30` | 0.868904 | 7 | 13 | 30 | 9m 48s | Patience 6 |
| 33 | DINOv3 Large LoRA32 | `dinov3_large_lora32` | 0.866161 | 13 | 19 | 30 | 2h 21m 24s | Patience 6 |
| 34 | DINOv3 Small CLS + mean | `dinov3_small_cls_mean` | 0.865789 | 5 | 11 | 20 | 8m 12s | Patience 6 |
| 35 | BiomedCLIP Base | `biomedclip_base` | 0.864368 | 6 | 12 | 30 | 17m 53s | Patience 6 |
| 36 | ImageNet ResNet-50 | `imagenet_resnet50` | 0.861902 | 8 | 14 | 30 | 11m 53s | Patience 6 |
| 37 | MaxViT Tiny | `maxvit_tiny` | 0.857384 | 12 | 18 | 30 | 42m 14s | Patience 6 |
| 38 | EfficientNetV2-S IN21k | `efficientnetv2_s_in21k` | 0.857114 | 7 | 13 | 30 | 15m 16s | Patience 6 |
| 39 | RadImageNet DenseNet-121 | `radimagenet_densenet121` | 0.816707 | 24 | 30 | 30 | 32m 56s | Patience 6 |
| 40 | RadiologyNet ResNet-50 | `radiologynet_resnet50` | 0.791780 | 5 | 11 | 30 | 9m 43s | Patience 6 |

## Per-target AUC

Target AUCs of the same selected checkpoints on the same 58 studies, in overall
rank order. **Bold** marks the target leader. Every AUC is a multiple of
1/(2 × positives × negatives), so four decimals identify each exact value.

| # | Model | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Effusion | Synovitis | Baker's | Contusion | Fracture |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ReXNet-200 | 0.9485 | 0.9433 | 0.8942 | 0.8273 | 0.9519 | 0.8878 | 0.8430 | 0.9702 | 0.8244 | 0.9728 | 0.8866 | 0.9500 |
| 2 | DINOv3 ConvNeXt Tiny | 0.9608 | 0.9138 | 0.8341 | 0.8658 | 0.9721 | **0.8975** | 0.8160 | 0.9665 | 0.8303 | 0.9475 | 0.9177 | 0.9306 |
| 3 | CoAtNet | 0.9534 | **0.9637** | 0.8522 | 0.8472 | 0.9814 | 0.8472 | 0.8082 | 0.9665 | 0.8363 | 0.9312 | 0.9231 | 0.9417 |
| 4 | RegNetY-160 IN12k | **0.9963** | 0.8889 | 0.8942 | 0.8062 | 0.9907 | 0.8143 | 0.8494 | 0.9615 | 0.7873 | 0.9384 | 0.9244 | 0.9611 |
| 5 | ReXNet-200 ordered map mixer | 0.9105 | 0.9478 | 0.8870 | 0.8273 | 0.9690 | 0.8395 | 0.8649 | 0.9752 | 0.8280 | 0.8841 | 0.8866 | 0.9444 |
| 6 | DINOv2 Base reg4 | 0.9167 | 0.9456 | 0.8089 | 0.8373 | 0.9333 | 0.8375 | 0.8237 | 0.9851 | 0.8483 | 0.9475 | 0.9001 | 0.9611 |
| 7 | ReXNet-300 | 0.9645 | 0.9025 | 0.8678 | 0.8559 | 0.9628 | 0.8704 | 0.7992 | 0.9255 | 0.7599 | 0.9348 | 0.9190 | **0.9722** |
| 8 | RegNetY-120 IN12k | 0.9240 | 0.9138 | 0.8870 | 0.8547 | 0.9721 | 0.7814 | 0.8584 | 0.9615 | 0.7849 | 0.9112 | 0.9325 | 0.9514 |
| 9 | DINOv3 ConvNeXt Small | 0.9007 | 0.8073 | 0.8726 | 0.8596 | 0.9628 | 0.8646 | 0.8391 | 0.9863 | 0.8411 | 0.9565 | 0.9190 | 0.8819 |
| 10 | RegNetY-064 IN1k | 0.9350 | 0.9410 | 0.8425 | 0.8646 | 0.9907 | 0.8298 | 0.7864 | 0.9441 | 0.8208 | 0.9076 | 0.9042 | 0.9139 |
| 11 | ReXNet-200 ordered window mixer | 0.8848 | 0.8912 | 0.8966 | 0.8087 | 0.9876 | 0.8588 | 0.8430 | 0.9540 | 0.7742 | 0.9312 | 0.8961 | 0.9389 |
| 12 | DINOv3 Small gated, cap 30 | 0.8873 | 0.8957 | 0.8125 | 0.8348 | 0.9783 | 0.7950 | 0.8378 | 0.9851 | 0.8280 | 0.9819 | 0.9231 | 0.9000 |
| 13 | DINOv3 Small mean/max, cap 30 | 0.8860 | 0.8912 | 0.8233 | 0.8099 | 0.9907 | 0.7814 | 0.8468 | 0.9814 | **0.8530** | 0.9764 | 0.9136 | 0.9021 |
| 14 | ConvNeXt Tiny supervised | 0.9007 | 0.8934 | 0.8510 | 0.8373 | 0.9752 | 0.8375 | 0.8185 | 0.9441 | 0.8411 | 0.9275 | 0.8907 | 0.9264 |
| 15 | DINOv3 Small gated | 0.8824 | 0.9093 | 0.8197 | 0.7714 | 0.9752 | 0.8201 | 0.8430 | 0.9826 | 0.8196 | 0.9819 | 0.9244 | 0.9111 |
| 16 | DINOv3 Base | 0.8775 | 0.9002 | 0.8305 | **0.8919** | 0.9721 | 0.7776 | 0.7941 | 0.9665 | 0.8268 | 0.9312 | 0.9285 | 0.9292 |
| 17 | DINOv3 Small gated acquisition | 0.8848 | 0.9138 | 0.8113 | 0.8025 | 0.9783 | 0.7814 | 0.8546 | 0.9689 | 0.8196 | 0.9837 | 0.9406 | 0.8764 |
| 18 | ReXNet-200 spatial queries | 0.9436 | 0.9048 | **0.9075** | 0.7839 | 0.9721 | 0.7814 | 0.8185 | 0.9491 | 0.7491 | 0.9094 | 0.9393 | 0.9486 |
| 19 | RadImageNet ResNet-50 | 0.8725 | 0.9093 | 0.8221 | 0.7590 | 0.9550 | 0.7911 | **0.8700** | 0.9627 | 0.8232 | 0.9239 | 0.9420 | 0.9694 |
| 20 | DINOv3 Small gated axis, cap 30 | 0.8897 | 0.9070 | 0.8413 | 0.7888 | 0.9752 | 0.7756 | 0.8533 | 0.9540 | 0.8088 | 0.9674 | 0.9285 | 0.8972 |
| 21 | DINOv3 Small gated axis | 0.8873 | 0.9297 | 0.8137 | 0.7839 | 0.9721 | 0.7756 | 0.8366 | 0.9739 | 0.8053 | 0.9710 | 0.9190 | 0.8986 |
| 22 | DINOv3 Small mean/max | 0.8873 | 0.8900 | 0.8173 | 0.7932 | 0.9783 | 0.7718 | 0.8327 | 0.9714 | 0.8483 | 0.9647 | 0.9069 | 0.8979 |
| 23 | DINOv3 Small Plus | 0.8934 | 0.8095 | 0.8257 | 0.8708 | 0.9938 | 0.7621 | 0.7979 | 0.9764 | 0.8303 | 0.9420 | 0.9163 | 0.9292 |
| 24 | CAFormer-S18 IN22k | 0.9240 | 0.8776 | 0.8077 | 0.8857 | 0.9240 | 0.8743 | 0.7799 | 0.9466 | 0.8076 | 0.9438 | 0.8502 | 0.9167 |
| 25 | MedDINOv3 Base | 0.9252 | 0.8639 | 0.7452 | 0.7640 | 0.9566 | 0.8549 | 0.8211 | 0.9491 | 0.8339 | 0.9457 | 0.9271 | 0.9153 |
| 26 | DINOv2 Small reg4 | 0.8787 | 0.9161 | 0.8293 | 0.8025 | 0.9659 | 0.7505 | 0.8417 | 0.9491 | 0.8363 | 0.8986 | 0.8866 | 0.9431 |
| 27 | DINOv3 Small | 0.8885 | 0.8980 | 0.8053 | 0.7739 | 0.9953 | 0.7698 | 0.8327 | 0.9652 | 0.7646 | 0.9565 | 0.9298 | 0.9153 |
| 28 | InceptionNeXt Tiny | 0.8725 | 0.9093 | 0.7680 | 0.8248 | 0.9659 | 0.7834 | 0.8211 | 0.9714 | 0.7766 | 0.9710 | 0.8745 | 0.9333 |
| 29 | DINOv3 Small last6 | 0.8676 | 0.8594 | 0.7849 | 0.8025 | **1.0000** | 0.7988 | 0.7773 | 0.9540 | 0.7814 | **0.9855** | 0.9298 | 0.9208 |
| 30 | ConvNeXt-V2 Tiny FCMAE | 0.8824 | 0.8277 | 0.8293 | 0.8671 | 0.9752 | 0.7756 | 0.8288 | 0.9379 | 0.7814 | 0.9275 | 0.9028 | 0.9222 |
| 31 | ImageNet DenseNet-121 | 0.9216 | 0.9229 | 0.7969 | 0.8075 | 0.9628 | 0.7795 | 0.8082 | 0.9379 | 0.7336 | 0.9076 | 0.9055 | 0.9458 |
| 32 | DINOv3 Small, cap 30 | 0.8836 | 0.8503 | 0.7909 | 0.7652 | 0.9922 | 0.7505 | 0.8185 | 0.9677 | 0.8112 | 0.9493 | **0.9474** | 0.9000 |
| 33 | DINOv3 Large LoRA32 | 0.8922 | 0.8934 | 0.7392 | 0.7677 | 0.9597 | 0.7814 | 0.8366 | **0.9913** | 0.7121 | 0.9529 | 0.9231 | 0.9444 |
| 34 | DINOv3 Small CLS + mean | 0.9007 | 0.8844 | 0.7692 | 0.8137 | 0.9643 | 0.7292 | 0.7992 | 0.9503 | 0.7718 | 0.9656 | 0.9271 | 0.9139 |
| 35 | BiomedCLIP Base | 0.8811 | 0.8458 | 0.7740 | 0.7826 | 0.9504 | 0.7505 | 0.8018 | 0.9689 | 0.8435 | 0.9384 | 0.8839 | 0.9514 |
| 36 | ImageNet ResNet-50 | 0.8873 | 0.9456 | 0.8269 | 0.7752 | 0.9535 | 0.7389 | 0.8250 | 0.9304 | 0.7599 | 0.9167 | 0.8947 | 0.8889 |
| 37 | MaxViT Tiny | 0.8272 | 0.8163 | 0.8534 | 0.8124 | 0.9953 | 0.7447 | 0.7992 | 0.9317 | 0.7551 | 0.9058 | 0.8961 | 0.9514 |
| 38 | EfficientNetV2-S IN21k | 0.8517 | 0.8231 | 0.7620 | 0.7354 | 0.9550 | 0.7892 | 0.8263 | 0.9379 | 0.7754 | 0.9764 | 0.9001 | 0.9528 |
| 39 | RadImageNet DenseNet-121 | 0.7855 | 0.7755 | 0.6971 | 0.7354 | 0.8806 | 0.6983 | 0.8391 | 0.9752 | 0.7790 | 0.8333 | 0.9028 | 0.8986 |
| 40 | RadiologyNet ResNet-50 | 0.7525 | 0.7483 | 0.6466 | 0.6422 | 0.9364 | 0.6750 | 0.8404 | 0.9553 | 0.8136 | 0.7826 | 0.8570 | 0.8514 |

## Target leaders

Runner-up and best overall top 10 list every tied model. Gain over top 10 is
the leader's AUC minus the best AUC among overall ranks 1–10; zero means a
top-10 model leads.

| Target | Positive / negative | Leader (overall #) | AUC | Runner-up | AUC | Best overall top 10 | AUC | Gain over top 10 |
|---|---:|---|---:|---|---:|---|---:|---:|
| ACL | 24 / 34 | RegNetY-160 IN12k (#4) | 0.9963 | ReXNet-300 | 0.9645 | RegNetY-160 IN12k | 0.9963 | 0.0000 |
| MCL | 9 / 49 | CoAtNet (#3) | 0.9637 | ReXNet-200 ordered map mixer | 0.9478 | CoAtNet | 0.9637 | 0.0000 |
| Medial Meniscus | 26 / 32 | ReXNet-200 spatial queries (#18) | 0.9075 | ReXNet-200 ordered window mixer | 0.8966 | ReXNet-200, RegNetY-160 IN12k | 0.8942 | +0.0132 |
| Lateral Meniscus | 23 / 35 | DINOv3 Base (#16) | 0.8919 | CAFormer-S18 IN22k | 0.8857 | DINOv3 ConvNeXt Tiny | 0.8658 | +0.0261 |
| Medial OA | 15 / 43 | DINOv3 Small last6 (#29) | 1.0000 | DINOv3 Small, MaxViT Tiny | 0.9953 | RegNetY-160 IN12k, RegNetY-064 IN1k | 0.9907 | +0.0093 |
| Lateral OA | 11 / 47 | DINOv3 ConvNeXt Tiny (#2) | 0.8975 | ReXNet-200 | 0.8878 | DINOv3 ConvNeXt Tiny | 0.8975 | 0.0000 |
| PF OA | 21 / 37 | RadImageNet ResNet-50 (#19) | 0.8700 | ReXNet-200 ordered map mixer | 0.8649 | ReXNet-200 ordered map mixer | 0.8649 | +0.0051 |
| Effusion | 35 / 23 | DINOv3 Large LoRA32 (#33) | 0.9913 | DINOv3 ConvNeXt Small | 0.9863 | DINOv3 ConvNeXt Small | 0.9863 | +0.0050 |
| Synovitis | 27 / 31 | DINOv3 Small mean/max, cap 30 (#13) | 0.8530 | DINOv2 Base reg4, DINOv3 Small mean/max | 0.8483 | DINOv2 Base reg4 | 0.8483 | +0.0048 |
| Baker's | 12 / 46 | DINOv3 Small last6 (#29) | 0.9855 | DINOv3 Small gated acquisition | 0.9837 | ReXNet-200 | 0.9728 | +0.0127 |
| Contusion | 19 / 39 | DINOv3 Small, cap 30 (#32) | 0.9474 | RadImageNet ResNet-50 | 0.9420 | RegNetY-120 IN12k | 0.9325 | +0.0148 |
| Fracture | 18 / 40 | ReXNet-300 (#7) | 0.9722 | RadImageNet ResNet-50 | 0.9694 | ReXNet-300 | 0.9722 | 0.0000 |

### Uncertainty versus ReXNet-200

Each target used 2,000 paired stratified bootstrap resamples with the trained
checkpoints fixed. The conditional 95% interval is for the leader's AUC minus
ReXNet-200's; **bold** intervals exclude zero. Leader first and top 3 are
shares of resamples; non-integer first-place counts show that tied first
places are shared. ReXNet-200's target # counts models with strictly higher AUC
among all 40, so ties share a position. Resampling does not correct model or
epoch selection or pipeline dependence.

| Target | ReXNet-200 AUC (target #) | Leader − ReXNet-200 | Conditional 95% interval | Leader first | Leader top 3 |
|---|---:|---:|---|---:|---:|
| ACL | 0.9485 (5) | +0.0478 | **[+0.0086, +0.1005]** | 97.3% | 100.0% |
| MCL | 0.9433 (5) | +0.0204 | [−0.0317, +0.0816] | 43.1% | 69.9% |
| Medial Meniscus | 0.8942 (3) | +0.0132 | [−0.0529, +0.0805] | 28.2% | 58.1% |
| Lateral Meniscus | 0.8273 (14) | +0.0646 | [−0.0447, +0.1789] | 30.3% | 55.8% |
| Medial OA | 0.9519 (35) | +0.0481 | **[+0.0078, +0.1070]** | 37.8% | 100.0% |
| Lateral OA | 0.8878 (2) | +0.0097 | [−0.0465, +0.0600] | 30.6% | 74.4% |
| PF OA | 0.8430 (8) | +0.0270 | [−0.0476, +0.1120] | 22.9% | 43.4% |
| Effusion | 0.9702 (13) | +0.0211 | [−0.0075, +0.0621] | 40.2% | 66.4% |
| Synovitis | 0.8244 (15) | +0.0287 | [−0.0263, +0.0908] | 10.1% | 34.3% |
| Baker's | 0.9728 (7) | +0.0127 | [−0.0163, +0.0507] | 27.0% | 53.8% |
| Contusion | 0.8866 (34) | +0.0607 | [−0.0094, +0.1498] | 31.6% | 61.3% |
| Fracture | 0.9500 (9) | +0.0222 | [−0.0167, +0.0695] | 32.7% | 62.9% |

## Reading the results

- Only ACL and Medial OA have intervals that exclude zero. RegNetY-160 IN12k
  leads ACL in 97.3% of resamples. DINOv3 Small last6 ranks Medial OA perfectly
  but leads only 37.8% of resamples, while staying in every top three;
  DINOv3 Small and MaxViT Tiny reach 0.9953.
- The largest point gains over ReXNet-200 are Lateral meniscus (+0.0646) and
  Contusion (+0.0607); both intervals include zero. ReXNet-200 ranks 14th and
  34th on those targets and 35th on Medial OA.
- ReXNet-200 leads no target, and eight of the twelve leaders rank outside the
  overall top 10. Its lowest target AUCs are Synovitis (0.8244), Lateral
  meniscus (0.8273) and PF OA (0.8430).
- Synovitis has the least stable leader: first in 10.1% of resamples and in
  the top three in 34.3%.

## Limitations and provenance

- A target leader is a candidate for further evaluation. Choosing winners on
  these 58 studies is not independent validation, and repeated model and epoch
  selection adds optimism.
- Counts are small: MCL has 9 positives and Lateral OA 11. A 1.0000 AUC only
  means perfect ranking on this sample.
- The original input-selector pipeline, likely the window selection above, was
  not fully independent of these 58 studies.
- Training labels were image-derived.
- Epoch caps were 20 or 30, as listed in the finished-model table.
- AUC measures ranking, not accuracy at a threshold.
- The set excludes newer single-slice and anchor-contrast experiments and a
  skipped OrthoFoundation attempt.

The per-target, leader and bootstrap values were imported on 2026-09-16 from an
untracked exploratory page, `target-performance.html`, titled "Original knee
MRI benchmark". It contained aggregate results only and described its 480
model-target AUCs as verified by the source. Import checks found that its
twelve positive counts match the fully labelled rows of `data/raw/train.csv`,
that each model's twelve-target mean reproduces its Best AUC to six decimals,
and that the reported leaders, runners-up, best top-10 AUCs and ReXNet-200 AUCs
agree with the per-target table.
