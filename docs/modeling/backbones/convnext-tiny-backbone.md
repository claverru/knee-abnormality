# ConvNeXt-Tiny backbone

**ConvNeXt-Tiny (`convnext_tiny.in12k_ft_in1k`, timm ImageNet-12k weights fine-tuned on
ImageNet-1k) is one of the three backbone families of the final solution**
([final solution](../../final-solution.md)), trained from `configs/convnext.yaml` and its
teacher-student variants `configs/convnext-{rebuild,v3}-{pretrain,gen2,gen3,gen4}.yaml`,
`configs/convnext-v2v3-gen4.yaml` and `configs/convnext-s188-student.yaml`. The family entered on
2026-10-13 on the fastMRI joint setup and was the best single family on the board from S155
(0.964) to S170 (0.965, five ConvNeXt folds); results per round are in the
[teacher-student results](../../validation/results/teacher-student/INDEX.md).

## Recipe

`configs/convnext.yaml` differs from `configs/resnet50.yaml` in the backbone settings only:

| Setting | ConvNeXt-Tiny | ResNet-50 |
|---|---|---|
| `model.backbone`, `model.pretrained_weights` | `convnext_tiny.in12k_ft_in1k`, `timm` | `resnet50.a1_in1k`, `radimagenet` |
| `data.mean`, `data.std` | ImageNet (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) | 0.3162, 0.3213 |
| `model.stochastic_depth` | 0.1 | 0 |
| `optimizer.backbone_lr` | 1e-4, multiplied by 0.7 per stage toward the stem (`backbone_lr_stage_decay`) | 5e-5 |
| `optimizer.anchor_alpha_early`, `anchor_alpha_late` | 0.01, 0.0001 | 0.01, 0.001 |

The early L2-SP group is the stem and the first stage. The untuned ResNet recipe on ConvNeXt was
stopped after epoch 0 (×1 0.889449 against 0.908666 for the ResNet-50 joint run); the tuned
settings above came from the first joint run (`configs/convnext-tiny-tuned.yaml` (removed) at the time,
fold 3: ×1 0.928074, ×4 0.933246 against the ResNet-50 joint run's 0.926211 and 0.930745).

## Code

- **Token positions:** token centres at 15.5 px with a 32 px pitch for the deep read and 7.5 px
  with 16 px for the stride-16 read, measured on a 2,560 px input as the centre of each token's
  receptive field and equal to the mapping (the ResNet-50's 0 px origins were confirmed by the
  same method). The mask decoder's stride-8 grid sat at 3.5 px with an 8 px pitch while the
  decoder existed (removed 2026-10-14).
- **Stage adapter:** timm's `ConvNeXt` reads through `StemStages` (stem, four stages, final norm;
  [stage adapters](mambavision-backbone.md#code)); the stride-8 feature augmentation (DSU) acts
  after the second stage.
- **New batch sizes:** ConvNeXt pays 4.4 s the first time each batch size is seen and 120 ms
  afterwards (11.7 s with `cudnn.benchmark` on; 214 ms flat with cuDNN off; the ResNet-50 has no
  such cost). The model therefore pads the image count of every ConvNeXt backbone call to a
  multiple of 16 (`CONVNEXT_IMAGE_COUNT_STEP`) with blank images and drops their outputs;
  ConvNeXt has no batch statistics, and the padded outputs match the unpadded ones to 3e-5.
- **ResNet-only adoptions:** the average-pool shortcuts are set by backbone name and apply to the
  ResNet-50 only; ConvNeXt downsamples with 2×2 stride-2 convolutions that drop no pixel. The OA
  detector reads the projected tokens and the knee geometry and applies unchanged. Neither was
  measured on a ConvNeXt.
- **Cost on the 3090** (2026-10-13, timing only): at batch 4, 0.64 s per 8 studies, 4.7 min per
  442-step epoch, 10.6 GB peak ([training time](mambavision-backbone.md#training-time)). The
  first joint run on the fastMRI setup took 46.6 min on the rented machine, with ×4 prediction
  in 91 s and an 18.9 GB peak.
