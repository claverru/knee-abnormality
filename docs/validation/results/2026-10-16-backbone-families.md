# 2026-10-16 Backbone families

Ten backbone families different in kind from the ResNet-50, ConvNeXt-Tiny and MambaVision-T, each
trained on fold 2 of `all_5fold_scanner_v2` (seed 1110484336) on the MambaVision-T recipe with
settings from its own weight scale, in the
[backbone-families campaign](../../../autoresearch/campaigns/backbone-families/INDEX.md)
(2026-10-15 18:11 to 2026-10-16 15:12 CEST, Codex, 33 runs). Reference: the decoder-free
MambaVision-T `mamba-silver-fold-2`, ×4 0.935160, ×1 0.930081. FocalNet-T SRF became the
campaign's incumbent; EfficientViT-B2 with stochastic depth 0.2 came level with it and was trained
on all five folds. Both are merged as `configs/focalnet.yaml` and `configs/efficientvit.yaml`
([code](../../modeling/focalnet-and-efficientvit-backbones.md)). Run details are in the campaign's
[run notes](../../../autoresearch/campaigns/backbone-families/run-notes.md).

## Families on fold 2

×4 macro AUC; Δ against the FocalNet-T incumbent (twelve epochs).

| Family | Epochs | ×4 | Δ | Notes |
|---|---:|---:|---:|---|
| FocalNet-T SRF (`focalnet_tiny_srf.ms_in1k`) | 12 | 0.938816 | incumbent | 1×1 convolutions as linear layers, 4,067 s, 23.0 GB |
| EfficientViT-B2, stochastic depth 0.2 | 10 | 0.938773 | −0.00004 | 80% interval −0.0018 to +0.0018, 3,096 s, 19.0 GiB |
| FocalNet-T SRF, early anchor following the rate | 10 | 0.938806 | −0.00001 | 3,308 s |
| FocalNet-T SRF | 10 | 0.938154 | −0.0007 | |
| EfficientViT-B2 | 10 | 0.937280 | −0.0015 | real-row BatchNorm, stages 0 and 1 checkpointed |
| DaViT-T | 10 | 0.936669 | −0.0021 | |
| EdgeNeXt-B | 10 | 0.936033 | −0.0028 | cheapest, 2,798 s |
| WTConvNeXt-T | 9 | 0.935647 | −0.0032 | vendored Triton WTConv |
| MambaVision-T (reference) | 12 | 0.935160 | −0.0037 | |
| DINOv3 ConvNeXt-T | 10 | 0.934199 | −0.0046 | |
| PVTv2-b2 | 10 | 0.933099 | −0.0057 | stage 0 checkpointed |
| SAM2.1 Hiera-T, HRNet-W18, RDNet-T | | | | aborted at epoch 7 (−0.021 ×1), over the time bound, blocked over the bound |

Every challenger led or matched FocalNet-T at ×1 in the early epochs and lost ground in the last
ones. Effusion is the label most of them gained on (DaViT-T, EdgeNeXt-B, EfficientViT-B2,
WTConvNeXt-T), FocalNet-T's weakest against MambaVision-T.

**Out of GPU memory under WSL shows as `CUDA driver error: device not ready`:** PVTv2-b2 (22.3 GB,
backbone alone at 256 images) and RDNet-T (20.4 GB) exceed the 3090 in training; activation
checkpointing of their early stages fits them. From run 11 a preflight measured every candidate on
60 real training batches before it trained.

## EfficientViT-B2 settings

On top of real-row BatchNorm padding and late anchor 3e-5, ×4 against FocalNet-T's 0.938816:

| Run | ×4 | ×1 | Training |
|---|---:|---:|---:|
| Plain, stages 0 and 1 checkpointed | 0.937280 | 0.931092 | 3,406 s, 11.4 GiB |
| Batch Renormalization | 0.936045 | 0.930695 | 3,076 s |
| LayerNorm on the stride-16 and stride-32 reads | 0.936863 | 0.929011 | |
| bfloat16 attention products | 0.937130 | 0.930151 | 3,319 s, 10.4 GiB |
| Stochastic depth 0.1 | 0.938641 | 0.929984 | 3,633 s, 11.4 GiB |
| Stochastic depth 0.1, stage 0 checkpointed, bfloat16 attention | 0.937366 | 0.929824 | 3,244 s, 14.8 GiB |
| **Stochastic depth 0.2, stage 0's first block checkpointed, bfloat16 attention** | **0.938773** | 0.930888 | 3,096 s, 19.0 GiB |

Stochastic depth is the one setting that helped (+0.0015 at 0.2). The merged version keeps it
with float32 attention (timm's own LiteMLA), since Kaggle's T4s have no bfloat16; float32
attention needs about 1 GiB more, so the merged code checkpoints all of stage 0 (17.0 GiB and
0.662 s per step in the preflight of run 31, against 0.604 s for the five-fold version).

Per label at ×4 against FocalNet-T, largest gain to largest loss: Contusion +0.0091, Medial OA
+0.0053, Fracture +0.0048, ACL +0.0041; the largest losses Lateral Meniscus −0.0075 and Lateral
OA −0.0130.

## Five folds

EfficientViT-B2 with stochastic depth 0.2 trained folds 0, 1, 3 and 4 locally from the fold-2 run's
frozen source (`outputs/training/effvit-b2-best-folds-0-1-3-4`, about 3,070 s and 20.4 GB per
fold); fold 2 is the campaign run. FocalNet-T's silver folds 0, 1, 3 and 4 trained twelve epochs
on the rented machine (`outputs/training/focalnet-silver-fold-{k}`).

| Fold | EfficientViT-B2 ×1 | ×4 |
|---:|---:|---:|
| 0 | 0.932040 | 0.937628 |
| 1 | 0.925873 | 0.930458 |
| 2 | 0.930888 | 0.938773 |
| 3 | 0.926042 | 0.932275 |
| 4 | 0.924680 | 0.930997 |

Out-of-fold macro AUC of the silver teachers (4,407 studies; gold: the 58 gold studies), equal
logit average for blends:

| Family or blend | Silver | Gold |
|---|---:|---:|
| ResNet-50 | 0.934632 | 0.938411 [0.915, 0.958] |
| FocalNet-T | 0.932771 | 0.930760 |
| EfficientViT-B2 | 0.932378 | 0.921699 [0.893, 0.947] |
| ConvNeXt-Tiny | 0.932376 | 0.937444 |
| MambaVision-T | 0.931554 | 0.928401 |
| ResNet-50 + ConvNeXt | 0.936153 | 0.941863 |
| ResNet-50 + ConvNeXt + FocalNet + EfficientViT | 0.937121 | 0.937783 |
| All five | 0.937101 | 0.937381 |

EfficientViT-B2 pairs as well as FocalNet-T on silver (+0.0020 over the ResNet-50 alone, the most of
any family) but loses on gold in every blend (−0.0036 left out of the five); its gold losses sit on
Lateral Meniscus (0.868 against about 0.907), Lateral OA (0.849 against about 0.90) and Medial OA.
Logit correlation with the other families 0.955 to 0.968, highest with MambaVision-T. By the
teacher rule (best gold among blends within 0.001 of the best silver), ResNet-50 + ConvNeXt stays
the teacher pair; EfficientViT-B2 is untested as a student.
