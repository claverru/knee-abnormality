# Prior work

## The reference

`outputs/training/mambavision-t-rn50-recipe`, trained 2026-10-13 on the 3090 from this worktree:
`mamba_vision_T` with the ImageNet-1k weights and ImageNet mean and std on the ResNet-50 recipe
unchanged (fold 2, seed 1110484336, ten epochs, batch 8, backbone rate 5e-5, head rate 6e-4,
anchors 0.01 early and 0.001 late, no stochastic depth). Training 2,594 s, prediction 64 s, peak
GPU memory 12.7 GB.

Against the ResNet-50 on the same fold (run 27 of targets-and-inputs, 2,696 s):

| | MambaVision-T | ResNet-50 | Δ |
|---|---:|---:|---:|
| ×4 | 0.922130 | 0.939634 | −0.017504 |
| ×1 (epoch 10) | 0.915009 | 0.930902 | −0.015893 |
| Equal logit blend, ×4 | 0.936435 | | −0.003199 |

Per epoch, ×1 validation macro AUC and training loss:

| Epoch | MambaVision-T | ResNet-50 | Δ | Loss | ResNet-50 loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.830513 | 0.895448 | −0.064935 | 0.934 | 0.848 |
| 2 | 0.875855 | 0.910520 | −0.034665 | 0.705 | 0.663 |
| 3 | 0.890679 | 0.912855 | −0.022176 | 0.658 | 0.631 |
| 4 | 0.901194 | 0.916385 | −0.015190 | 0.641 | 0.618 |
| 5 | 0.902778 | 0.921380 | −0.018602 | 0.629 | 0.608 |
| 6 | 0.906077 | 0.926351 | −0.020275 | 0.618 | 0.595 |
| 7 | 0.911292 | 0.927395 | −0.016102 | 0.605 | 0.583 |
| 8 | 0.914144 | 0.929688 | −0.015544 | 0.600 | 0.579 |
| 9 | 0.914230 | 0.930445 | −0.016215 | 0.599 | 0.572 |
| 10 | 0.915009 | 0.930902 | −0.015893 | 0.593 | 0.571 |

**The reference underfits:** its training loss is above the ResNet-50's at every epoch and its
validation AUC is below, so the gap is not overfitting. Epochs take the same time (4.3 against
4.5 minutes with validation).

Per label at ×4, largest gain to largest loss:

| Label | MambaVision-T | ResNet-50 | Δ |
|---|---:|---:|---:|
| Contusion | 0.9509 | 0.9421 | +0.0088 |
| Synovitis | 0.8884 | 0.8962 | −0.0079 |
| Baker's | 0.9497 | 0.9587 | −0.0091 |
| Effusion | 0.9282 | 0.9380 | −0.0098 |
| Medial Meniscus | 0.9479 | 0.9594 | −0.0115 |
| Medial OA | 0.9463 | 0.9654 | −0.0191 |
| MCL | 0.9214 | 0.9416 | −0.0202 |
| Lateral Meniscus | 0.9119 | 0.9344 | −0.0225 |
| ACL | 0.9348 | 0.9589 | −0.0241 |
| PF OA | 0.8828 | 0.9126 | −0.0298 |
| Fracture | 0.9113 | 0.9413 | −0.0300 |
| Lateral OA | 0.8918 | 0.9268 | −0.0349 |

Diffuse findings sit closest and Contusion is ahead; the largest losses are on small, located
structures. The mean correlation of the two models' logits per label is 0.934.

## Why it underfits

Logged by the two runs and measured on the pretrained weights (2026-10-13).

**The weights are larger, so the same rate moves them less.** Adam changes every weight by about
the rate per step whatever its size. Weight RMS per stage of the pretrained backbones (tensors of
two or more dimensions):

| Stage | MambaVision-T | ResNet-50 | Ratio | Rate with the ResNet-50's relative step |
|---|---:|---:|---:|---:|
| Stem | 0.0918 | 0.0623 | 1.5 | 7e-5 |
| 1 | 0.0843 | 0.0230 | 3.7 | 1.8e-4 |
| 2 | 0.0783 | 0.0149 | 5.3 | 2.6e-4 |
| 3 | 0.0791 | 0.0114 | 6.9 | 3.5e-4 |
| 4 | 0.0924 | 0.0073 | 12.7 | 6.3e-4 |

**The backbone barely left ImageNet.** Logged drift from the pretrained weights at the last epoch:

| Stage | MambaVision-T | ResNet-50 |
|---|---:|---:|
| Stem | 0.0109 | 0.0082 |
| 1 | 0.0057 | 0.0176 |
| 2 | 0.0153 | 0.0895 |
| 3 | 0.0138 | 0.1009 |
| 4 | 0.0036 | 0.1056 |

**The anchor is far stronger here.** It penalises the squared absolute distance, so on weights 7
to 12.7 times larger the same relative movement costs 48 to 161 times more. The drift of stages 1
and 4 shrank from epoch 2 on (0.0088 to 0.0057, 0.0053 to 0.0036) as the rate fell. The summed
penalty is alike in both runs (0.016 against 0.021).

**The gradient clip acts on every step.** Gradient norm per epoch: 9.5 falling to 4.3, against
1.0 to 1.5 for the ResNet-50, with the clip at 1.0. The stem's gradient RMS is 15 to 30 times the
ResNet-50's (0.026 to 0.064 against 0.002) and sets the norm.

**The feature-statistics augmentation is stronger.** Logged mean absolute change of the stride-8
features: 0.035 to 0.040, against 0.014 on the ResNet-50.

## Seeds removed before the start

| Seed | Why it was removed |
|---|---|
| Backbone rate 1e-4 | Twice the reference's rate where the weight scale asks for 4 to 13 times; dominated by the 2e-4 run |
| Stage-wise decay at 1e-4 and 0.7 (ConvNeXt's values) | Every stage at or below 1e-4; replaced by the rates measured for this model |
| Late anchor at 0.0001, and no anchors | Replaced by the strengths the weight scale gives (0.001 and 0.00001), which keep the site-generalisation device |
| Fifteen epochs | Twelve by the user's decision |
| Stochastic depth 0.1 | A regulariser on a model that underfits; a subject again once an incumbent overfits |
| Dataset mean and std | The stem is a bias-free convolution followed by BatchNorm in training mode: a shift and scale of the input is removed after the first layer, apart from the padded border |
| Selective scan in float32 | Measured on the trained model, 32 images: stride-32 features differ from float32 by 1.08% under bfloat16 and by 0.81% with the mixers in float32; the ResNet-50 differs by 0.65% |

## How the model differs from the ResNet-50

| Difference | MambaVision-T | ResNet-50 | Setting it makes suspect |
|---|---|---|---|
| Weights | ImageNet-1k, RMS 0.08 to 0.09 | Refined RadImageNet, RMS 0.007 to 0.06 | Rates and anchor strength, both set for the smaller weights |
| Stages 3 and 4 | Mamba mixer and attention, LayerNorm | Convolutions, BatchNorm | Rates, stage-wise decay, stochastic depth |
| Stages 1 and 2 | Convolutions, BatchNorm, 80 and 160 channels | Convolutions, BatchNorm, 256 and 512 | Feature-statistics augmentation acts on 160 channels |
| Final norm | BatchNorm on 640 channels | none | |
| Pretraining regulariser | Stochastic depth 0.2 | none | Run at 0 in the reference |
| Input statistics | ImageNet mean and std | Dataset mean 0.3162, std 0.3213 | None: BatchNorm after the first convolution removes the difference |
| Precision | Scan kernel under bfloat16 | Convolutions under bfloat16 | None: measured, see the removed seeds |
| Weight decay | Replaced by the anchors on every backbone weight | The same | Not a separate subject: anchored weights carry no decay |

## Precedents

| Mechanism | Setting | Run | Macro Δ | Source |
|---|---|---|---:|---|
| Backbone rate | ConvNeXt-Tiny 2e-4 flat with stochastic depth 0.1, 16 epochs | fold 0, 2026-10-04 | −0.0048 against 1e-4 | ConvNeXt line notes |
| Stage-wise rate decay | ConvNeXt-Tiny 1e-4 with 0.7 per stage, stochastic depth 0.1, late anchor 0.0001 | fold 3 joint, 2026-10-13 | ×4 0.933246 against the ResNet-50 joint 0.930745 | fastMRI worktree, `configs/convnext-tiny-tuned.yaml` |
| Rerun distance | ResNet-50 recipe twice on fold 2 | targets-and-inputs | 0.0010 at ×4, 0.0013 at ×1 | `docs/validation/results/campaigns-and-queues/2026-10-13-targets-and-inputs-campaign.md` |
| Authors' fine-tuning rate | MambaVision detection and segmentation at 1e-4 | none here | | recalled from the paper, not rechecked |

The ConvNeXt-Tiny rows are another model and, for the second, another fold and training set: they
give values to try, not expected effects.

## Code the seeds can port

- Stage-wise rate decay (`optimizer.backbone_lr_stage_decay`) exists in the uncommitted
  `training/lightning.py` and `config.py` of the worktree `knee-abnormality-fastmri-pretraining`.
- `model.stochastic_depth` already exists here; for this backbone it spreads the rate linearly
  over the model's DropPath modules.

## Blocked directions

Model size, other pretrained weights, fastMRI pretraining, image size, slice counts and
ensembling are outside this campaign ([contract](contract.md#out-of-bounds)).

**Generalisation rule, scope (the user, 2026-10-14):** the standing rule that holds out of adoption any
change that could cost generalisation to unseen sites applies only to models trained on RadImageNet
weights. It does not apply to this campaign's ImageNet-pretrained MambaVision-T: anchor schedules
and strengths and the robustness augmentations are open subjects here, adopted by the shared rule
like any other setting.
