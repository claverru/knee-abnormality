# RadioDINO settings: prior work

Measured 2026-10-14 on the pretrained weights and the recorded runs; no RadioDINO training yet.

## The model

[Snarcy/RadioDino-s16](https://huggingface.co/Snarcy/RadioDino-s16), revision
`773d28d873455153495d39179fc42ea816aa1212`: DINO ViT-S/16 pretrained on RadImageNet (the card
reports 1.35M CT, MRI and ultrasound images), CC BY 4.0. It loads strictly into timm's
`vit_small_patch16_224.dino`. The choice among medical encoders (RAD-DINO, MedSigLIP,
BiomedCLIP, Curia, MRI-CORE, OrthoFoundation, OmniRad) is in
[Hugging Face image encoders](../../../docs/research/huggingface-image-encoders.md#practical-2d-shortlist):
it is the only one at our size and speed.

## The adapter

`radiodino_s16` in `src/knee_abnormality/modeling/staged_backbones.py`
([notes](../../../docs/ongoing/radiodino-backbone.md)):

- The stem is the 16×16 patch convolution plus the patch position embeddings; the CLS token and
  its position embedding are dropped.
- Twelve blocks in stages of 1, 2, 5 and 4. The stride-8 point, where feature-statistics
  augmentation acts, is after block 3 (a 14×14 grid at stride 16). The stride-16 middle read is
  after block 8. The final read is block 12, the ViT norm and a 2×2 average pool to 7×7 (origin
  15.5, stride 32, ConvNeXt-T's anchors).
- The early anchor covers the stem and block 1; the late anchor every other two-dimensional
  weight. Anchored weights carry no weight decay, so weight decay acts only on the heads.
- CPU check: the final maps equal timm's own blocks, norm and pooling exactly.

## How the model differs from MambaVision-T

| Difference | RadioDINO-S/16 | MambaVision-T | Setting it makes suspect |
|---|---|---|---|
| Pretraining | DINO self-distillation on RadImageNet CT, MRI, ultrasound | supervised ImageNet-1k | Anchor strength, rate |
| Weight RMS | 0.049 to 0.063 over depth, stem 0.075, position embedding 0.140 | 0.078 to 0.092 | Rates and anchors |
| Parameters | 21.7M | 31.8M (paper) | Regularisers |
| Blocks | twelve equal pre-norm attention blocks over all 196 tokens, one width (384) | convolutions in stages 1 and 2, Mamba mixers and attention in 3 and 4, widths 80 to 640 | Rate profile over depth, stochastic depth |
| Normalisation | LayerNorm only | BatchNorm in stages 1 and 2, LayerNorm later | Feature-statistics augmentation |
| Position | learned absolute embedding added to every token | none (convolutions) | Feature-statistics augmentation |
| Resolution | stride 16 at every block | strides 4 to 32 | Read depths of the adapter |

## Weight scale

Weight RMS of tensors with two or more dimensions, per adapter stage, against the two recipes'
models (MambaVision-T and ResNet-50 from the
[MambaVision campaign](../mambavision-settings/prior-work.md#why-it-underfits)). Relative step
is the rate over the RMS, the movement per Adam step.

| Stage | RadioDINO | MambaVision-T | ResNet-50 (RadImageNet) | RadioDINO / MambaVision |
|---|---:|---:|---:|---:|
| Stem | 0.0752 | 0.0918 | 0.0623 | 0.82 |
| 1 | 0.0494 | 0.0843 | 0.0230 | 0.59 |
| 2 | 0.0541 | 0.0783 | 0.0149 | 0.69 |
| 3 | 0.0597 | 0.0791 | 0.0114 | 0.75 |
| 4 | 0.0626 | 0.0924 | 0.0073 | 0.68 |

Per block the RMS rises smoothly from 0.0494 (block 1) to 0.0633 (block 12).

| Relative step | Stem | Last stage |
|---|---:|---:|
| ResNet-50 at 5e-5 | 8.0e-4 | 6.8e-3 |
| MambaVision-T at 2e-4 | 2.2e-3 | 2.2e-3 |
| RadioDINO at 2e-4 | 2.7e-3 | 3.2e-3 |

The ResNet-50's steps rise 8.5 times with depth because its weights shrink; RadioDINO's do not.

## Precedents

| Mechanism | Setting | Run | Macro change | Source |
|---|---|---|---|---|
| Backbone | Swin-T on the MambaVision recipe | fold 2, 2026-10-14 | ×4 0.937176 against 0.937801 | `docs/validation/results/` and main 3639bac |
| Rate | 2e-4 flat against 5e-5 (MambaVision-T) | backbone-rate-2e-4 | ×4 +0.0024 | [MambaVision findings](../mambavision-settings/findings.md) |
| Rate | 4e-4 flat (MambaVision-T) | backbone-rate-4e-4 | ×4 +0.0004 | same |
| Rate profile | 6e-4 top, 0.6 per stage (MambaVision-T) | stage-rates-by-weight-scale | ×4 -0.0028 | same |
| Anchors | 0.001 and 0.00001 against 0.01 and 0.0001 (MambaVision-T) | anchors-by-weight-scale | ×4 +0.0101 | same |
| Anchors | per stage by weight scale (MambaVision-T) | anchors-per-stage-by-weight-scale | ×4 -0.0007, ×1 +0.0022 | same |
| Late anchor | follows the rate (MambaVision-T) | late-anchor-follows-learning-rate | ×4 +0.0014 | same |
| Schedule | twelve epochs, linear decay (MambaVision-T) | twelve-epochs, twelve-epoch-linear-decay | ×4 +0.0015, +0.0010 | same |
| Feature statistics | off (MambaVision-T) | feature-statistics-augmentation-off | ×4 -0.0011 | same |
| Feature statistics | from the second epoch (MambaVision-T) | feature-statistics-after-first-epoch | ×4 -0.0003, ×1 -0.0023 | same |
| Gradient clip | 10 against 1 (MambaVision-T) | gradient-clipping-at-10 | ×4 -0.0003 | same |
| Stochastic depth | 0.1 (MambaVision-T) | stochastic-depth-0-1-on-a-fitted-model | ×4 -0.0009, ×1 +0.0011 | same |
| Warm-up | two epochs (MambaVision-T) | two-warm-up-epochs | ×4 -0.0004, ×1 +0.0013 | same |
| Head rate | twice (MambaVision-T) | head-rate-twice | ×4 -0.0011 | same |

The MambaVision-T rows are against that campaign's incumbent at the time, not against the
reference here.

## The reference against today's source

`mambavision-t-combined-12-epochs` was trained before main's cleanups of 2026-10-14 (94868bf,
9a93087). Declared in `campaign.yaml`, by the user's choice of this run as reference:

- **Equivalences:** the run's code applied linear decay and the late anchor's schedule without
  options and had no stage decay, so `optimizer.lr_decay: linear`,
  `optimizer.late_anchor_follows_lr: true` and `optimizer.backbone_lr_stage_decay: 1.0`
  reproduce it (checked in its `source_snapshot.zip`).
- **Config exceptions:** options removed since. The anatomical frame, meniscus map, plane
  readout, soft-tissue routing and view label bias were on and are now fixed behaviour; backbone
  eval, checkpointing, dropout, series dropout, the report-attribute and scanner-confusion losses
  and unused folds were off; the label manifest path differs only by worktree.
- **Identity exceptions:** `report_attribute_manifest_sha256`, recorded while the report-attribute
  loss was at 0 and no longer written; `preprocessing_identity.sources_sha256`, the hash of the
  hashed sources, changed by main's cleanups and the adapter; and
  `preprocessing_identity.spatial_tokens.view_label_bias`, whose text lost "optional" when the
  option became fixed behaviour (the reference had it on). These were the only differences of the
  baseline replay's identity from the reference's (first start, 2026-10-15 01:34, paused at the
  seal; that runtime is kept as `outputs/autoresearch/radiodino-settings.failed-seal-20261015T0140`).
- **Not covered by any key:** the reference trained with the mask decoder's training-only
  supervision (`mask_decoder_bone_loss` and `mask_decoder_soft_tissue_loss` in its logs), which
  9a93087 removed. Seed 1's comparison with the reference therefore carries the decoder's
  removal as well as the backbone; every later seed compares RadioDINO with RadioDINO.
