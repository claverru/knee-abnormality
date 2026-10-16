# Backbone families: prior work

Measured 2026-10-15 on the pretrained weights with timm 1.0.26 on the CPU; no training yet.

## The reference

`mamba-silver-fold-2`: MambaVision-T on `configs/mambavision.yaml` (rate 2e-4, anchors 0.001 and
0.00001 with the late anchor following the rate, linear decay, twelve epochs), fold 2, no mask
decoder; ×4 0.935160, ×1 0.930081 (epoch 12), training 4,812 s on the rented machine, peak 12.3 GB.
Copied from the rented machine to `outputs/training/mamba-silver-fold-2`. Its source snapshot
differs from main in GPU-synchronisation speedups (asynchronous assertions, cached matrices,
non-blocking copies) and an unused extra-training option; declared as reference exceptions: the
machine paths, `validation.train_on_validation` (off) and `preprocessing_identity.sources_sha256`.
The other input-identity fields equal main's.

The MambaVision-T run with the mask decoder (`mambavision-t-combined-12-epochs`, ×4 0.937801) is not
the reference: main no longer has the decoder, which was worth about 0.0026 at ×4
([RadioDINO campaign](../radiodino-settings/run-notes.md)).

## The families

Shortlisted by a search on 2026-10-15 for families different from those tried, at 20 to 35M
parameters and 3 to 6 GMAC; the user kept these five. Measured at 224 px:

| Family (timm checkpoint) | Params | Stage channels at strides 4, 8, 16, 32 | Depthwise convs / BatchNorm | Weight RMS by depth | Licence |
|---|---:|---|---|---|---|
| SAM2.1 Hiera-T (`sam2_hiera_tiny.fb_r896_2pt1`) | 26.9M | 96, 192, 384, 768 | 0 / 0 | stem 0.018; blocks 0.049 falling to 0.021 | Apache-2.0 |
| HRNet-W18 (`hrnet_w18.ms_aug_in1k`) | 11.0M without head | 128, 256, 512, 1024 (incre) | 0 / 321 | stem conv 0.154, then 0.04 falling to 0.013 | MIT |
| PVTv2-b2 (`pvt_v2_b2.in1k`) | 24.9M | 64, 128, 320, 512 | 16 / 0 | 0.045 to 0.056, flat | Apache-2.0 |
| FocalNet-T SRF (`focalnet_tiny_srf.ms_in1k`) | 27.7M | 96, 192, 384, 768 | 24 / 0 | 0.043 to 0.051, flat | MIT |
| RDNet-T (`rdnet_tiny.nv_in1k`) | 22.8M | 256, 440, 744, 1040 | 21 / 0 | 0.039 to 0.043, flat | Apache-2.0 |

All five take ImageNet mean and std, as the recipe has. Hiera's stage ends are blocks 0, 2, 9 and 11;
RDNet's are dense stages 0, 1, 5 and 6. GMAC per image from the search: Hiera-T 4.90, HRNet-W18 4.28,
PVTv2-b2 4.03, FocalNet-T 4.40, RDNet-T 5.03 (MambaVision-T 4.4).

What differs from the families tried: Hiera is attention only (local mask units early, global late)
with segmentation pretraining; HRNet keeps a high-resolution stream throughout; PVTv2 attends globally
from stride 4 with spatially reduced keys; FocalNet replaces attention with gated depthwise context
aggregation; RDNet concatenates features densely instead of summing them.

## Starting settings

The MambaVision-T settings were derived from its weight scale: Adam moves a weight by about the rate
per step, so the relative step is the rate over the weight RMS, and the anchor's cost of one relative
movement scales with the RMS squared ([MambaVision prior work](../mambavision-settings/prior-work.md#why-it-underfits)).
RadioDINO-S/16 (RMS 0.05 to 0.06) confirmed them: rates from 1.4e-4 to 4e-4 and two rate profiles
landed within ±0.0006 at ×4, doubled anchors lost 0.0024 and a sevenfold weaker early anchor 0.0007
([RadioDINO findings](../radiodino-settings/findings.md)).

- **PVTv2-b2, FocalNet-T, RDNet-T** (flat, RMS 0.04 to 0.056, in RadioDINO's range): the recipe's
  rate 2e-4 and anchors 0.001 and 0.00001.
- **Hiera-T** (shrinking from 0.049 to 0.021, ResNet-like): rate 1e-4, a relative step of 2e-3 in the
  first stage to 4.9e-3 in the last, inside the ResNet-50 recipe's 8e-4 to 6.8e-3; anchors 0.01 early
  (the ResNet-50's) and 0.0001 late, whose cost per relative movement on the last stage (0.0001 ×
  0.0205² = 4.2e-8) matches the ResNet-50's (0.001 × 0.0073² = 5.3e-8) and MambaVision's (0.00001 ×
  0.092² = 8.5e-8).
- **HRNet-W18** (a BatchNorm convolution network, 0.04 falling to 0.013): the ResNet-50 recipe's rate
  5e-5; late anchor 0.001 × (0.0073 / 0.0135)² ≈ 0.0003, early anchor 0.005 for a stem conv at 0.154
  (the ResNet-50's 0.01 on a 0.062 stem, eased for the larger stem).

Stochastic depth stays 0 (the MambaVision-T and RadioDINO runs with 0.1 were not adopted).

## Varying image counts

cuDNN plans depthwise convolutions once per unseen image count: ConvNeXt-T paid 4.3 s per new count
until image counts were padded to multiples of 16 ([MambaVision backbone](../../../docs/modeling/mambavision-backbone.md#new-batch-sizes)).
PVTv2, FocalNet and RDNet have depthwise convolutions and no BatchNorm, so they pad the same way.
Hiera has neither; HRNet has BatchNorm, so it is never zero-padded (the padded rows would enter its
batch statistics).

## DINOv3 ConvNeXt-T

Added at the user's request on 2026-10-15, after the start. `convnext_tiny.dinov3_lvd1689m`: 27.8M
parameters, 96, 192, 384 and 768 channels at strides 4, 8, 16 and 32, LayerNorm only, depthwise
convolutions, ImageNet mean and std, licence `dinov3-license`. Distilled from the DINOv3 7B teacher on
LVD-1689M; second of 34 in the [backbone search](../../../docs/modeling/backbone-search.md) (0.904386),
and ×4 0.927235 at best on fold 0 on the older ReXNet-style recipe on 2026-10-03
([backbone swaps](../../../docs/modeling/backbone-swaps.md)). Weight RMS (matrices and kernels): stem 0.0096,
stages 0.0179, 0.0142, 0.0153 and 0.0159, flat but about 0.6 of the in12k ConvNeXt-T's (0.0208 stem,
0.025 to 0.028); layer-scale gammas 0.49 in the first block to 3.06 in the last (0.78 to 2.52 in12k).

Settings by the same rule: rate 5e-5 (relative step 2.8e-3 to 3.6e-3), late anchor 0.0002
(0.0002 × 0.0159² = 5.1e-8 per relative movement), early anchor 0.01 as the ResNet-50, the in12k
ConvNeXt-T config and Hiera-T. Image counts padded to 16 by the existing ConvNeXt adapter; no new code
beyond registering the name.

## Four families added on 2026-10-15

Added at the user's request after the start, from a CPU evaluation on 2026-10-15 (timm 1.0.26, 224 px,
bf16 forward cost proxies checked against the families already timed: Swin-T 5.6 min per epoch and
23.4 GB, FocalNet-T 5.5 to 5.8 min and 23 GB, HRNet-W18 7.9 min, MambaVision-T 4.3 min and 12.7 GB). RMT-S
was evaluated and left out: its weights are only on Baidu with expired links and the code has no licence.

| Family (checkpoint) | Params | GMAC | Stage channels | Weight RMS (stem, stages) | Norm / depthwise | Expected min per epoch | Licence |
|---|---:|---:|---|---|---|---|---|
| DaViT-T (`davit_tiny.msft_in1k`) | 27.6M | 4.45 | 96, 192, 384, 768 | 0.089; 0.089, 0.086, 0.086, 0.087 | LayerNorm / 24 | 5.2 to 5.6, 21 to 23 GB | Apache-2.0 |
| EdgeNeXt-B (`edgenext_base.in21k_ft_in1k`) | 17.9M | 2.93 | 80, 160, 288, 584 | 0.073; 0.085, 0.090, 0.089, 0.090 | LayerNorm / 21 | 4.0 to 4.6, about 17 GB | MIT |
| EfficientViT-B2 (`efficientvit_b2.r224_in1k`) | 15M without head | 1.53 | 48, 96, 192, 384 | 0.076; 0.057, 0.053, 0.049, 0.044 | BatchNorm (47) / 30 + 10 grouped | 4.0 to 4.8 once padded | Apache-2.0 |
| WTConvNeXt-T (official, Google Drive) | 30.4M | 4.50 | 96, 192, 384, 768 | about 0.10 flat | LayerNorm / 78 + Haar | 6.5 to 7.5 original layer; 4.6 to 5.3 with the Triton kernels | MIT |

- **DaViT-T:** spatial window attention alternating with image-wide channel attention, a 3×3 depthwise
  position convolution per block. Beats Swin-T on ADE20K (46.3 against 44.5 mIoU) and COCO (45.0/41.1
  against 43.7/39.8 AP). Geometry: 7×7 stride-4 stem, origins 2, 6 and 14 px at strides 8, 16, 32.
- **EdgeNeXt-B:** ConvNeXt-like stages with kernels 3 to 9 and split depthwise transposed attention;
  ConvNeXt geometry (origins 1.5, 3.5, 7.5, 15.5); layer-scale gammas up to 6.07.
- **EfficientViT-B2:** MobileNet blocks at strides 4 and 8, ReLU linear attention with multi-scale token
  aggregation at 16 and 32 (fp32 in timm); origin 0 at every stride; no final norm. BatchNorm with
  depthwise convolutions: padded with BatchNorm statistics from the real images only. Late anchor
  0.00003 so that its cost per relative movement on the last stage (5.8e-8) matches the ResNet-50's and
  MambaVision-T's.
- **WTConvNeXt-T:** ConvNeXt-T with wavelet convolutions (5×5 base plus 5, 4, 3, 2 Haar levels per stage).
  Paper against ConvNeXt-T: ADE20K 46.6 against 46.0, COCO 51.0/44.4 against 50.4/43.7, shape bias 29.2%
  against 21.7%. Official checkpoint kept at `~/.cache/torch/hub/checkpoints/wtconvnext_tiny_official.pth`
  (sha256 1df1052d…091f), the MIT repository cloned at `~/.cache/wtconv-src/WTConv`; its fused Triton
  backend (arXiv 2608.10805) runs the layer about 3× faster than the original.

**WTConvNeXt-T GPU parity, keeper measurement 2026-10-16 03:15 CEST** (`evidence/wtconv_gpu_parity.py`,
the candidate `wtconvnext-tiny-20261016T005805Z-b03b5f00`'s `modeling/wtconv.py` against the official
`wtconv/wtconv2d.py` with the same random weights, 16 images, float32, TF32 off in cuDNN and matmul,
forward and backward on the 3090). Relative L2 errors, vendored against official:

| Stage shape | Levels | Output | Input gradient | Largest parameter gradient (every base and wavelet weight, bias and scale) |
|---|---:|---:|---:|---:|
| 96 ch, 56 px | 5 | 1.4e-7 | 9.1e-8 | 6.4e-7 |
| 192 ch, 28 px | 4 | 1.4e-7 | 8.8e-8 | 5.3e-7 |
| 384 ch, 14 px | 3 | 1.3e-7 | 8.5e-8 | 4.2e-7 |
| 768 ch, 7 px (odd) | 2 | 1.2e-7 | 8.0e-8 | 3.0e-7 |

All below the seed's 1e-5 tolerance, including the five-level cascade and the odd 7 px map. With TF32
left on (PyTorch's default for cuDNN convolutions) the official layer's own rounding gives parameter
gradient differences up to 3e-4; under bf16 autocast both differ from float32 by 3.3e-3 alike.

All four take ImageNet mean and std and stochastic depth 0; DaViT-T, EdgeNeXt-B and WTConvNeXt-T sit at
MambaVision-T's weight scale, so the recipe's rate and anchors apply.

## Epochs that fit the time bound, per family

Every experiment training first runs the preflight (`scripts/tools/preflight_training.py`): 60 real
training batches and 20 validation batches; it projects 1.13 × epochs × (442 median steps + validation)
plus the first-epoch excess it measures, and stops the candidate before training when that exceeds
4,090 s or when the predicted peak memory exceeds 22.5 GiB. Real training ran 1.04 to 1.09 times the
unscaled projection for DaViT-T, EdgeNeXt-B, EfficientViT-B2, FocalNet-T, PVTv2-b2 and WTConvNeXt-T,
and 1.13 for RDNet-T. Measured per-epoch cost (median step × 442 + validation, unscaled) and the
largest epoch count the preflight passes:

| Family (as run here) | Seconds per epoch, unscaled | Epochs the preflight passes |
|---|---:|---:|
| EdgeNeXt-B | 270 | up to 13 |
| FocalNet-T SRF (1×1 convs as linear) | 312 to 316 | up to 11; twelve is rejected (4,309 s projected; the incumbent's own twelve-epoch run took 4,067 s, 23 s inside the bound) |
| DaViT-T | 323 | up to 11 |
| EfficientViT-B2 (stages 0–1 checkpointed) | 322 | up to 11 |
| PVTv2-b2 (stage 0 checkpointed) | 342 | up to 10 |
| WTConvNeXt-T (Triton WTConv) | 356 | up to 9 |

A FocalNet-T idea therefore runs at ten or eleven epochs: the ten-epoch FocalNet-T run is level with the
twelve-epoch incumbent (×4 0.938154 against 0.938816, 80% interval −0.0022 to +0.0009).

## Precedents

| Backbone | Run | ×4 | Source |
|---|---|---:|---|
| ResNet-50 | targets-and-inputs run 27 | 0.939634 | targets-and-inputs campaign |
| MambaVision-T with the mask decoder | mambavision-t-combined-12-epochs | 0.937801 | MambaVision settings campaign |
| Swin-T on the MambaVision recipe | 2026-10-14 | 0.937176 | docs/validation/results/2026-10-14-swin-t.md |
| MambaVision-T without the mask decoder | mamba-silver-fold-2 | 0.935160 | this campaign's reference |
| RadioDINO-S/16, best | radiodino-settings, top-down warm-up | 0.929641 | radiodino-settings campaign |
