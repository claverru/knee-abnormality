# Backbone families 2: prior work

Measured 2026-10-17 on the CPU (torch 2.10, timm 1.0.26) with the official weights; no training yet.
The campaign runs on main's Kaggle-aligned environment (a89bb1c: Python 3.13.15, torch 2.11, timm
1.0.29); FocalNet-T's backbone outputs are bit-identical between the two on the CPU in float32.
Tags: measured here, from source (linked), or a guess.

## The reference and the blend partner

**Reference** `outputs/training/focalnet-silver-fold-2`: FocalNet-T SRF, the incumbent of the
[first round](../backbone-families/INDEX.md) (`focalnet-tiny-srf-20261015T172800Z-0542173d`), on the
MambaVision-T recipe settings (rate 2e-4, anchors 0.001 and 0.00001 with the late anchor following the
rate, linear decay), twelve epochs, fold 2, seed 1110484336. ×4 0.938816, ×1 0.933241 (epoch 10),
training 4,067 s locally, peak 21.5 GiB (23.0 GB). Its recipe is `configs/focalnet.yaml` at twelve epochs
and patience twelve (`configs/campaign.yaml`, uncommitted; main's config trains ten).

Its source snapshot is the first round's worktree before the EfficientViT-B2 merge. Against main
(485e779) the training code differs only by EfficientViT-B2 additions (`efficientvit` branches in
`model.py`, `backbone_stages(backbone, stochastic_depth)`, the EfficientViT mapping) and harness files;
no FocalNet-T path changed (measured: `diff` of the extracted snapshot and main). Declared as reference
exceptions: `paths.label_manifest` (the old worktree's path) and `preprocessing_identity.sources_sha256`.

**Blend partner** `outputs/training/resnet50-silver-fold2`: the ResNet-50 on its silver recipe, same
fold and seed, ×4 0.937820, the control of the crop-views campaign. Equal logit blends with it on fold 2,
×4 (measured):

| Run | Alone | Blend with the ResNet-50 | Blend gain |
|---|---:|---:|---:|
| FocalNet-T `focalnet-silver-fold-2` (reference) | 0.938816 | 0.940999 | +0.003179 |
| EfficientViT-B2 `efficientvit-sd02-fold-2` | 0.938773 | 0.940918 | +0.003098 |
| MambaVision-T `mamba-silver-fold-2` | 0.935160 | 0.939660 | +0.001840 |

The blend is a reading; adoption follows the shared rule ([contract](contract.md#adoption)).

## The families

The user's list of 2026-10-17, MobileCLIP2-S2 dropped by the user (35.8M parameters, 6.0 GMAC, apple-amlr
weights). Measured at 224 px:

| Family (checkpoint) | Params without head | GMAC | Channels at strides 8, 16, 32 | Norm / depthwise convs | Weight RMS (early anchor; stages) | Custom op | Licence (code; weights) |
|---|---:|---:|---|---|---|---|---|
| OverLoCK-XT (`overlock_xt_in1k_224.pth`) | 15.5M | 2.84 | 112, 340, 420 | BatchNorm 187, LayerNorm 25 / 161 | 0.078; 0.065 to 0.077 flat | natten (replaced, pure PyTorch) | Apache-2.0; ImageNet-1K terms |
| TransNeXt-Micro (`transnext_micro_224_1k.pth`) | 12.4M | 2.65 | 96, 192, 384 | LayerNorm 69 / 21 | 0.057; 0.049 to 0.055 | swattention CUDA, optional | Apache-2.0; Apache-2.0 |
| VSSD-Tiny MESA (`vssd_tiny_e300_with_mesa.pth`) | 23.8M | 4.66 | 128, 256, 512 | BatchNorm 8, LayerNorm 51 / 53 | 0.075; 0.062 to 0.073 flat | none in this config | none in the repo; MIT on the weights |
| FlashInternImage-T (`flash_intern_image_t_1k_224.pth`) | 28.7M | 4.85 | 128, 256, 512 | LayerNorm 69 / 0 | 0.093; 0.085 to 0.107 | DCNv4 CUDA, needed | MIT (Apache-2.0 op); not stated |
| MedSAM2 Hiera-T (`MedSAM2_latest.pt`) | 26.8M | 4.9 | 192, 384, 768 | LayerNorm / 0 | stem 0.013, block 0 0.036; 0.024, 0.016, 0.013 falling | none | Apache-2.0; CC-BY-SA-4.0 plus research and education only |
| LingBot-Vision ViT-S/16 (`robbyant/lingbot-vision-vit-small`) | 21.6M | 4.33 | 384 at 16 (plain ViT) | LayerNorm / 0 | patch embed 0.023; blocks 0.052 to 0.057 | none | Apache-2.0; Apache-2.0 |

**Expected training time per epoch against FocalNet-T SRF** (339 s per epoch locally, 4,067 s for twelve;
the bound allows about 409 s per epoch at ten epochs, 1.21 times). Measured on this 3090: ConvNeXt-T 0.83
(4.7 min), SAM2.1 Hiera-T 0.84 (4 min 46 s), RadioDINO-S/16 0.66 (2,701 s for twelve). The rest are guesses
from the CPU measurements and published throughput above, converted through ConvNeXt-T's 0.83:

| Family | Time per epoch vs FocalNet-T | Ten epochs | Basis | Licence (code; weights) |
|---|---:|---:|---|---|
| OverLoCK-XT | 0.85 to 1.25 | 2,900 to 4,200 s | guess: 1.0 to 1.5 × ConvNeXt-T | Apache-2.0; ImageNet-1K terms |
| TransNeXt-Micro (pure PyTorch) | 0.8 to 1.2 | 2,700 to 4,100 s | guess | Apache-2.0; Apache-2.0 |
| VSSD-Tiny, stem and stage 0 checkpointed | 1.0 to 1.45 | 3,400 to 4,900 s | guess: 1.1 to 1.6 × ConvNeXt-T plus the recompute | none in the repo; MIT |
| FlashInternImage-T, DCNv4 in float32 | 1.15 to 1.65 | 3,900 to 5,600 s | guess: 1.4 to 2 × ConvNeXt-T | MIT (op Apache-2.0); not stated |
| MedSAM2 Hiera-T | 0.84 | about 2,900 s | measured on the SAM2.1 weights, same architecture | Apache-2.0; CC-BY-SA-4.0, research and education only |
| LingBot ViT-S/16 | about 0.7 | about 2,400 s | RadioDINO-S/16 measured, same size plus four registers | Apache-2.0; Apache-2.0 |

VSSD-Tiny and FlashInternImage-T may need nine epochs or more checkpointing to fit; the preflight decides.

All take ImageNet mean and std. Licences: the host's rule is that non-commercial terms alone do not
disqualify a source ([supervision](../../../docs/modeling/supervision.md#official-labeling-context)), and
MedSAM2 passed the user's licence and rules pass of 2026-10-07
([segmenters](../../../docs/research/geometry-and-views/anatomy-from-pretrained-segmenters/INDEX.md)).

Weights and sources on this machine:

| Family | Weights (sha256 prefix) | Source |
|---|---|---|
| OverLoCK-XT | `~/.cache/torch/hub/checkpoints/overlock_xt_in1k_224.pth` (b8e1166c5ca9), the training form, not the reparameterised one | `~/.cache/overlock-src`, [LMMMEng/OverLoCK](https://github.com/LMMMEng/OverLoCK) fb6f22b, `models/overlock.py`, `models/contmix.py` |
| TransNeXt-Micro | `~/.cache/torch/hub/checkpoints/transnext_micro_224_1k.pth` (18dbc88e9c1e) | `~/.cache/transnext-src`, [DaiShiResearch/TransNeXt](https://github.com/DaiShiResearch/TransNeXt) c8a9974, `classification/transnext_native.py` path |
| VSSD-Tiny | `~/.cache/torch/hub/checkpoints/vssd_tiny_e300_with_mesa_model.pth` (2c530c024d1d), the `model` entry (not `model_ema`) of the HF file (489cddbf7f42) re-saved as a plain state dict | `~/.cache/vssd-src`, [YuHengsss/VSSD](https://github.com/YuHengsss/VSSD) c10de2f8, `classification/models/mamba2.py` (`VMAMBA2`), `configs/vssd/vssd_tiny_e300.yaml` |
| FlashInternImage-T | `~/.cache/torch/hub/checkpoints/flash_intern_image_t_1k_224_model.pth` (b664d5a308f6), the `model` entry re-saved as a plain state dict | `~/.cache/flashinternimage-src/DCNv4`, [OpenGVLab/DCNv4](https://github.com/OpenGVLab/DCNv4) 4b848f7 |
| MedSAM2 Hiera-T | HF `wanglab/MedSAM2` snapshot e4a6f35, `MedSAM2_latest.pt` (c92743b99f00) | `~/.cache/medsam2-src/repo` 332f30d; timm's `sam2_hiera_tiny` |
| LingBot ViT-S/16 | HF `robbyant/lingbot-vision-vit-small` revision 127cbcec380d, `model.pt` (dca36562cb6b) | `~/.cache/lingbot-src/lingbot-vision` 151e463; timm's `vit_small_patch16_dinov3_qkvb` |

### OverLoCK-XT

A convolutional network in three parts: a Base net to stride 16, an Overview net that runs one more stage
to stride 32 and projects 84 context channels back to stride 16, and a Focus net whose ContMix blocks
build a per-pixel kernel from the query against the context pooled to 7×7 (softmax with a relative bias,
over 13×13 or 7×7 and 5×5 windows) and run the stride-16 and stride-32 stages again with that context.

- **Stages (measured):** stem stride 4 (56 ch) → `blocks1` 56 ch at 4 → `blocks2` **112 ch at 8** →
  `blocks3` 256 ch at 16 (Base) → Overview `blocks4` 336 ch at 32 (aux head only) → Focus `sub_blocks3`
  **256 + 84 ch at 16** → `sub_blocks4` **420 ch at 32**, the map the classifier reads. No final norm.
- **Geometry:** every downsampler is a 3×3 stride-2 convolution with padding 1, so the origin is 0 px at
  every stride, the convolution-centres convention of EfficientViT-B2 and MambaVision-T (measured shapes,
  source formula).
- **natten:** the official file needs `natten.functional.na2d_av`, removed in NATTEN 0.20 (torch ≥ 2.7
  only), so no natten build works with torch 2.10 or 2.11 (from source). Two pure-PyTorch replacements were
  checked against a direct gather with natten's window rule (measured): the repository's unfold fallback
  (exact, 368 MiB saved activations per image) and a dense scatter of the window weights into an HW×HW
  matrix followed by a matmul (≤3e-7, 266 MiB; HW is 196 or 49). mmengine is used only to load the file;
  `torch.load` with `load_state_dict` loads it strictly (1,564 keys).
- **Normalisation:** 187 BatchNorm in the training form (76 after reparameterisation), so batch statistics
  must come from the real images only; 161 depthwise convolutions, so image counts are padded to 16 as for
  EfficientViT-B2. GRN and SE act per image. LayerScale multiplies the identity branch (`res_scale`).
- **Cost:** activations 266 MiB per image against FocalNet-T SRF's 178 and ConvNeXt-T's 157 (measured,
  same counter); CPU forward and backward 0.61 s for two images against FocalNet-T's 0.25. Published
  inference 1,672 img/s against FocalNet-T's 1,251 (L40S, with natten). Guess: 1.0 to 1.5 times
  ConvNeXt-T's step and 1.5 times FocalNet-T's activation memory, so stage checkpointing is likely needed.
  Pretraining used drop path 0.1; the constructor default is 0.

### TransNeXt-Micro

Aggregated attention in stages 1 to 3: one softmax over a 3×3 sliding window and 7×7 pooled tokens, with
cosine queries and keys, a softplus temperature, a log-length scale, a query embedding, learned local
tokens and a small MLP that produces the pooled tokens' position bias; full attention over the 49 tokens in
stage 4; a convolutional GLU (3×3 depthwise) in every block.

- **Stages (measured, pure-PyTorch path):** 48, **96, 192, 384** channels at 4, **8, 16, 32**; depths 2, 2,
  15, 2. Every stage ends with its own LayerNorm; stage 4 stays as 49 tokens and needs a reshape.
- **Geometry:** 7×7 stride-4 stem with padding 3, 3×3 stride-2 downsamplers with padding 1, all
  overlapping: origin 0 px at every stride (measured by impulse centroids).
- **Fixed size:** built with `img_size=224, pretrain_size=224`; 192 and 256 px inputs fail on such a model
  (measured). No size-dependent tensor is stored in the checkpoint; the position tables are rebuilt at
  construction (on CUDA whenever CUDA is visible).
- **The CUDA extension is optional:** the pure-PyTorch module takes the same weights (identical buffers,
  measured; the author calls the two equivalent). The extension trains 1.84 times faster and uses 17% less
  memory (A100, from source) but needs a build per torch version and has no bfloat16. The pure path works
  under float16 and bfloat16 autocast (logits within 0.0014 and 0.016, measured) and needs nothing on
  Kaggle. `transnext.py` imports `pkg_resources`, absent here; the native module is imported directly.
- **Special parameters:** temperature 2.7 to 3.4, query embedding 0.15 to 0.24, local position bias 0.53 to
  0.89, position MLPs 0.19 to 0.40 RMS (measured), against matrices at 0.05; the repository keeps them out
  of weight decay.
- **Cost (guess):** 0.8 to 1.2 times FocalNet-T's step on the pure path (2.65 GMAC, but a 3,136-token first
  stage with unfold); memory 1.0 to 1.3 times ConvNeXt-T's. CPU forward and backward 0.66 s for two images.
  Pretraining drop path 0.15; constructor default 0.

### VSSD-Tiny (MESA)

Non-causal state-space duality: the causal mask of Mamba-2's SSD is dropped, so each block is
y = C·(Bᵀ(V·dA)) + D·V, two matmuls over all tokens; standard global attention in the last stage.

- **Checkpoint:** the README's MESA row (84.1%, 24M, 4.5G) loads strictly into `VMAMBA2` with
  `configs/vssd/vssd_tiny_e300.yaml` (embed 64, depths 2-4-8-4, `LINEAR_ATTN_DUALITY=True`), not the ICCV
  model (28M, RoPE). Both the `model` and `model_ema` entries load; `model` is the 84.06% one (measured).
- **Stages (measured):** 64, **128, 256, 512** channels at 4, **8, 16, 32**; each code stage ends with its
  downsampler, so a stage's map is read before `layers[i].downsample`. The final LayerNorm (512) acts on
  the stride-32 tokens.
- **Geometry:** a five-convolution BatchNorm stem (3×3 stride 2, padding 1, twice) and downsamplers of a
  1×1 expansion, a 3×3 stride-2 depthwise convolution with padding 1 and a 1×1 projection: origin 0 px at
  every stride (measured).
- **No compiled kernel:** with `LINEAR_ATTN_DUALITY=True` the mixer is elementwise operations and two
  matmuls; mamba_ssm and Triton are not called (measured: the model ran with mamba_ssm's import failing).
  Peaks at 224 px max|KV| 69 and max|y| 130, far inside float16 (measured).
- **Normalisation:** 8 BatchNorm (stem and the three downsamplers) and 53 depthwise convolutions: real-image
  BatchNorm statistics and padding to 16, both, as EfficientViT-B2.
- **Cost:** activations 275 MiB per image against FocalNet-T SRF's 174 (measured, 1.58 times);
  checkpointing the stem and stage 0 saves about 31%. Guess: 1.1 to 1.6 times ConvNeXt-T's step. The
  mixer's state is an unnormalised sum over all tokens, so its scale follows the token count (224 px is the
  pretraining size). Pretraining drop path 0.2.
- **Licence:** the GitHub repository has no licence file; the HF weights are tagged MIT.

### FlashInternImage-T (DCNv4)

Deformable convolution v4: each 3×3 aggregation samples nine learned offsets per group bilinearly, with
unnormalised (no softmax) modulation, on a value projection; pre-norm blocks with an MLP.

- **Stages (measured):** 64, **128, 256, 512** channels at 4, **8, 16, 32**; depths 4, 4, 18, 4; groups 4,
  8, 16, 32 of 16 channels; each level ends with a LayerNorm. Tokens are (N, HW, C) inside levels.
- **Geometry:** stem of two 3×3 stride-2 convolutions with padding 1 and LayerNorm; 3×3 stride-2
  downsamplers with padding 1: origin 0 px at every stride.
- **Normalisation:** LayerNorm only (the stem's and downsamplers' norm layers default to LN; no running
  statistics in the checkpoint), no depthwise convolution: no padding constraint.
- **The DCNv4 op is CUDA only:** no CPU kernel and no prebuilt wheel (PyPI holds source distributions only;
  no releases). `import DCNv4` fails without a GPU (it reads the device's compute capability at import;
  7.5 and 8.6 are listed). It needs nvcc matching torch's CUDA 12.x, `TORCH_CUDA_ARCH_LIST` 8.6 here and 7.5
  for Kaggle's T4s, and one build per torch version. Since main a89bb1c the local environment is
  Kaggle's (Python 3.13, torch 2.11), so one wheel built for compute 7.5 and 8.6 serves both: the
  private notebook `claverru/knee-kernel-wheels-build` builds it in Kaggle's image. nvcc is not
  installed here. A search on 2026-10-17 found no usable build: PyPI holds only source distributions; the TestPyPI
  wheels of the [inspiros/dcnv4](https://github.com/inspiros/dcnv4) port (MIT-0) are built for torch 2.2.1,
  CUDA 12.1 and CPython 3.8 to 3.11 and do not load here (torch 2.10, CPython 3.12); no Kaggle, Hugging
  Face, Astral or Triton build exists.
- **Our build (2026-10-17):** compiled from 4b848f7 in Kaggle's image (T4, torch 2.11, 546 s) with
  `TORCH_CUDA_ARCH_LIST="7.5;8.6"` (`dcnv4-1.0.0.post2`, sha256 6b0b298a88e2), then its Python wrapper patched
  for the batch bug below and repacked as `dcnv4-1.0.0.post2+knee1-cp313-cp313-linux_x86_64.whl` (sha256
  585e56e54e7e; same compiled extension; `findspec` and `find_spec_bwd` take the multiplier from the factors of
  H·W and no longer read or fill the TABLE caches). Stored in `outputs/submission/dependencies/kaggle-py313-dcnv4`
  beside Kaggle's full `pip freeze`, installed in this worktree's environment as a path source (uncommitted
  `pyproject.toml`).
- **Upstream batch bug (measured on the 3090):** a CUDA block handles `block_multiplier` consecutive output
  positions and takes its image index as `blockIdx.x · block_multiplier / Q` with Q = H·W, so the multiplier
  must divide H·W; upstream's `findspec` and `find_spec_bwd` choose it from the factors of B·H·W. With 4 images
  of 14×14 and 16 groups it picks 16 (196 is not a multiple), and images 2 to 4 come out wrong (max error
  9.6 to 11 against torchvision's `deform_conv2d`, image 1 within 1e-5); at 208 images of 56×56 the
  offset gradient is wrong. With the patched wheel, `DCNv4Function` against the pure-PyTorch DCNv3 core in
  float64 (4 images at 14 px with 16 groups and at 28 px with 8, 208 at 7 px with 32, 64 and 8 at 56 px with 4):
  output within 1e-4 on every image, value gradient within 9e-5, offset/mask gradient within 2.2e-4 of a scale
  near 80 except 22 of 21.7M and 5 of 2.7M elements at 56 px, where float32 coordinates (about 4e-6 of
  rounding at 56) fall on the other side of a pixel boundary from float64, the kink of bilinear sampling.
- **Exact CPU references (measured):** the repository's `dcnv3_core_pytorch` fed DCNv4's layout (per group
  18 offsets then 9 masks, the last dimension padded to a multiple of 8) matches a transcription of the CUDA
  kernel to 1.4e-6; `torchvision.ops.deform_conv2d` with an all-ones depthwise weight matches to 1.5e-5 per
  op and 9.5e-7 on the logits, and has CUDA kernels with autograd (speed on the GPU unmeasured).
- **Precision:** the kernels dispatch float, half and bfloat16, but the bfloat16 backward fails ("expected
  scalar type Float but found BFloat16", issues #8, #15, #24); the recipe trains under bf16-mixed, so the op
  runs in float32 with autocast off around it. The backward accumulates the value gradient with atomicAdd
  (nondeterministic). Batches must hold at most `im2col_step` (256) images or a multiple of it.
- **Cost (guess):** 1.4 to 2 times ConvNeXt-T's step with the CUDA op (published inference 2,316 img/s
  float32 against ConvNeXt-T's 2,485 on an A100); several times slower through torchvision. Pretraining
  drop path 0.1.

### MedSAM2 Hiera-T

SAM2.1 Hiera-T fine-tuned in full on 3D medical images and videos (363k CT, 77k MRI, 15k PET image-mask
pairs, ultrasound and endoscopy frames; no knee MRI named) at 512 px with ImageNet normalisation.

- **Load:** timm's `hieradet_sam2.checkpoint_filter_fn(state, prefix="image_encoder.trunk.")` maps the 154
  trunk tensors into `sam2_hiera_tiny` with no unexpected key or shape mismatch; only `head.norm` is
  missing (ones and zeros in timm's own checkpoint, unused by the stage outputs). Position embeddings equal
  timm's (7×7 background, 8×8 window). Stage ends at blocks 0, 2, 9 and 11, as SAM2.1's (measured).
- **Drift from SAM2.1:** relative change 0.28 (stem) rising to 0.71 (stage 4); from the stem to stage 2
  mostly a uniform shrink to 0.68 to 0.74 of SAM2.1's RMS, stages 3 and 4 also rotated (cosine 0.91 and
  0.71) (measured).
- **Maps on knee slices** (two real training slices, ImageNet-normalised, measured): RMS at strides 4, 8,
  16, 32 of 0.300, 0.209, 0.198, 0.109, against SAM2.1's 0.501, 0.306, 0.364, 0.136 and the ResNet-50's
  1.7 and 0.19 at 16 and 32. No final norm.
- **Training resolution** 512 px, 2.3 times ours per side (SAM2.1: 1024, 4.6 times).

### LingBot-Vision ViT-S/16

Robbyant (Ant Group), arXiv 2607.05247, 2026-07: the DINOv2/v3 recipe plus masked boundary modelling (the
teacher predicts a sub-pixel line-segment boundary field; boundary patches are forced into the iBOT mask and
supervised with a 32-bin field loss); 161M curated web images; the ViT-S/16 distilled from the ViT-g at 256
and then 512 px. Ahead of DINOv3-S on ImageNet linear (82.22 against 80.37), NYU depth and ADE20K, behind on
KITTI and Cityscapes (from source).

- **Load:** strict into timm 1.0.26's `vit_small_patch16_dinov3_qkvb` through `timm.models.eva.checkpoint_filter_fn`
  (registers, gammas and the Q/V biases renamed, the K bias dropped); outputs identical to the official
  model (max difference 0.0, measured). `vit_small_patch16_dinov3` drops the QKV biases silently and
  diverges (11.8).
- **Architecture:** 12 blocks of width 384, CLS plus 4 register tokens, axial 2D RoPE on the patch tokens
  only (coordinates normalised to [−1, 1], so 224 px needs no interpolation), layer scale, one final norm.
- **Scale:** patch-token RMS per block 0.58, 0.36, 0.31, 0.27, 0.26, 0.37, 0.46, 0.75, 0.97, 1.42, 2.55,
  5.99 before the final norm (0.739 after); registers up to 420 from block 5 (measured, synthetic input).
  Layer-scale gammas grow from 0.15 mean absolute in blocks 0 to 2 to 3.0 in blocks 9 to 11.
- **Precedent:** the RadioDINO-S/16 adapter
  ([patch](../radiodino-settings/radiodino-adapter.patch)): CLS and its position embedding dropped, stages of
  1, 2, 5 and 4 blocks, the middle read after block 8, the final read through the ViT norm and a 2×2
  average pool (origins 7.5 and 15.5, the ConvNeXt-T anchors).

## Starting settings

By the weight-scale rule of the first round ([prior work](../backbone-families/prior-work.md#starting-settings)):
Adam moves a weight by about the rate per step, so the relative step is the rate over the weight RMS, and
the anchor's cost of one relative movement scales with the RMS squared. The reference's late anchor costs
1e-5 × 0.047² = 2.2e-8 per relative movement, MambaVision-T's 8.5e-8, the ResNet-50's 5.3e-8.

- **OverLoCK-XT, TransNeXt-Micro, VSSD-Tiny, FlashInternImage-T** (flat, RMS 0.05 to 0.107, the range of
  FocalNet-T, DaViT-T and MambaVision-T on these settings): the recipe's rate 2e-4 and anchors 0.001 and
  0.00001 (late cost 2.6e-8 to 1.1e-7).
- **MedSAM2 Hiera-T** (falling from 0.036 to 0.013, ResNet-like, as SAM2.1 Hiera-T was): rate 1e-4, a
  relative step of 2.8e-3 in block 0 to 7.6e-3 in the last stage (the ResNet-50 recipe's 8e-4 to 6.8e-3);
  anchors 0.01 early, as the ResNet-50 and the SAM2.1 run, and 0.0003 late, whose cost on the last stage
  (0.0003 × 0.0132² = 5.2e-8) matches the ResNet-50's.
- **LingBot ViT-S/16** (blocks 0.052 to 0.057, RadioDINO's band, where rates from 1.4e-4 to 4e-4 landed
  within ±0.0006): the recipe's rate 2e-4 and anchors 0.001 and 0.00001.

Stochastic depth stays 0 for every family, as in the first round; ten epochs.

## Varying image counts

cuDNN plans depthwise convolutions once per unseen image count, so families with depthwise convolutions pad
image counts to multiples of 16; a family with BatchNorm pads only with BatchNorm statistics taken from the
real images ([first round](../backbone-families/prior-work.md#varying-image-counts)). OverLoCK-XT and
VSSD-Tiny have both and pad as EfficientViT-B2 does; TransNeXt-Micro pads as ConvNeXt does;
FlashInternImage-T, MedSAM2 Hiera-T and LingBot have no depthwise convolution and no BatchNorm.

## Kaggle inference

T4s, torch 2.11, float16 or float32, offline. OverLoCK-XT (dense ContMix), TransNeXt-Micro (native path),
VSSD-Tiny, MedSAM2 and LingBot are pure PyTorch on timm 1.0.26. FlashInternImage-T needs a DCNv4 wheel
built for torch 2.11 and sm_75 in a Kaggle GPU session and shipped as a dataset, or the torchvision
replacement.

## Time and memory bound

The bound is 1,684 to 4,881 s of training (0.414 to 1.20 of the reference's 4,067 s, the user's 120% of
2026-10-18; it started at 4,090 s, 1.006), the first round's
absolute window. Every candidate first runs the preflight (`scripts/tools/preflight_training.py`: 60 real
training batches and 20 validation batches; it stops a candidate projected over the bound or 22.5 GiB). The
first round's measured costs per epoch, unscaled: EdgeNeXt-B 270 s, FocalNet-T SRF 312 to 316 s, DaViT-T
323 s, EfficientViT-B2 322 s, PVTv2-b2 342 s, WTConvNeXt-T 356 s (nine epochs); ten epochs pass up to about
350 s per epoch ([first round](../backbone-families/prior-work.md#epochs-that-fit-the-time-bound-per-family)).
Activation checkpointing leaves outputs and gradients unchanged; a checkpointed BatchNorm must not update its
running statistics twice (EfficientViT-B2's recompute).

## Precedents

Fold 2 of `all_5fold_scanner_v2`, ×4 macro AUC.

| Backbone | Run | ×4 | Blend gain with the ResNet-50 | Source |
|---|---|---:|---:|---|
| FocalNet-T SRF, twelve epochs | `focalnet-silver-fold-2` | 0.938816 | +0.003179 | this campaign's reference |
| EfficientViT-B2, stochastic depth 0.2 | `efficientvit-sd02-fold-2` | 0.938773 | +0.003098 | [first round](../../../docs/validation/results/backbones/2026-10-16-backbone-families.md) |
| ResNet-50, silver recipe | `resnet50-silver-fold2` | 0.937820 | (the blend partner) | crop-views campaign |
| DaViT-T | first round | 0.936669 | | first round |
| EdgeNeXt-B | first round | 0.936033 | | first round |
| WTConvNeXt-T, nine epochs | first round | 0.935647 | | first round |
| MambaVision-T without the mask decoder | `mamba-silver-fold-2` | 0.935160 | +0.001840 | first round's reference |
| DINOv3 ConvNeXt-T | first round | 0.934199 | | first round |
| PVTv2-b2 | first round | 0.933099 | | first round |
| SAM2.1 Hiera-T, rate 1e-4, anchors 0.01 and 1e-4 | first round, aborted at epoch 7 | ×1 −0.021 against MambaVision-T at epoch 7 | | [run notes](../backbone-families/run-notes.md#1-sam2-hiera-tiny-aborted-at-epoch-7-on-the-users-order-invalid) |
| RadioDINO-S/16, best | radiodino-settings | 0.929641 | | [result](../../../docs/validation/results/backbones/2026-10-15-radiodino-s16.md) |

The SAM2.1 Hiera-T run localised bones level with or better than the reference (kneecap overlap 0.709
against 0.652) and lagged on the menisci and cartilage maps; its candidate causes were no convolution after
the patch embedding, the 1024 px pretraining, the weak stride-16 map and the low rate with a strong early
anchor. RadioDINO's gap sat 85% on located structures (Lateral Meniscus and PF OA −0.015 each).
