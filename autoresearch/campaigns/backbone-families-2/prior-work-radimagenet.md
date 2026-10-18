# Backbone families 2: original RadImageNet CNNs

Added 2026-10-17 at the user's request: the three original RadImageNet CNNs never run here, as forced
seeds of this campaign. Measured by the keeper on the CPU and the 3090; no training yet.

## Weights

Source: Kaggle dataset `ipythonx/notop-wg-radimagenet` (the official Keras "notop" files, Keras
2.4.0), in `~/.cache/radimagenet-keras/` (sha256 prefixes: DenseNet121 5266539752a3, InceptionV3
b4fd87165c56, IRV2 68ed497bbe7f). Converted by the keeper to timm state dicts, read with h5py only
(no unpickling, no Keras model loading):

| Backbone | timm architecture | File in `~/.cache/torch/hub/checkpoints/` | sha256 |
|---|---|---|---|
| DenseNet-121 | `densenet121` | `radimagenet_densenet121_keras.pt` | 58f410cd84dc0963789f3a715d3526358eb0792dfbb87b851a264ef4cdc06ef6 |
| InceptionV3 | `inception_v3` | `radimagenet_inception_v3_keras.pt` | af10f5171cb38e5a519c10745a71fceca6a267d8a011be3bc32874e0f9a98c7a |
| InceptionResNetV2 | `inception_resnet_v2` | `radimagenet_inception_resnet_v2_keras.pt` | 3322cd7f261c7680bf44b887cbde5e2e65277b4a4d55ff55ed31bb25736ea178 |

Each file is a plain state dict that loads with `load_state_dict(strict=False)` into
`timm.create_model(arch, pretrained=False, num_classes=0)` with no missing key except
`num_batches_tracked`. Kernels were transposed from HWIO; Keras BatchNorm without scale gets weight 1.

**Exactness (measured):** on eight real training slices, timm `forward_features` equals the Keras
model loaded from the same file (TensorFlow 2.17, CPU): DenseNet-121 max difference 5e-6 (after the
final ReLU Keras applies), InceptionResNetV2 5e-6, InceptionV3 3e-6 **only with Keras pooling**: the
three branch-pool `F.avg_pool2d(x, 3, 1, 1)` calls in timm's InceptionA, InceptionC and InceptionE
must use `count_include_pad=False` (Keras `same` average pooling excludes padding); with timm's
default the output correlates 0.976. The Keras BatchNorm epsilon is 0.001 for both Inceptions, as
timm's; 1.001e-5 for DenseNet (timm 1e-5).

## Input

The checkpoints' own BatchNorm statistics match our slices in **[0, 1]** (quantile-normalised, the
recipe's intensity range), not [−1, 1] or ImageNet normalisation: stem variance ratio 1.16 to 1.19
against 4.7 for [−1, 1] and 23 for ImageNet; DenseNet-121 matches every layer (median ratio 1.00,
all 121 within 3×). So `data.mean [0, 0, 0]` and `data.std [1, 1, 1]`, with the repeated middle
slice (RadImageNet images are grayscale copied across RGB). The deeper Inception layers see lower
variance than their running statistics (median ratio 0.1 in the last fifth), as the original
ResNet-50's last layer did (0.10); the Keras reference shows the same, so it is the data, not the
conversion.

## Geometry

- **DenseNet-121:** 7×7 stride-2 stem (padding 3), 3×3 stride-2 max pool (padding 1), dense blocks at
  56, 28, 14, 7 joined by 2×2 average-pool transitions: the stride-8 point is `denseblock2`'s output
  (28×28, after `transition1`), the middle map `denseblock3`'s (14×14), the final map `denseblock4`
  through `norm5` and a ReLU (7×7, as Keras applies). The 2×2 average pools shift origins by half
  the incoming stride.
- **InceptionV3 and InceptionResNetV2:** valid (unpadded) stride-2 convolutions and max pools, so at
  224 px the maps are 111, 109, 54, 52, then **25** (stride 8), **12** (stride 16) and **5**
  (stride 32), not 28, 14 and 7, with non-zero origins. InceptionV3: stride-8 point after `Mixed_5d`,
  middle map after `Mixed_6e`, final after `Mixed_7c`. InceptionResNetV2: stride-8 point after the
  `repeat` (block35) group, middle after `repeat_1` (block17), final after `conv2d_7b`.
  The spatial-token mapping derives each origin from the layers (valid k×k stride-s: centre
  s·i + (k−1)/2).

## Weight scale and starting settings

Convolution-weight RMS (measured): DenseNet-121 0.105 to 0.150, median 0.117; InceptionV3 0.103 to
0.195, median 0.156; InceptionResNetV2 (BatchNorm convolutions) median 0.132; all flat across depth,
2.5 to 3.3 times FocalNet-T's 0.047. An exact rescale to 0.047 is not possible: with Keras's epsilon
0.001 the rescaled variances fall to the order of epsilon (output change 0.5 on a scale of 1), and
DenseNet's concatenated features feed many BatchNorms. So the settings follow the weight-scale rule
of [prior work](prior-work.md#starting-settings) against FocalNet-T (RMS 0.047, rate 2e-4, anchors
0.001 and 0.00001): the rate in proportion to the RMS (equal relative step) and the anchors in
inverse proportion to its square (equal cost per relative movement):

| Backbone | RMS ratio to FocalNet-T | backbone_lr | anchor_alpha_early | anchor_alpha_late |
|---|---:|---:|---:|---:|
| DenseNet-121 | 2.49 | 5.0e-4 | 1.6e-4 | 1.6e-6 |
| InceptionV3 | 3.32 | 6.6e-4 | 9.1e-5 | 9.1e-7 |
| InceptionResNetV2 | 2.81 | 5.6e-4 | 1.27e-4 | 1.27e-6 |

The head rate stays 6e-4. Ten epochs.

## Time and memory (measured, 3090, bf16, channels last, 64 images forward and backward)

| Backbone | s per 64 images | GiB per 64 images | Ten epochs, projected |
|---|---:|---:|---:|
| ResNet-50 (calibration) | 0.064 | 2.9 | 2,617 s measured |
| FocalNet-T SRF (calibration) | 0.103 | 4.5 | 3,390 s (339 s per epoch measured) |
| InceptionV3 | 0.051 | 1.9 | about 2,400 s |
| DenseNet-121 | 0.083 | 4.1 | about 3,000 s |
| InceptionResNetV2 | 0.108 | 3.5 | about 3,500 s |

Projection: seconds per epoch ≈ 138 + 1,954 × (backbone seconds per 64 images), fitted on the two
calibration runs. All fit the 4,092 s bound of the time (now 4,881 s) at batch 8 and under 17 GiB at 256 images; no batch
change is needed. DenseNet-121 builds with `memory_efficient=False` (timm's default), no
checkpointing.

## Precedents

- Original RadImageNet ResNet-50 (2026-09-28, old ReXNet-era recipe, single rates 1.23e-3 and 3e-4,
  no anchors): ×1 0.91223 against the ReXNet control's 0.924637; weights at RMS 0.314 flat. Never run
  on the current recipe.
- Refined-RadImageNet ResNet-50 (the same images, torchvision training): the submission backbone,
  `resnet50-silver-fold2` ×4 0.937820.
- Old backbone search (gold-selected, old pipeline): RadImageNet DenseNet-121 0.816707, 39th of 40.

## Integration defect seen on the first build

The first DenseNet-121 candidate wrapped timm's model in its own module without `pretrained_cfg`;
the preflight (`scripts/tools/preflight_training.py`, which records
`model.backbone.pretrained_cfg`) stopped on `AttributeError` after 13 s. A wrapper must expose
`pretrained_cfg` (the timm model's, with this file's mean 0 and std 1) like every other backbone.

The first InceptionV3 candidate stopped in the preflight with `The size of tensor a (144) must match
the size of tensor b (100)`: `modeling/model.py` adds the stride-32 tokens to the stride-16
meniscus map with `repeat_interleave(2)` on both axes, which assumes the fine grid is exactly twice
the coarse one (14 and 7). The Inception grids are 12 and 5 (origins 21 and 37 for InceptionV3;
InceptionResNetV2 has the same grids), so 5 × 2 = 10 rows meet 12. The fix (keeper's analysis): map
each fine cell to the coarse cell whose centre is nearest, from the two mappings (`meniscus_mapping`
and the final map's), clamped to the grid, as an index gather on the token axis. Fine cell i has
centre origin_f + stride_f·i; its coarse cell is clamp(round((origin_f + stride_f·i − origin_c) /
stride_c), 0, coarse − 1), with ties rounded down. For every existing backbone this equals the
current 2×2 repeat exactly (ConvNeXt-style anchors 7.5 and 15.5: (16i − 8)/32 rounds down to
⌊i/2⌋; origin-0 anchors: 16i/32 = i/2 rounds down to ⌊i/2⌋), so their outputs stay bit-identical;
check that equality on the CPU for every existing backbone. The same 2×2 repeat appears a second time,
in the cartilage-zone weighting (`model.py`, the second `repeat_interleave(2, 1)` /
`repeat_interleave(2, 2)` pair, near `cartilage_weight`); both sites need the gather.
