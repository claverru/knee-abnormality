# Refined-RadImageNet prior work

What is known before the first run. Every run row is one run and weak evidence. Measurements
without a run are label-free checks made on 2026-10-04.

## What the refinement is

- **Dataset:** Refined-RadImageNet is the original RadImageNet (1,354,913 images, 165 classes)
  with ultrasound overlays removed and new stratified splits. MRI and CT images pass through
  unchanged ([converter](https://github.com/convergedmachine/Refined-RadImagenet/blob/main/process.py)).
  For knee MRI the pretraining data is the original's.
- **Recipe:** torchvision's reference classification script, 90 epochs, SGD 0.1, step decay,
  AutoAugment, train crop 224. Its presets normalize with **mean 0.3162 and std 0.3213** on all
  three channels (`presets.py` in the authors' training code), not the ImageNet values.
  CutMix is configured but the loader's collate does not apply it.
- **Knee content:** 18 knee MRI classes, 179,555 images (13% of the set): chondral abnormality
  65,686, meniscal abnormality 44,793, soft-tissue fluid collection 25,757, ACL pathology
  10,085, bone inflammation 8,943, MCL pathology 7,911, normal 4,593, intra 4,290, and ten
  classes under 2,000 (soft-tissue edema, fracture, patella, PCL, quad, muscle strain, FCL,
  soft-tissue mass, hematoma, post-operative ACL). Counted from the public manifests. Class
  indices follow the sorted folder names `MR__knee__<class>`, 83–100.

## Checkpoints

All in the torch hub cache, hashed in `PRETRAINED_WEIGHTS`. Overall convolution-weight RMS is
the quantity of the [weight-scale rule](../../../docs/modeling/backbone-swaps.md#checkpoint-weight-scale).

| Weights | Backbone | What it is | Conv RMS |
|---|---|---|---:|
| `radimagenet` | ResNet-18 | standard training, the incumbent's start | 0.01244 |
| `radimagenet` | ResNet-10t | standard training | 0.01576 |
| `radimagenet` | ResNet-50 | standard training | 0.00952 |
| `radimagenet_policy` | ResNet-10t | classifier trained under the authors' policy mask ("heated") | 0.05305 |
| `radimagenet_policy_ft` | ResNet-10t | the heated classifier after ten more epochs without the mask | 0.05181 |
| `radimagenet_policy` | ResNet-50 | heated | 0.02653 |
| `radimagenet_policy_ft` | ResNet-50 | heated, then fine-tuned | 0.02591 |
| `radimagenet_crossd` | ResNet-18 with Cross-D convolutions | 3D kernels (K×K×K) and a rotation network per convolution, AdamW, 90 epochs, 33.4M parameters | 0.190 |

- The policy-gradient files hold only the classifier (86 and 320 tensors, with `fc`). The mask
  policy network was never published: the weight repository has held the same four files since
  its first commit, the code repository has no releases, and the authors' other repositories
  hold nothing related.
- Every checkpoint carries its 165-class `fc`.

## Pretrained head on our slices

The frozen Refined-RadImageNet ResNet-18 with its 165-class head, on the prepared middle slices
of the first 320 fold-0 validation studies (8,615 slices), CPU, no training:

| Input normalization | Probability mass on the 18 knee classes | Slices whose top class is a knee class |
|---|---:|---:|
| ImageNet mean and std (the incumbent) | 0.088 | 7.8% |
| mean 0.3162, std 0.3213 (the checkpoint's own) | 0.978 | 98.5% |
| none, inputs in [0, 1] | 0.984 | 99.1% |
| mean 0.5, std 0.5 | 0.976 | 98.4% |

Under ImageNet normalization the top class is `US__thyroid` (0.387 of the mass). The earlier
note that normalization "rests on the recipe" assumed the script's ImageNet default
([weights](../../../docs/research/radimagenet-weights.md#refined-radimagenet-resnets)); the
authors' presets set their own values. Our prepared slices have mean 0.335 and std 0.252.

Zero-shot signal under the checkpoint's normalization, study mean of the slice logit of the
class mapped to each target, AUC against `rebuild_state_v2` on those 320 studies: ACL 0.699, MCL
0.710, Medial Meniscus 0.695, Lateral Meniscus 0.705, Medial OA 0.706, Lateral OA 0.764, PF OA
0.640, Effusion 0.532, Synovitis 0.570, Baker's 0.603, Contusion 0.827, Fracture 0.851 (mean
0.692). Under ImageNet normalization no class of the 165 exceeds 0.74 for any target. This is a
go/no-go check on validation labels: the [class mapping](contract.md#build-facts) was fixed from
the class names, not from these numbers, and nothing may be tuned on them.

## The authors' mask method

From the [paper](https://arxiv.org/abs/2406.14568) and the
[training code](https://github.com/convergedmachine/Policy-Gradient-Driven-Noise-Mask):

- A policy network (a ResNet-10t built without pretrained weights, 512 features, and two MLP
  heads of 512-512-4096 with BatchNorm and ReLU) outputs per image the logarithms of the two
  Beta parameters of each cell of a 64 × 64 grid: `new = 0.9 · base + 0.1 · head(features)`,
  where `base` is a running batch mean of the head outputs with momentum 0.99, started from
  `torch.rand`.
- The mask is one sample of `Beta(exp(new_alpha), exp(new_beta))`, not reparameterized,
  resized to 224 by nearest neighbour and Gaussian-blurred (kernel 13, sigma 6). The classifier
  sees `normalized_image × mask`.
- One loss trains both: `mean(log(sum_cells exp(log_prob)) × cross_entropy)`. The policy
  optimizer is SGD at a tenth of the classifier's rate with cosine decay.
- At evaluation the heated model is run with masks sampled from `base` alone. On unmasked input
  it does poorly until fine-tuned without the mask (the `_FT` checkpoint).
- The paper reports that the dataset-level parameters converge to an almost uniform
  distribution, and best results at grid 64, kernel 13, sigma 6.
- Reported RadImageNet macro F1: ResNet-10t 0.484 against 0.452 for the baseline, ResNet-50 0.547
  against 0.544. No knee result.

## Cross-D-Conv

From the [code](https://github.com/convergedmachine/Cross-D-Conv) and the checkpoint:

- Each convolution holds a 3D kernel `weights_3d` (out × in × K × K × K; K is 7 for the stem, 3
  in the blocks, 1 in the downsample paths) and a small network that predicts, per image, an
  axis and an angle. In 2D use the kernel is shifted in the Fourier domain by
  `1 + angle · (axis cross-term)` voxels per axis and its middle depth slice is the 2D kernel
  for that image.
- For 3D use the authors convert every convolution to a plain `Conv3d` with the same stride on
  all three axes, BatchNorm to 3D and max pooling to 3D, and load `weights_3d` as the 3D weight.
- The checkpoint is a torchvision-layout ResNet-18, 302 tensors, keys `conv1.weights_3d`,
  `conv1.rotation_params.*`, `layerX.Y.convZ.*`, `layerX.0.downsample.0.*`, `fc.*`.
- Its weights are 15× the incumbent's scale and uniform across depth, like the original
  RadImageNet ResNet-50, whose first run at the rule's rate overwrote its early layers
  ([first run](../../../docs/ongoing/radimagenet-backbone.md#first-run-learning-rate-too-high-2026-09-28)).

## Runs

| Mechanism | Run | Recipe | ×4 macro | Note |
|---|---|---|---:|---|
| Incumbent | `20261003-100844-rn18-view-label-bias` | ResNet-18 | 0.932477 | reference, 1,181 s |
| ResNet-10t, standard weights | `20261004-094416-rn10t-view-label-bias` | incumbent recipe, backbone LR 6e-5 | 0.927543 | reference of seeds 2 and 3, 1,141 s |
| Refined ResNet-50 | `20261003-110045-refined-rn50-views-label-bias` | incumbent recipe | 0.937129 | 0.954 public as S102 |
| Original RadImageNet ResNet-50 | `20260928-180252-radimagenet-resnet50-lr3e-4` | older recipe, [−1, 1] inputs | ×1 0.91223 against 0.924637 | weights 20× the usual scale |

## Measured on ResNet-18

Do not propose these again at the same value
([radimagenet-rn18 findings](../radimagenet-rn18/findings.md),
[revisits prior work](../revisits/prior-work.md#measured-on-resnet-18)):

- **Rates and schedule:** backbone LR 1e-4 (−0.0017), layer-wise LR decay (−0.0006), two warmup
  epochs (−0.0007), warmup-stable-decay (−0.0003), 14 epochs (+0.0010 for 40% more time),
  separate backbone and head clipping (−0.0024).
- **Backbone handling:** linear probe then fine-tune (−0.0012), frozen stem and layer1
  (−0.0029), L2-SP (adopted), BatchNorm-adapted head start (−0.0004), frozen BatchNorm
  (−0.0086), early-stage anchor (−0.0004), plane-conditioned BatchNorm (−0.0010).
- **Adjacent slices:** mean-initialized adjacent slices as input channels (−0.0039), gated
  adjacent residuals (−0.00005). Both feed neighbours through the first 2D convolution; the
  Cross-D and ACS seeds use depth kernels instead.
- **Regularization:** series dropout 0 (adopted), DSU 0.25 (+0.00005), stochastic depth
  (rejected in revisits).
