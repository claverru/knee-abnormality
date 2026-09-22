# CoAtNet 224 spatial backbone

These notes describe the [full-slice model](current-model.md), which supports
CoAtNet again since the restoration of 2026-09-16. The
dropped crop-selection and learned-zoom models could not use it: the crop models rejected its
anchor origin of 12, which is not a multiple of the scout stride, and learned zoom's 160 px
context glimpses did not fit its fixed input size.

The supported alternative is `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k` with
224-pixel inputs and per-channel mean/std `[0.5, 0.5, 0.5]`, matching its
pretrained normalization. [`configs/coatnet.yaml`](../../configs/coatnet.yaml) follows the ReXNet
[training config](../../configs/training.yaml) with `model.backbone`, `data.mean` and `data.std`
changed, and `trainer.batch_size: 2` with `accumulate_grad_batches: 4` to keep the effective batch
of eight. Since 2026-09-21 its optimizer uses transformer fine-tuning values:

| Setting | Value | ReXNet value |
|---|---|---|
| `backbone_lr` | 7e-5 | 2e-4 |
| `head_lr` | 6e-4 | 6e-4 |
| `weight_decay` (norms and biases excluded) | 0.05 | 0.01 |
| `warmup_epochs` | 2 | 1 |
| `warmup_start_ratio` | 0.1 | 0.1 |
| `minimum_lr_ratio` | 0.001 | 0.01 |

These values follow common practice for fine-tuning attention models; none is measured on this
data. Per-layer LR decay and stochastic depth would need code and are not available; the warm-up start
is tunable through `warmup_start_ratio`. The batch split rests on an estimate, not a measurement: per image, CoAtNet's leaf-module
outputs are about 2.9× ReXNet-200's, whose batch-8 peak is 20.02 GB. Keep the saved effective
configuration for reproduction; a run name does not select a backbone.

The model passes `data.image_size` explicitly to CoAtNet so timm builds its
relative-position geometry for the actual input size. The adapter requires a
size divisible by 32. At 224, the backbone produces `[N, 1024, 7, 7]`, giving
49 tokens with a learned 1024→256 projection and the existing MIL head.
The supported checkpoint, normalization and feature shape were checked against
the [model card](https://huggingface.co/timm/coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k)
and installed timm 1.0.26 source on 2026-09-14.

Spatial anchors follow the MBConv main-branch convolution centers. The stem
and two convolution stages retain origin 0 at strides 2, 4 and 8. The two
attention stages use 2×2 average pooling, shifting the origin by 4 and then
8 input pixels. The final anchor on either axis is therefore `12 + 32 * index`.
The MBConv downsampling shortcuts use average-pool centers offset by half the
incoming stride from the main branch. The adapter checks that offset, equal
branch strides and aligned attention-stage shortcuts, while keeping the
main-branch convention fixed. These are nominal token/query anchors; residual
mixing and attention do not give each output a unique receptive-field center.

Construction derives and checks the actual mapping and NCHW channel layout.
Checkpoint, input and source identities record the selected backbone, origin,
stride, layout and anchor convention; `backbone_identity.json` records the same
mapping. Validation, resume and bundled export use the saved backbone when
checking identities. Historical checkpoints retain their archived source or
bundle and must not have their identities rewritten.

Static review of the CoAtNet addition in `ea19772` on 2026-09-14 found no change to ReXNet's model arguments,
convolution trace, anchor formula or parameter initialization order. The CoAtNet
image-size argument and divisibility guard are conditional on that backbone.
All model methods outside construction, feature geometry, spatial embeddings,
training steps, loss and optimizer were unchanged by that addition. ReXNet's additional NCHW channel
check agrees with timm 1.0.26's `num_features` definition. The baseline keeps its
ReXNet normalization, learning rates, batch size and augmentation settings.

The source fingerprint and mapping metadata changed, so historical ReXNet
checkpoints also require their archived runtime; this is distinct from changing
fresh ReXNet training behavior. No model execution, tests or GPU access were used
for this review. It establishes static noninterference, not measured numerical
parity, throughput or a new CoAtNet validation result.
