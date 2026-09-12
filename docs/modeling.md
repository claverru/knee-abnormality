# Modeling

Evidence: current repository artifacts through 2026-09-11 and unverified
working notes; see [provenance](INDEX.md#provenance).

The most concrete baseline lead is a fast ReXNet backbone applied to
three-slice windows, with multiple-instance learning (MIL) at study level.
The full architecture is unavailable.

The separate [public-source investigation](public-research.md#geometry-sampling-encoders-and-losses)
adds controlled and anecdotal positive/negative results, including small-versus-
large encoders, slice geometry, losses and medical pretraining. Its
[notebook comparison](public-research.md#public-code-worth-inspecting) identifies
inspectable DINO and CoAtNet implementations. Those are additional baselines,
not implementations of the unverified model described below.

## Implemented baseline

The completed `outputs/baseline3` fold-0 run uses a pretrained
`resnet50d.ra2_in1k` encoder and a learned 2,048-to-256 projection. The
maintained [baseline config](../configs/baseline.yaml) now selects pretrained
`rexnet_200.nav_in1k`, with its 2,560-to-256 projection, for the next fresh run.
Both use target-specific attention MIL over every valid triplet from the
retained plane–contrast slots. Type embeddings and centered physical slice
positions are added before attention. Training drops each present slot with
probability 0.1 and always restores one when every slot would otherwise be
removed.

Baseline3 selected epoch 7 at **0.849819 macro AUC** over all 903 fold-0
studies. Its separate 15-study gold diagnostic was **0.915586** and its
report-derived diagnostic was **0.849472**. Training all ten epochs took
1,753.10 seconds for 4,380 optimizer steps and peaked at 11,677,091,328 CUDA
bytes. This is one-fold development evidence on the mixed-label endpoint, not
a complete five-fold OOF result or a comparison with baseline1's different
12-study endpoint. The saved configuration, checkpoint selection, target-level
metrics and timing live under `outputs/baseline3/`.

The completed `outputs/baseline4` fold-0 run uses
`rexnet_200.nav_in1k`, doubled validation image limits, contiguous NCHW
tensors, and a physical batch of eight. It selected epoch 7 at **0.872510
macro AUC**, trained for 2,370.34 seconds, and peaked at 21,505,119,744 CUDA
bytes. Its exact checkpoint bundle produced Kaggle public score **0.910** as
submission `56167669`, using the historical 1× test image limits. This remains
single-fold development evidence. The same checkpoint subsequently scored
**0.913 at 2×** and **0.912 at 3×**, both with inference batch size 2. Current
inference uses 2×; see the [completed comparisons](inference-efficiency.md#inference-image-count-and-memory).

The backbone trains in ordinary training mode: `backbone_eval: false` and
`checkpoint_backbone: false`. Lightning owns precision, optimization, gradient
clipping, early stopping, logging and independent best/latest checkpoints.
Saved checkpoint configuration governs prediction; later edits to the baseline
YAML do not change an existing artifact.

### Three-channel inputs

The baseline uses `data.input_mode: adjacent_triplets`: each three-channel
CNN input contains three consecutive slices from the physically sorted series.
Channels retain their slice order and use the selected ReXNet and historical
RA2 checkpoints' ImageNet normalization: mean `[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]`, after dividing uint8 pixels by 255. The 224 px crop at
0.67 mm covers **150.08 × 150.08 mm**.

The dataset supplies uint8 `[inputs, 3, height, width]` tensors per series.
Collation pads only to the largest input count present in the batch. Each
validity-mask entry represents a complete triplet, and padding never enters the
CNN or attention. A triplet's positional encoding uses its middle slice's
millimeter position relative to the physical center of its acquisition. All
triplets of an acquisition share sampled augmentation strengths; normalization
is applied per channel.

Legacy configs default to `data.input_mode: single_slice`. Their prepared
one-channel tensors are either broadcast to RGB or kept single-channel according
to the saved configuration. Checkpoint loading compares input mode, channel
count, normalization and slice limits, so an older artifact cannot silently
adopt current input semantics. Triplet mode requires three input channels and
raw-slice limits divisible by three. Baseline1 compatibility and exact
prediction reproduction are recorded in the
[baseline review](baseline-review.md#implemented-corrections).

### Adjacent triplet sampling

Implemented on 2026-09-11. `max_images` continues to count **raw slices**.
Training limits round the previous half-median budgets up to multiples of
three. `validation_image_multiplier: 2` doubles validation limits. Test
prediction independently defaults to 2×. Export can freeze a different inference
multiplier without changing the checkpoint's training settings:

| Plane | Fluid-sensitive / fat suppression | Training raw slices | Training CNN inputs | Validation raw slices | Validation CNN inputs |
|---|---|---:|---:|---:|---:|
| Axial | 0 / 0 | 18 | 6 | 36 | 12 |
| Axial | 1 / 1 | 18 | 6 | 36 | 12 |
| Coronal | 0 / 0 | 15 | 5 | 30 | 10 |
| Coronal | 1 / 1 | 15 | 5 | 30 | 10 |
| Sagittal | 0 / 0 | 15 | 5 | 30 | 10 |
| Sagittal | 1 / 1 | 15 | 5 | 30 | 10 |

A study with all six types uses at most 96 raw slices and 32 CNN inputs during
training, or 192 raw slices and 64 CNN inputs during validation. The multiplier
does not change model parameters or checkpoint tensor shapes. Current test
inference defaults to the same 2× limits; 3× allows 288 raw slices and 96 CNN
inputs per study, capped by available acquisitions. Earlier submitted bundles
used 1× and retain their frozen code. Saved configs without the validation field
use validation multiplier one. See [inference memory measurements](inference-efficiency.md#inference-image-count-and-memory).

### ReXNet-100 runtime

The maintained config uses BF16 mixed precision and processes an effective
batch of eight as one physical batch. A bounded maximum-load run on the local
RTX 3090 peaked at 8.40 GiB with `batch_size: 8`; the earlier physical batch of
four with two-step accumulation peaked at 4.55 GiB and did twice as many
forward/backward passes for the same effective batch.

ReXNet-100 uses contiguous NCHW tensors. With channels-last tensors, each new
flattened image count triggered roughly ten seconds of cuDNN setup; MIL batches
vary that count frequently because studies contain different valid series and
series dropout changes it further. Contiguous tensors handled changing counts
in 0.14–0.15 seconds after a 0.79-second first step in the isolated local
benchmark. This is slightly slower for a repeated fixed shape but avoids the
repeated dynamic-shape penalty that dominated real training.

The bounded production-path profile then sustained **35.43 studies/second**
across two measured optimizer steps with real DICOM batches, compared with
**0.54 studies/second** in the channels-last diagnostic dominated by repeated
setup. This profile did not run an epoch or produce a validation score.

Training disables Lightning's parameter summary because its size calculation
does not recognize `bf16-mixed`; this affects only the displayed model-size
estimate. Float32 matrix multiplication uses the `high` policy so supported
operations can use the RTX 3090 Tensor Cores.

`data.acquisition_sampling: pooled` keeps every usable acquisition in a
plane–contrast slot and distributes that slot's input budget across them.
Training samples from their combined capacities; validation first applies its
limit multiplier, then allocates counts proportionally with deterministic
remainder handling. A failed acquisition is
removed and the budget is reallocated among those remaining.

Within each acquisition, `slice_sampling: center_weighted` assigns possible
triplet starts weight 3 at the physical center and weight 1 at the edges.
Training samples a reproducible non-overlapping packing from those weights.
Evaluation deterministically chooses non-overlapping starts near the weighted
quantiles. This favors central anatomy without restricting inputs to a fixed
central band. Uniform triplet placement remains supported for saved configs and
the planned paired comparison.

Preparation retains joint `[0.005, 0.995]` clipping bounds over valid pixels in
the selected complete triplets of each acquisition. A decode or non-finite
pixel failure drops its complete triplet. An acquisition with no surviving
input is removed; an absent slot is masked, and a study with no usable slot
fails explicitly. Adjacency follows the physically indexed usable records
after known malformed files and unreadable headers are excluded.

The [augmentation viewer](data-preparation.md#preview-augmentations) displays
both the three-channel composite and each individual grayscale channel.

### Ten-epoch schedule and augmentation

Training uses
`max_epochs: 10`, one warmup epoch, and cosine decay over the remaining nine
epochs toward 1% of the peak learning rates. `patience: 10` prevents ordinary
validation fluctuations from stopping this ten-epoch budget early. Best and
latest checkpoints remain separate. The peak backbone/head learning rates are
`2e-4` and `6e-4`, adopted in `configs/baseline.yaml` on 2026-09-12 from
the [overnight comparison](validation.md#2026-09-12-overnight-results).

Each acquisition independently samples affine augmentation with probability
0.7, gamma/contrast with probability 0.5 and Gaussian noise with probability
0.15. Affine strengths are up to 10 degrees rotation, 4% translation and
`[0.95, 1.05]` scale. Gamma uses `[0.85, 1.15]`, contrast uses `[0.9, 1.1]`,
and noise standard deviation uses `[0.005, 0.02]` on the `[0, 1]` scale.
Geometry and scalar strengths are shared by every triplet in an acquisition;
noise pixels differ. Zero background is restored after augmentation.

Evaluation bypasses augmentation. Each family defaults to disabled when absent
from an older config, and probability zero disables it for a paired comparison.
The current combined recipe does not establish the contribution of any one
family; those comparisons remain in
[future work](future-work.md#additional-mri-research-candidates).

Use the [augmentation viewer](data-preparation.md#preview-augmentations) to
inspect original/augmented example-test slices for each plane–contrast type,
with per-series parameters and seeded draws from the shared implementation.

### Baseline history and compatibility

The completed `outputs/baseline1` fold-0 run selected epoch 4 at **0.843997
macro AUC** over twelve reference studies, taking 52.37 minutes through ten
epochs with batch 8 and accumulation 1. It used one-channel inputs, ordinal
positions, a ResNeXt26ts encoder, one CLS transformer and masked max pooling
under the historical gold-only split. Its exact configuration, defects and
reproduction procedure belong in the [baseline review](baseline-review.md).
Later saved configs preserve the RA4 experiment and other intermediate recipes;
none overrides baseline3's current defaults.

## Batch-4 memory fix

Superseded single-slice ResNeXt profiles established two durable constraints.
First, processing hundreds of slices in one encoder batch exceeded the RTX
3090's practical memory capacity; channels-last plus activation checkpointing
reduced one 740-slice batch-four profile from 31.04 GiB allocated to 12.84 GiB.
Second, four to eight loader workers mostly hid preparation time in those
bounded workloads. Exact configurations and measurements remain in
`outputs/profiling/batch4-*`, while the more representative loader comparison
and its negative threading result live in
[data preparation](data-preparation.md#standalone-loading-measurement).

### Profiled training runtime

Those profiles do not describe baseline3, whose triplets greatly reduce CNN
input count and whose saved configuration disables activation checkpointing.
Use full-run timing and memory from the current baseline above for planning.

## Unverified model structure

- The setup was described as MIL with one representation per window rather
  than one per series. The exact pooling/attention mechanism, metadata inputs,
  and output heads are unknown. The available detail does not establish
  whether the implementation contains a transformer.
- Separate metadata features may have been tried early and later dropped.
  This does not establish whether metadata was
  absent from the classifier, or change its role in geometry preprocessing.
- Three neighboring slices are packed into the input channels. Five-slice
  inputs reportedly helped some models, without a quantified result.
- Repeating one slice across three channels tested the same or worse than
  using neighboring slices. Reducing the input convolution to one channel was
  debated without a demonstrated result.
- Input normalization appeared to depend on the backbone, mostly using
  ImageNet normalization.
- `rexnetr200` was recommended as a fast backbone, with about one hour per
  fold at 384 pixels in the recorded setup. `rexnet300` and later
  `rexnetr300` were also named. Confirm the exact library identifiers and
  pretrained checkpoints rather than treating these spellings as interchangeable.
- An unnamed stronger model took roughly 6–8 hours to train. Improvements
  found with a small model reportedly transferred to that larger model.

## Window sampling

Early experiments trained on a changing random subset of about half the slices.
Training on all slices showed no gain; using all slices at inference initially
gave a small lift.

The more specific 2026-09-08 description was:

- Training: 24 windows, each containing three slices.
- Validation: 94 windows were stated explicitly.
- A later explanation used 96 windows and said about 70% of studies had fewer
  than that; increasing the count reportedly reduced the score.
- Outer slices were described as frequently uninformative.

The 94/96 discrepancy, cap scope (study versus series), overlap between windows,
and exact inference policy remain unresolved. The later capped description
qualifies the earlier claim that inference uses all images. Twenty-four windows
also need not mean 72 distinct slices if windows overlap.

## Augmentation and complementary views

- Adding an unspecified augmentation set produced a reported local improvement
  large enough for a single-fold model to beat an earlier seven-model ensemble.
  Public confirmation was explicitly still pending the next day.
- Horizontal flipping improved local results despite concerns about
  side-dependent targets. The label/orientation policy and exact transform
  configuration were not recorded, so this is an ablation lead, not an
  established label-preserving default.
- Whole-series dropout was described on 2026-09-09 as the best overall
  improvement found, forcing learning from other planes. No dropout probability
  or score delta was given.
- Different models selected important windows from different planes. Sagittal
  views were said to dominate, with rough estimates varying from 60% to 70%;
  coronal contribution was once estimated at 25%. The attribution method and
  target breakdown are absent. This does not establish that other planes can
  be discarded without loss.

## Fast backbone search

An unverified experiment used three models' out-of-fold predictions with
leave-one-out inference to identify an important three-slice window.
A reduced dataset mapping one selected window to study targets let candidate
backbones train in about 5–10 minutes. This reportedly helped discover the
`rexnetr300` candidate and transferred to the full task.

A proposed extension would select one important window per target, about
12 windows per study. That extension was not reported as completed. Selection
details, ranking aggregation across models, and validation isolation are missing.

### Finished model comparison

The following unverified completed-run summary was recorded on 2026-09-11.
Configuration, split, seed and artifact identities are unavailable, so these
values rank that experiment set but do not establish a controlled comparison
with repository baseline3. ReXNet-200 has the highest reported AUC and is the
next maintained baseline-config backbone.

| Finished model | Best AUC | Selected epoch | Epochs completed | Cap | Run time | Stop reason |
|---|---:|---:|---:|---:|---:|---|
| ReXNet-200 | *0.908347* | 5 | 11 | 20 | 13m 46s | Patience 6 |
| DINOv3 ConvNeXt Tiny | 0.904386 | 7 | 13 | 20 | 11m 33s | Patience 6 |
| CoAtNet | 0.904336 | 5 | 11 | 20 | 40m 17s | Patience 6 |
| RegNetY-160 IN12k | 0.901071 | 14 | 20 | 30 | 46m 52s | Patience 6 |
| ReXNet-200 ordered map mixer | 0.897025 | 6 | 12 | 30 | 14m 51s | Patience 6 |
| DINOv2 Base reg4 | 0.895418 | 5 | 11 | 30 | 22m 53s | Patience 6 |
| ReXNet-300 | 0.894535 | 8 | 14 | 20 | 24m 40s | Patience 6 |
| RegNetY-120 IN12k | 0.894422 | 6 | 12 | 30 | 23m 46s | Patience 6 |
| DINOv3 ConvNeXt Small | 0.890972 | 8 | 14 | 30 | 19m 04s | Patience 6 |
| RegNetY-064 IN1k | 0.890054 | 12 | 18 | 30 | 27m 41s | Patience 6 |
| ReXNet-200 ordered window mixer | 0.888753 | 9 | 15 | 30 | 18m 04s | Patience 6 |
| DINOv3 Small gated, cap 30 | 0.888279 | 12 | 18 | 30 | 13m 56s | Patience 6 |
| DINOv3 Small mean/max, cap 30 | 0.887999 | 18 | 24 | 30 | 17m 40s | Patience 6 |
| ConvNeXt Tiny supervised | 0.886954 | 9 | 15 | 30 | 13m 14s | Patience 6 |
| DINOv3 Small gated | 0.886726 | 19 | 20 | 20 | 15m 16s | Epoch cap |
| DINOv3 Base | 0.885491 | 5 | 11 | 30 | 20m 00s | Patience 6 |
| DINOv3 Small gated acquisition | 0.884663 | 22 | 28 | 30 | 21m 27s | Patience 6 |
| ReXNet-200 spatial queries | 0.883935 | 10 | 16 | 30 | 20m 46s | Patience 6 |
| RadImageNet ResNet-50 | 0.883363 | 29 | 30 | 30 | 25m 13s | Epoch cap |
| DINOv3 Small gated axis, cap 30 | 0.882248 | 19 | 25 | 30 | 19m 33s | Patience 6 |
| DINOv3 Small gated axis | 0.880551 | 19 | 20 | 20 | 15m 20s | Epoch cap |
| DINOv3 Small mean/max | 0.879972 | 19 | 20 | 20 | 14m 29s | Epoch cap |
| DINOv3 Small Plus | 0.878961 | 7 | 13 | 30 | 11m 26s | Patience 6 |
| CAFormer-S18 IN22k | 0.878179 | 15 | 21 | 30 | 36m 16s | Patience 6 |
| MedDINOv3 Base | 0.875170 | 5 | 11 | 30 | 20m 04s | Patience 6 |
| DINOv2 Small reg4 | 0.874858 | 8 | 14 | 20 | 10m 43s | Patience 6 |
| DINOv3 Small | 0.874582 | 19 | 20 | 20 | 14m 44s | Epoch cap |
| InceptionNeXt Tiny | 0.872661 | 7 | 13 | 30 | 12m 08s | Patience 6 |
| DINOv3 Small last6 | 0.871846 | 10 | 16 | 30 | 7m 55s | Patience 6 |
| ConvNeXt-V2 Tiny FCMAE | 0.871493 | 9 | 15 | 30 | 33m 00s | Patience 6 |
| ImageNet DenseNet-121 | 0.869147 | 8 | 14 | 30 | 16m 19s | Patience 6 |
| DINOv3 Small, cap 30 | 0.868904 | 7 | 13 | 30 | 9m 48s | Patience 6 |
| DINOv3 Large LoRA32 | 0.866161 | 13 | 19 | 30 | 2h 21m 24s | Patience 6 |
| DINOv3 Small CLS + mean | 0.865789 | 5 | 11 | 20 | 8m 12s | Patience 6 |
| BiomedCLIP Base | 0.864368 | 6 | 12 | 30 | 17m 53s | Patience 6 |
| ImageNet ResNet-50 | 0.861902 | 8 | 14 | 30 | 11m 53s | Patience 6 |
| MaxViT Tiny | 0.857384 | 12 | 18 | 30 | 42m 14s | Patience 6 |
| EfficientNetV2-S IN21k | 0.857114 | 7 | 13 | 30 | 15m 16s | Patience 6 |
| RadImageNet DenseNet-121 | 0.816707 | 24 | 30 | 30 | 32m 56s | Patience 6 |
| RadiologyNet ResNet-50 | 0.791780 | 5 | 11 | 30 | 9m 43s | Patience 6 |

## Loss and ensemble diversity

Focal loss improved some target-level diagnostics but worsened others, and
standalone models were described as slightly worse overall. Mixing loss
functions was proposed to improve ensemble diversity; no isolated ensemble
gain was reported. See [validation](validation.md#target-level-diagnostics)
for the AUC/F1 distinction and ensembling uncertainty.

Architectural diversity, decorrelating small models, and distilling ensembles
were recurring directions. Earlier notes also suggested that older, weaker
models could hurt an ensemble. None of the projected final scores are measured
results. The unverified runtime/score pairs are in
[inference and efficiency](inference-efficiency.md#unverified-submission-results).

## Expensive attempt and exploratory ideas

A denoising model took about 24 hours to train, followed by roughly 24 hours
for an autoencoder. Classification using the compressed representation was
slightly worse than using slices. The attempt did not encode across the slice
axis; its intended advantage was faster subsequent iteration through a smaller
cache. Claims about near-lossless compression were aspirations, not results.

Other ideas included exploiting imperfectly aligned retakes for denoising,
3D reconstruction, and model-assisted annotations or segmentation. No
controlled competition results are recorded for those directions.
