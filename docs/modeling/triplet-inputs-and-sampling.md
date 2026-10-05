# Triplet inputs and sampling

## Three-channel inputs

Every input is an adjacent triplet (`SLICES_PER_INPUT = 3` in `config.py`; there is no
input-mode or channel-count field): each three-channel
CNN input contains three consecutive slices from the physically sorted series.
Channels retain their slice order and are normalized, after dividing uint8 pixels by 255,
with the statistics the Refined-RadImageNet checkpoints were trained with: mean 0.3162 and
standard deviation 0.3213 on every channel (`data.mean` and `data.std` take exactly three
values; since 2026-10-05,
[result](../validation/results/2026-10-04-refined-radimagenet-campaign.md)). The ImageNet
values `[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]` were used before, for the ReXNet
and historical RA2 checkpoints. The training config prepares 224 px
triplets at 0.536 mm, covering **120 × 120 mm** (since 2026-10-01,
[result](../validation/results/2026-10-01-mixed-field-campaign.md); the ReXNet recipe keeps
0.625 mm and 140 mm), after the
[completed crop comparison](../validation/results/2026-09-12-140-mm-crop-result.md). The dropped
[crop-selection investigation](../history/crop-selection-investigation.md#recipes) used 300 px
canvases at 0.5 mm, and then 360 px canvases at 0.4167 mm for learned zoom.

The dataset supplies uint8 `[inputs, 3, height, width]` tensors per series.
Collation pads only to the largest input count present in the batch. Each
validity-mask entry represents a complete triplet, and padding never enters the
CNN or attention. A triplet's positional encoding uses its middle slice's
millimeter position relative to the physical center of its acquisition. All
triplets of an acquisition share sampled augmentation strengths; normalization
is applied per channel.

Channel-appearance limits must be divisible by three. Checkpoint loading compares
normalization, slice limits and the other data and model settings, so an older artifact
cannot silently adopt current input semantics. Single-slice configs such as baseline1's
(`data.input_mode: single_slice`, one channel, scalar normalization) no longer validate and
need their archived sources; their exact prediction reproduction is recorded in the
[baseline review](../history/baseline1-review.md#implemented-corrections).

## Adjacent triplet sampling

The current recipe draws independently inside each Gaussian probability band
at every training epoch, and uses deterministic midpoints during validation.
See the [implemented sampler](../data-preparation/physical-sampling/implemented-sampler.md).
The table below records historical validation x2 budgets; current training
comparisons use x1, so validation counts match the training-density column.

Updated on 2026-09-13. Budgets are now set per view as `slices`, the "CNN inputs"
column below ([view table](current-model.md#view-table)); until 2026-10-02 the setting was
`max_images`, which counted **channel appearances**, including slices reused by overlapping
triplets.
Training limits round the previous half-median budgets up to multiples of
three. `validation_image_multiplier: 2` doubles validation limits. Test
prediction independently defaults to 2× (`TEST_IMAGE_MULTIPLIER` in `config.py`). Export can freeze a different inference
multiplier without changing the checkpoint's training settings:

| Plane | Fluid-sensitive / fat suppression | Training channel appearances | Training CNN inputs | Validation channel appearances | Validation CNN inputs |
|---|---|---:|---:|---:|---:|
| Axial | 0 / 0 | 18 | 6 | 36 | 12 |
| Axial | 1 / 1 | 18 | 6 | 36 | 12 |
| Coronal | 0 / 0 | 15 | 5 | 30 | 10 |
| Coronal | 1 / 1 | 15 | 5 | 30 | 10 |
| Sagittal | 0 / 0 | 15 | 5 | 30 | 10 |
| Sagittal | 1 / 1 | 15 | 5 | 30 | 10 |

A study with all six types uses at most 96 channel appearances and 32 CNN inputs during
training, or 192 channel appearances and 64 CNN inputs during validation. The multiplier
does not change model parameters or checkpoint tensor shapes. Current test
inference defaults to the same 2× limits; 3× allows 288 channel appearances and 96 CNN
inputs per study, capped by available acquisitions. Earlier submitted bundles
used 1× and retain their frozen code. Historical checkpoints require their archived runtime. See [inference memory measurements](../inference/inference-image-count-and-memory.md).
