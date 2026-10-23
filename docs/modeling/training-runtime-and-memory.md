# Training runtime and memory

## ReXNet-100 runtime

Historical ReXNet-100 measurements; the current ReXNet-200 recipe's time and memory are in
its run records. The config of the time used BF16 mixed precision and processed an effective
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

The shared [physical sampler](../data-preparation/physical-sampling/implemented-sampler.md) selects
Gaussian target depths about the accepted study center, then chooses suppliers
independently of slice count. `data.sampling_sigma_mm` (80.0 at the time; 40.0 later) was
the only variable sampling parameter and has since been replaced by
`data.sampling_sigma_fraction` (0.867 in the current recipe; this profile ran at 0.76), which scales the width with
each candidate pool's own acquired reach instead of a fixed millimetre value. Deterministic validation uses midpoint quantiles; training
uses stratified random quantiles. Both permit overlapping consecutive triplets
and assign each distinct middle-plane geometry at most once per type.
Study geometry, acquisition-midpoint 1D positions, crops, augmentations and model
shape remain unchanged. The prior allocator, triangular selector and their
configuration fields have been removed.

Preparation clips each unique retained source slice once per acquisition, then
gathers its channel appearances. Path failures split original runs and trigger
fresh type selection using fixed seeds and the original study center. Local
percentile fallback retains trustworthy millimetre positions and unknown XYZ
when a type has no physical candidates. Unusable types remain masked and a
study without a complete usable input fails explicitly. See
[overlapping preparation](../data-preparation/current-preparation.md#overlapping-triplet-preparation).

The [augmentation viewer](../data-preparation/preview-augmentations.md) displays
both the three-channel composite and each individual grayscale channel.

## Batch-4 memory fix

Superseded single-slice ResNeXt profiles established two durable constraints.
First, processing hundreds of slices in one encoder batch exceeded the RTX
3090's practical memory capacity; channels-last plus activation checkpointing
reduced one 740-slice batch-four profile from 31.04 GiB allocated to 12.84 GiB.
Second, four to eight loader workers mostly hid preparation time in those
bounded workloads. Training defaults the CUDA allocator to `expandable_segments:True` unless
`PYTORCH_ALLOC_CONF` or `PYTORCH_CUDA_ALLOC_CONF` is set, and records the setting in the run
provenance. Exact configurations and measurements remain in
`outputs/profiling/batch4-*`, while the more representative loader comparison
and its negative threading result live in
[data preparation](../data-preparation/threading-inside-dataset-loading.md#standalone-loading-measurement).

### Profiled training runtime

Those profiles do not describe baseline3, whose triplets greatly reduce CNN
input count and whose saved configuration disables activation checkpointing.
Use full-run timing and memory from the current baseline above for planning.

## Host memory with larger inputs

These measurements come from the dropped
[crop-selection runs](../history/crop-selection-investigation.md). Their 300 px canvases hold
1.79× the pixels of 224 px triplets, so the measurements apply to any larger-input run, such as
the deferred [280 px test](../backlog/untried-ideas.md#views-and-fields) at 1.56×.
The restored baseline, like `00-baseline`, uses 16 workers and `prefetch_factor: 2`, and has no
separate validation worker count. The crop-era `trainer.validation_num_workers` field was
removed. A larger-input run should revisit both settings.

The first [crop-selection run](../validation/results/crops-and-views/2026-09-15-crop-selection-result.md) was
killed by the Linux out-of-memory killer during epoch 8. WSL got 30 GB of the host's
64 GB by default; after the
[per-label crops run](../validation/results/crops-and-views/2026-09-16-per-label-crops-result.md) was killed the
same way, through 7–8 GB of loader shared memory, WSL was raised to 47 GiB.
- **Batch sizes:** 300 px canvases make validation batches about 200 MB and training
  batches about 100 MB of padded uint8 images.
- **Queues:** persistent workers keep both loaders alive, 16 workers each. With
  `prefetch_factor: 2` both queues together held about 10 GB of shared memory, and the
  main process held another 10.6 GB.
- **Symptoms before the kill:** the 8 GB swap was full and memory stalls ran at 19%. A
  py-spy profile put 44% of busy worker time in the `np.stack` copy in `prepare_series`,
  where allocations stalled rather than computed. Validation starved the GPU.
- **Fix:** `trainer.prefetch_factor: 1` halves the queues. The resumed run kept about
  5 GB available with no swap and peaked at 14.5 GB of CUDA allocation. Resume accepts
  changes to `prefetch_factor` and `num_workers`.

### Validation loader waves

In the `two-crops` run, validation took about 70 s of a roughly 340 s epoch and paused every
16 batches, once per loader worker round.
- **Why it stalls:** each validation batch carries 2× images at 300 px, and workers hand
  out new work in bursts. All 16 workers finish together, the GPU drains their 16 batches
  in a few seconds, then waits for the next round.
- **Why training doesn't:** training was GPU-bound, at 93% average utilization with 2–7
  busy workers.
- **Setting at the time:** `trainer.validation_num_workers` (28 in the crop configs) sized
  the validation loader separately, using the cores the idle training workers left free. The
  field was removed on 2026-09-16, so validation uses `num_workers`.

## Posed-crop recipe: where a step goes and what was removed (2026-10-08)

Measured on the anatomy-masks recipe (ResNet-18, posed sagittal and coronal crops, first look),
RTX 3090, with `scripts/tools/bench_training_steps.py` (110 training and 40
validation steps under the real `Trainer`; outputs in `outputs/analysis/a/anatomy-masks-speed/`).

| | Before | After |
|---|---:|---:|
| Training step | 304 ms | 245 ms |
| Wait on the loader per step | 10 ms | 3 ms |
| Training steps per second | 3.18 | 4.04 |
| Validation step | 231 ms | 168 to 182 ms |
| Epoch, from a two-epoch run | 167 to 170 s | 138 to 141 s |
| Ten epochs | 1,687 s | about 1,390 s (projected from two) |

The two-epoch run is `outputs/training/rn18-anatomy-map-centre-fast` (stopped by hand after
epoch 2; validation macro 0.895404 and 0.900571 against 0.893990 and 0.900518 for the same
recipe before the changes).

- **The step is not loader-bound.** The GPU executes about 180 ms of kernels per step; the rest
  is the main thread issuing about 5,500 small operations and waiting at about 265 points that
  force a GPU synchronisation (boolean indexing, `nonzero`, `.item()`, `unique`, Python branches
  on tensor values), most of them in the augmentation and geometry code. Removing them was done on 2026-10-14 (sync-free training step, commit 9f277d4). Audit on the ResNet-50 student recipe (`configs/rn50-distilled-student.yaml` (never committed)), RTX PRO 6000 with an EPYC 7543 host: 281 synchronisations per step, 199 ms of an 809 ms step blocked in them, 7,749 kernel launches, GPU busy 71%. The changes: training logging as masked sums read once per epoch; `solve_ex`/`inv_ex` and a closed-form Rodrigues rotation in place of `linalg` error checks and `matrix_exp`; per-step constants built on the device and cached; `torch._assert_async` checks; contrast mixing vectorised over study and plane groups and image augmentations drawing their parameters up front, in the original order, with `nonzero_static`; GIN normalising per series index instead of `unique`. Parity: same weights on 64 validation studies at ×4, mean difference 7e-8 (max 6e-6) on the ConvNeXt student, 2.5e-6 (max 7e-4) after merging main's model split, ResNet-50 student 3.6e-6 (max 4e-4); the first 12 losses stay within the 0.001 spread of two old-code runs. Speed: 275–276 ms to 258 ms per step (260 ms after the merge, −6%), synchronisations 281 to 50, GPU busy 84%. About 50 sites remain: kornia inside affine, gamma, contrast and noise (13); the slice-mask `nonzero` in `_encode_crops` and `_augment_images` (5); feature-statistics augmentation (4); `_log_views`, the masked target loss, anatomy centre checks and soft-tissue routing (about 15); the count reads of contrast mixing and augmentation; Lightning's step logging. Sites are listed with `torch.cuda.set_sync_debug_mode("warn")` over a few real steps.
- **What gave the gain, in order:** only the middle channel of each canvas is sent to the GPU
  when `slice_channels` is `repeated_middle` (33 MB instead of 99 MB per batch; 270 to 246 ms);
  `model.channels_last: true` (304 to 291 ms; on ResNet-18 the changing image count per step
  caused none of the cuDNN setup delays recorded above for ReXNet-100); the batch's nested
  sampling metadata is carried in an opaque container, so the pin thread and the device transfer
  no longer walk about 5,500 Python objects per batch; tokens, routing weights and position
  priors are computed once per encoded batch instead of two to seven times; the epoch metric
  accumulator stacks at the end of the epoch instead of multiplying and adding per call;
  gradient diagnostics every tenth step; bicubic resampling only for turned, valid crops.
- **Parity:** seed 2's weights give the same ×4 validation predictions on the old and the new
  code: mean absolute difference 0.000004, macro 0.927078 on both, where two runs of the old
  code differ from the saved predictions by 0.00046 on average (BF16 and the rounded crop
  shift). The first eight training losses follow the same trajectory.
- **Tried and reverted:** skipping the per-study frame head when the anatomical frame is on. Its
  global rotation reaches the across axis the anatomical frame keeps, so the predictions moved
  (mean difference 0.0010, macro −0.0001).
- **Not allowed:** fused AdamW, which Lightning refuses together with gradient clipping.
- **Left on the table:** the first look is about 65 ms of the step and the image augmentation
  about 35 ms; both are part of the design.
