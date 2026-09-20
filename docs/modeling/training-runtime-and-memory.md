# Training runtime and memory

## ReXNet-100 runtime

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
bounded workloads. Exact configurations and measurements remain in
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
the deferred [280 px test](../backlog/endgame.md#bigger-images) at 1.56×.
The restored baseline, like `00-baseline`, uses 16 workers and `prefetch_factor: 2`, and has no
separate validation worker count. The crop-era `trainer.validation_num_workers` field was
removed. A larger-input run should revisit both settings.

The first [crop-selection run](../validation/results/2026-09-15-crop-selection-result.md) was
killed by the Linux out-of-memory killer during epoch 8. WSL got 30 GB of the host's
64 GB by default; after the
[per-label crops run](../validation/results/2026-09-16-per-label-crops-result.md) was killed the
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
