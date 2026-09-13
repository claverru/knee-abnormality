# Future work

Ideas to explore. Follow the [work-tracking rules](../AGENTS.md#work-tracking)
when moving an idea into active work.

## Fast baseline experiments

These are the remaining low-cost experiments for the
[current 224 px baseline](../configs/baseline.yaml). All are unvalidated
hypotheses grounded in the measured [baseline review](baseline-review.md),
public evidence and broader competition research.

### Evaluation rule

Use **fold 0** of the [current validation protocol](validation.md#project-validation-protocol)
for iterations until near the end of the competition, as selected on 2026-09-11.
Keep its frozen training and validation memberships fixed. Compare full-fold macro
AUC and per-target results against frozen evaluation labels thresholded at
`>= 0.5`, retaining the separate gold diagnostic. Preserve seed, model size,
training budget and validation density unless that setting is the experimental
factor. Training targets remain soft unless explicitly testing supervision.
The optional export parity utility is not required for each run or submission.

### Deferred report-label supervision

Pooling and augmentation ablations were dropped from the proposed experiments.

Keep **soft versus hard report-derived training targets** as a future experiment.
Retain the original gold labels and frozen validation targets. Investigate
uncertainty weighting separately; report-derived probabilities need not be
calibrated confidence. This would require a small loss or label-preparation
change with similar model compute and memory. No gain is established and no
implementation is scheduled. See the [public label audit](public-label-audit.md)
and [supervision evidence](public-research.md#report-derived-supervision).

## Sampling, physical coverage and resolution

Proposed in the 2026-09-11 discussion and deferred for future work. This section
records the remaining proposed implementation and comparisons. The 224-pixel,
140-mm crop comparison is complete; see its
[local and public results](validation.md#2026-09-12-140-mm-crop-result).
Continue iterating on fold 0. CoAtNet and label-supervision experiments are
listed separately in this document; pooling and augmentation changes are excluded.

The shared-center Gaussian and overlap idea is now in
[ongoing work](ongoing.md#physical-sampling-plan). Its completed
[metadata analysis](physical-sampling.md) supersedes the earlier curve-only
proposal; the original file-count inventory is preserved there.

### 2. Increase shared in-plane spacing from 0.67 to 0.72 mm

Keep 224 × 224 inputs and use **0.72 mm in every plane**, giving a centered
field of view of **161.28 × 161.28 mm**, compared with 150.08 × 150.08 mm at
0.67 mm. This changes physical coverage while keeping the CNN input dimensions
and slice budget fixed. The tradeoff is slightly less spatial detail per mm.

Do not introduce separate spacing values or switches for individual planes.
A read-only sample of 768 series, 128 per plane–contrast combination, found a
median acquired field of view of approximately 160 × 160 mm in all six groups.
That larger sample did not support the earlier 48-header sample's suggestion
that sagittal acquisitions generally need a larger field of view.

Sampling method: sort the count inventory by study and series UID, sample 128
series within each type using Polars seed 42, and read the lexicographically
first DICOM header per selected series. All 768 headers supplied valid geometry.
Extent is rows × row spacing and columns × column spacing. Native image extent
does not establish useful anatomy coverage; inspect crop boundaries before
interpreting an AUC change as evidence that anatomy was previously excluded.

### 3. Compare 336-pixel training with 224 pixels

Train ReXNet-200 with **336 × 336 inputs**, holding the selected physical field
of view fixed. Set shared spacing to `field_of_view_mm / 336`: approximately
0.416667 mm for the preferred 140-mm field of view. Use the same image size
and spacing for validation and test prediction.

This isolates finer spatial sampling from physical crop size. Increasing
`image_size` while leaving `spacing_mm` unchanged would also widen the crop.
Hold the backbone, labels, slice allocation, sampling and effective training
batch size fixed; record any physical-batch or accumulation adjustment needed
to fit memory. A 336-pixel input has 2.25 times the pixel area of a 224-pixel
input, but measure actual runtime and peak VRAM rather than assuming they scale
by that factor.

The public [single-model Raptor notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model)
provided the lead for comparison 3 and the completed 140-mm crop comparison: its inspected source uses a 140-mm
crop, 336-pixel preparation and a 384-pixel CoAtNet encoder. Kaggle metadata
checked on 2026-09-11 records a best public score of 0.924. Its different
backbone, labels, training membership and sampling prevent attributing that
score to either resolution or crop size. The separate project crop comparison
now has measured results; the resolution comparison remains unvalidated.

### Original proposed comparison sequence

1. The original Gaussian-only comparison has been superseded by the active
   [physical sampling work](physical-sampling.md).
2. Compare 0.67 versus 0.72 mm while holding the selected sampling recipe fixed.
3. Compare 224 versus 336 pixels while preserving the selected physical field
   of view through the corresponding spacing adjustment.

The 224-pixel, 140-mm comparison completed ahead of this proposed sequence;
use its [recorded result](validation.md#2026-09-12-140-mm-crop-result) when
choosing the reference geometry for subsequent experiments.

Keep fold-0 membership, seed 42, ReXNet-200, training budget, labels, slice
limits and validation image multiplier fixed except for the factor under
comparison. The completed spatial-token run on `all_5fold_v3` supplies the current
reference; no baseline rerun is scheduled.
Retain full 904-study validation predictions, macro/per-target AUC,
gold diagnostics, runtime and peak VRAM. Keep validation and test inference at
2× image limits, with inference batch size 2, following the
[completed density comparison](inference-efficiency.md#inference-image-count-and-memory).
Do not mix a change in inference image multiplier into these comparisons.
Gains from the remaining sampling, wider-coverage and resolution proposals
remain unverified until measured;
no additional folds, ensemble, or routine submission parity pass is required.

## Deferred preparation experiments

- **Exact percentile selection:** reconsider using the completed
  [DICOM integration](data-preparation.md#production-dicom-integration) as the baseline. The earlier combined
  profile spent 2.617 aggregate worker-seconds here; current production already
  gathers valid resampled float32 values into contiguous scratch and partitions
  all unique floor/ceil ranks together. Compare a native multi-selection loop
  with NumPy partition including scratch construction and rank interpolation.
  If still material, investigate exact coarse-histogram assistance: locate the
  bins containing the requested ranks, then select exactly within those bins
  using cumulative offsets. Keep the population acquisition-wide over all
  successfully decoded complete groups. Do not repeat single-rank partition,
  full sorting or per-slice bounds; approximate quantiles are outside the
  equivalent-input contract.
  [NumPy partition reference](https://numpy.org/doc/2.0/reference/generated/numpy.ndarray.partition.html).
- **GPU preprocessing boundary:** consider only after remeasuring DICOM and
  percentile work. Compare transfer of compact stored-pixel regions or float
  crops with current uint8 output at the same logical model batch. Include IPC,
  pinning, host-to-device bytes, percentile/CUDA synchronization, model
  contention and peak VRAM. A resize-kernel timing is insufficient; stop if the
  complete fixed-workload pipeline does not improve. Keep experimental
  candidates in the existing
  [preprocessing benchmark](../scripts/adhoc/benchmark_dataloader_optimizations.py).
- **Learned per-series intensity bounds:** revisit after the fixed-clipping
  baseline. A shared predictor could estimate bounds for each acquisition,
  including unseen series. This needs unclipped floating-point input before
  the learned operation; uint8 produced by fixed clipping has already lost
  information. Parameters indexed by training series UID would not generalize
  to unseen acquisitions.
- **Prepared pixel or feature caches:** consider pixel caching only if measured
  loading throughput limits training after worker/prefetch tuning. Record
  source, preprocessing and code identities, preserve enough slice coverage,
  and budget disk space. Feature caching is useful only while the encoder is
  frozen and also requires checkpoint and fold-lineage identities.

## Public-evidence experiment candidates

Recorded on 2026-09-09 after the
[discussion/notebook investigation](public-research.md). These remain candidates
after the [sampling, physical coverage and resolution comparisons](#sampling-physical-coverage-and-resolution);
none is a demonstrated project improvement.

- **Label handling:** using the [audited public sources](public-label-audit.md),
  compare hard versus soft targets and explicit uncertainty weights. Isolate
  targeted Synovitis repair
  from all-target imputation. Preserve an evaluation endpoint not used in
  label generation or repair.
  [Evidence](public-research.md#report-derived-supervision).
- **Slice coverage:** compare sampled-window coverage with fixed split and
  labels. Compare smaller inference budgets and greater sagittal allocation
  separately. The [0.924 Raptor notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model)
  uses a 64-slice inference stack and 42 overlapping triplets; our 2× inference
  allows 192 raw slices and 64 non-overlapping triplets. Its result does not
  establish a need for more slices. Record AUC and runtime. Shared-center
  Gaussian placement is now in the active
  [physical sampling work](physical-sampling.md); physical crop and output
  resolution remain in
  [sampling, physical coverage and resolution](#sampling-physical-coverage-and-resolution).
  [Evidence](public-research.md#geometry-sampling-encoders-and-losses).
  [Cross-competition comparison design](research/experiment-priorities.md#pooling-and-coverage).
- **CoAtNet backbone:** compare `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`
  against ReXNet-200 after the planned geometry comparisons. Retain fold 0,
  Steven v4 labels, triplet sampling, target-specific attention pooling and
  the training budget. Use a supported common input resolution for the two
  backbones; isolate any required resolution change from the encoder comparison.
  Record full-fold and per-target AUC, runtime and peak VRAM. The public
  [single-model Raptor inference notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model)
  has a best public score of **0.924**, verified through Kaggle metadata on
  2026-09-11, but its different training membership, unpublished label table
  and image pipeline confound attribution to CoAtNet. Its
  [training source](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model)
  is an implementation reference, not a reproduced project result. Keep this
  as a single-model experiment; additional folds and ensembling remain deferred.
- **Other complementary model families:** compare an understandable DINO baseline
  or a medical-pretrained branch with the selected baseline and target-specific
  pooling.
  Measure independent validation gains and incremental ensemble cost before
  scaling the encoder or importing a large public ensemble.
  [Source implementations](public-research.md#public-code-worth-inspecting).
- **Cheap inference diversity:** evaluate a few complementary predictions
  against top-epoch weight averaging or a small amount of TTA, recording total
  hidden-test-like runtime. Avoid extended public-LB blend-weight sweeps when
  local evaluation cannot resolve the differences.
  [Evidence](public-research.md#validation-and-ensembling).

## Additional MRI research candidates

Recorded on 2026-09-10 after the [broader research](research/README.md).
These extend the candidates above and remain unvalidated.

- **Efficient encoder adaptation:** compare a frozen head, lightweight adapters,
  and full fine-tuning with the same encoder, labels, windows, and seeds before
  increasing backbone size. Measure actual forward-pass cost as well as
  trainable parameters.
  [Paper evidence and cache limitations](research/pretraining-and-external-data.md#adapting-2d-foundation-models-to-volumes).
- **Anatomy-guided pooling and external knee pretraining:** investigate only
  after the baseline reveals a representation or localization limitation.
  Retain global image context,
  evaluate predicted crops without reference correction, and establish
  eligibility and label mappings before acquiring external data.
  [Resource comparison](research/pretraining-and-external-data.md#external-resources-and-label-compatibility)
  and [implementation references](research/code-and-notebooks.md).

## Near the end of the competition

Keep additional-fold training and ensembling deferred while improving the
single-fold recipe. Once that recipe is sufficiently settled, consider the
remaining fixed folds, complete 4,407-study OOF evaluation with the separate
58-study gold diagnostic, and any final ensemble within the inference budget.
This is not a prerequisite for the current experiments.
