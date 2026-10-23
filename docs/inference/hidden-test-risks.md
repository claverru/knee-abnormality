# Test-time risks

Things that could go wrong on the hidden test set (about 1,300 studies, no reports) but not on
training data or the three local test examples; it was the checklist for a Kaggle submission that
failed or scored unexpectedly low. Each entry says whether it is measured, inferred from code, or
external evidence.

## Submission failures

Hidden reruns return no logs or traceback: the one failed rerun, submission `56224993`, reported
only an unhandled error ([record](submissions/early-submission-details.md)).
A submission therefore shows whether a crash happened, never why, and warnings stay invisible.
About 25 submissions have completed on the full hidden test with the crash paths below in place,
so the two crash cases do not occur there with that code; the 2026-09-17 filtering and ranking
code also completed ([submission](submissions/early-submission-details.md)).

- **A study with no usable series stops the whole run** (inferred from code). Per-series decode
  and preparation failures are only logged, but `KneeStudyDataset.__getitem__`
  (`src/knee_abnormality/data/dataset.py`) raises `RuntimeError` when no series of a study
  prepares, and there is no fallback prediction. No submission hit it (S190's hidden failure was GPU
  memory), so the hidden test did not trigger it with the code used. Possible guard: predict training prevalences
  for such a study and log it.
- **Contrast flag combinations never seen in training** (inferred from code; the host warns).
  The [data page](submission-requirements-and-output-contract.md#inputs-and-output-contract)
  says `Fluid_Sensitive` and `Fat_Suppression` are not guaranteed to coincide, but training and
  the local test examples only contain (0,0) and (1,1). `build_series_index`
  (`src/knee_abnormality/data/sources.py`) ignores other combinations with a warning, so those
  series silently disappear; a study with only such series is missing from the series index and
  fails with a `KeyError`. Possible guard: map every combination other than (1,1) to the (0,0)
  slot, which matches the host's training convention (fat-suppressed T1 and T2 without fat
  suppression are both labelled (0,0), [census](../metadata/series-filtering-census/census-and-categories.md)).
- **Study membership mismatch** (code). Inference raises `ValueError` when the studies in
  `test_series.csv` differ from `sample_submission.csv` (`prediction_studies`).
- **Unverified encodings** (local evidence). The data page lists uncompressed little-endian, JPEG
  Lossless and JPEG 2000; successful JPEG-LS decoding was never verified locally and compressed
  throughput was never measured
  ([DICOM descriptor experiment](../data-preparation/dicom-descriptor-experiment.md)). A series
  that fails to decode is dropped from its study; a study where all fail hits the first risk.

## Silent input changes

- **Missing or altered headers** (external evidence). In the 2025 aneurysm competition the hidden
  test headers had deleted fields, and in the 2024 lumbar competition a `SpacingBetweenSlices`
  rule that worked on training broke on the hidden test
  ([research](../research/data-and-validation/series-filtering-signals/INDEX.md)). The tie-break ranking treats a missing
  tag as a tie and the series filters keep series whose tags are missing; geometry tags remain
  required.
- **Non-knee coil filter drops a knee series** (measured on training; very unlikely). A series
  whose receive-coil name contains a non-knee body-part word (ankle, foot, wrist, hand, elbow,
  shoulder, hip, head, neck, breast) and not "knee" is dropped when its study mixes coil names.
  Of 38 training coil names only `SENSE_ANKLE_8_AC` matches; only 8 of 4,410 studies mix coil
  names; combined names such as `QD Knee Foot` are exempt. A wrong drop needs a coil combination
  never seen in training and removes only part of one study.
- **Other-knee filter drops a real series** (measured on training). When series centres spread
  more than 150 mm in patient x, a side with a single series is dropped. In training this happens
  once, and all 8 studies above the threshold are true other-knee cases.
- **Instance guard removes images** (measured on training: none removed). Instances that do not
  match their series' majority shape, pixel spacing and orientation are dropped; a mixed test
  series loses its minority instances, which is intended for scouts inside a series.
- **Localizer or scout series** (measured: none in training). The host's series table appears to
  exclude them. A separate localizer series in the test would be sampled like any other series;
  only scouts inside a series are guarded. The header guard waits in
  [endgame backlog](../backlog/untried-ideas.md#hidden-test-guards).
- **Rejected study centres** (measured: 4% of training studies). They fall back to local
  sampling; expect a similar share in the test. A robust centre did not score better
  ([history](../history/robust-study-centre.md)).
- **Unusual patient positioning** (measured: one training study). Plane labels follow scanner
  axes, so a patient positioned sideways gets anatomical sagittals in the coronal slot and
  mirrored along the wrong axis under reflection (training only).

## Numerics and scores

- **T4 precision** (measured). Kaggle T4s lack BF16, so the backbone runs in fixed FP16 chunks
  ([export recipe](justfile-recipes.md#export-and-submit-a-checkpoint)); local BF16 and
  Kaggle FP16 predictions differ slightly. Casting a whole model with `model.half()`, or only its
  Conv and Linear layers, failed in the side head (`model.py` side_head: FP32 input in an
  autocast-off region); only the backbone in FP16 (`model.backbone.half()`) ran, as in S191 (the
  [×8 per-label bundle](../validation/results/teacher-student/2026-10-19-labels-v3-final-ensembles-and-students.md#board-results)).
- **Batch composition** (measured). Under BF16, the same study predicted at batch size 1 and 8
  differs by up to 0.006 in probability
  ([census inference check](../metadata/series-filtering-census/flags-and-inference-effect.md)).
- **Ensemble members share one loader** (inferred from code). A multi-checkpoint bundle decodes
  and samples each study once and runs every member on the same tensors, so the whole `data`
  section belongs to the bundle rather than to the member. A member trained under a different
  `sampling_sigma_fraction`, `reference_field_mm`, `image_size`, `series_types` or `study_slice_total`
  would run under another model's preprocessing and degrade **silently** — no crash, just a lower
  score indistinguishable from ordinary noise. Export validates that every member's `data` config
  matches and refuses a mismatch, so the guard is the only thing between a mixed bundle and an
  invisible regression; check it first if an ensemble scores below its members. Members may still
  differ freely in their `model` section, including `backbone`, because the spatial token mapping
  sits downstream of the CNN.
- **Supporting mixed loading is affordable but unbuilt** (measured). Sampler-only differences
  would need the union of the members' selected slices decoded once, and geometry differences one
  resample per geometry over the same decoded frames: decoding costs 15.54 s per 100 studies
  against 8.02 s for resample and crop
  ([profile](profiling/2026-09-18-kaggle-x4-profile-after-index-changes.md#cold-loading-stages-100-studies-with-cpu)),
  so a second geometry is about a third of a second pass, not a doubling. Neither is implemented,
  which is why the guard above is a hard refusal rather than a fallback.
- **Public score noise is not a bug** (measured). The same recipe trained four times scored
  0.928–0.934 publicly ([seed controls](submissions/early-submission-details.md)),
  so a public drop below about 0.006 needs no failure explanation.
