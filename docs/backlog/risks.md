# Test-time risks

Things that could go wrong on the hidden test set (about 1,300 studies, no reports) but not on
training data or the three local test examples. Check this list first when a Kaggle submission
fails or scores unexpectedly low. Add a risk whenever a new rule depends on data the hidden test
may present differently; remove it once a guard or a submission settles it. Each entry says
whether it is measured, inferred from code, or external evidence.

## Submission failures

Hidden reruns return no logs or traceback: the one failed rerun, submission `56224993`, reported
only an unhandled error ([record](../inference/submissions/2026-09-14-original-1c-comparison-submission.md)).
A submission therefore shows whether a crash happened, never why, and warnings stay invisible.
About 25 submissions have completed on the full hidden test with the crash paths below in place,
so the two crash cases do not occur there with that code; the 2026-09-17 filtering and ranking
code also completed ([submission](../inference/submissions/2026-09-17-ranked-filtered-submission.md)).

- **A study with no usable series stops the whole run** (inferred from code). Per-series decode
  and preparation failures are only logged, but `KneeStudyDataset.__getitem__`
  (`src/knee_abnormality/data/dataset.py`) raises `RuntimeError` when no series of a study
  prepares, and there is no fallback prediction. Every submission so far completed, so the hidden
  test has not triggered it with the code used then. Possible guard: predict training prevalences
  for such a study and log it.
- **Contrast flag combinations never seen in training** (inferred from code; the host warns).
  The [data page](../inference/submission-requirements-and-output-contract.md#inputs-and-output-contract)
  says `Fluid_Sensitive` and `Fat_Suppression` are not guaranteed to coincide, but training and
  the local test examples only contain (0,0) and (1,1). `build_series_index`
  (`src/knee_abnormality/data/sources.py`) ignores other combinations with a warning, so those
  series silently disappear; a study with only such series is missing from the series index and
  fails with a `KeyError`. Possible guard: map every combination other than (1,1) to the (0,0)
  slot, which matches the host's training convention (fat-suppressed T1 and T2 without fat
  suppression are both labelled (0,0), [census](../metadata/series-filtering-census.md#5-contrast-measured)).
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
  ([research](../research/series-filtering-signals.md)). The tie-break ranking treats a missing
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
  only scouts inside a series are guarded.
- **Rejected study centres** (measured: 4% of training studies). They fall back to local
  sampling; expect a similar share in the test. A robust centre did not score better
  ([big bets](big-bets.md#robust-study-center)).
- **Robust centre drops knee series** (inferred from code; applies only when
  `data.robust_center_outlier_mm` is set). A rejected test centre is refitted without its outlier
  series. In a refitted study, a series whose slab does not contain the new centre leaves the
  input whenever another series in its slot contains it.
  - If most of a study's series lie off the knee, the centre could settle on the wrong anatomy,
    and knee series would then be the ones dropped.
  - Training shows the scale (measured, 20 mm): 58 series leave the input in 130 rescued studies.
    This census did not check their anatomy. In an earlier review, 16% of the series the fit
    excluded were not the target knee
    ([census](../metadata/series-filtering-census.md#flag-precision-and-recall)).
- **Unusual patient positioning** (measured: one training study). Plane labels follow scanner
  axes, so a patient positioned sideways gets anatomical sagittals in the coronal slot and
  mirrored along the wrong axis under reflection (training only).
- **Training-only exclusions** (code). `KNOWN_BAD_SERIES` and `KNOWN_BAD_INSTANCES` in
  `src/knee_abnormality/data/sources.py` list training UIDs only; similar test files are handled
  by the decode-failure path above.

## Numerics and scores

- **T4 precision** (measured). Kaggle T4s lack BF16, so the backbone runs in fixed FP16 chunks
  ([fast submission preparation](../inference/fast-submission-preparation.md)); local BF16 and
  Kaggle FP16 predictions differ slightly.
- **Batch composition** (measured). Under BF16, the same study predicted at batch size 1 and 8
  differs by up to 0.006 in probability
  ([census inference check](../metadata/series-filtering-census.md#inference-effect-fold-0)).
- **Public score noise is not a bug** (measured). The same recipe trained four times scored
  0.928–0.934 publicly ([seed controls](../inference/submissions/2026-09-17-seed-control-submissions.md)),
  so a public drop below about 0.006 needs no failure explanation.
