# Possible repeated studies across folds

The bounded 2026-09-20 audit found no confirmed repeated patients or copied
acquisitions. The exhaustive 2026-09-22 pixel scan later found one copied study
pair across folds 1 and 2; see [cross-study repeat](repeated-series.md#cross-study-repeat).
The 2026-09-20 audit investigated indirect links after finding that all 4,407 sampled
training-study headers had distinct PatientID and FrameOfReferenceUID values.
Distinct anonymized identifiers do not establish longitudinal patient separation.
No fold assignments, training settings or datasets were changed.

## Candidate retrieval and image comparison

The [audit script](../../scripts/adhoc/audit_study_overlap.py) reads the existing
representative-header paths, normalized reports and fixed fold membership. It
compares exact representative slice geometry: position, orientation, spacing and
matrix dimensions. It also retrieves six nearest report neighbors per study using
character TF-IDF, retaining cosine similarity at least 0.94 and at least 200
normalized characters. Existing exact report groups are excluded from report
candidates because the split already keeps those together. Unicode/punctuation
normalization preserves word order, laterality, negation and numbers.

Artifacts: `outputs/analysis/s/study-overlap-20260920/`. The manifest preserves input
hashes and the executed source snapshot. A later source snapshot and separate
manifest preserve the supplementary approximate-pixel analysis.

Reproduction (CPU only, fresh output directory; folds default to `all_5fold_v3`,
`--pixel-pairs` sets the 40-pair image shortlist):

```bash
env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python -m scripts.adhoc.audit_study_overlap --output-dir outputs/analysis/study-overlap-new
```

The shortlist decodes one complete representative series per study and gives
cross-fold pairs priority within each evidence tier. Exact hashes cover decoded
pixel arrays and dimensions only, ignoring identifiers and compression containers.

| Check | Result |
|---|---:|
| Representative headers read successfully | 4,407 |
| Repeated PatientID, FrameOfReferenceUID or exact geometry signatures | 0 |
| Near-report candidate pairs | 885 |
| Candidate pairs crossing folds | 701 |
| Candidate pairs with the same recorded scanner model | 300 |
| Candidate pairs with identical Unicode/punctuation-normalized reports | 11 |
| Highest-priority pairs checked for exact image matches | 40 |
| Distinct studies in that image shortlist | 47 |
| DICOM images decoded from their representative series | 1,202 |
| Decode errors | 0 |
| Shortlisted pairs sharing a nonconstant decoded slice | 0 |

Candidate counts are not duplicate-patient counts. The retrieval is bounded, not
all-pairs exhaustive. One sampled series may have a different contrast or plane
between studies; a negative pixel comparison cannot clear their other series.
Exact fingerprints detect identical decoded integer arrays and dimensions, not
resampled images, intensity-adjusted copies, or separately acquired follow-ups.

A supplementary approximate comparison resized those same slices to 32x32,
clipped intensities at the first and ninety-ninth percentiles, and compared
centered unit-length pixel vectors. Mutual nearest slices had a maximum cosine
of 0.9031; the strongest pair's top-five mean was 0.8970. No compared slice pair
reached 0.99. These are uncalibrated similarity measurements, not thresholds
establishing different patients. No registration, orientation alignment or
physical-scale correction was performed. Results are in
`approximate_pixel_pairs.json`, with `approximate_manifest.json` and its source
snapshot. This stage did not reveal an obvious near-identical image copy.

## Bilateral-report clue

Three candidate pairs had the same scanner model and long normalized reports
differing only in the English words left/right. Supplemental header reads examined
all 43 series across their five studies; the saved tables are
`bilateral_candidate_headers.json` and `bilateral_candidate_pairs.json`.
One pair had conflicting recorded sex. All three reused rather than complemented
their series-number ranges, and none shared exact imaging-frequency values.
One pair even had left laterality tags on both sides despite different report
sides. These findings weaken a shared-session interpretation; they do not prove
different patients, since fields may be wrong and acquisitions can change.

Near-identical reports can reflect institutional templates. They should retrieve
candidates for corroboration, never directly merge patient groups.

## Further evidence that could help

The remaining investigation is tracked in
[big bets](../backlog/untried-ideas.md#inputs-beyond-the-2d-recipe).
Anatomical matching across physically ordered volumes could detect repeat scans
whose pixels differ: register the same knee and compare stable bone shape,
trabecular texture, surgical hardware and other distinctive structural features.
Require agreement across multiple slices and preferably another plane. Similar
diagnoses, scanner settings or generic knee shape are inadequate by themselves.
Opposite knees need separate treatment: mirroring normal anatomy cannot establish
that they belong to the same patient.

Acquisition fingerprints combining frequency, coil, sequence inventory, series
numbering and geometry could prioritize same-session or split-bilateral cases,
but none of those fields alone establishes a patient relationship. Calibration
against unrelated studies from the same scanner and protocol is essential.

Confirmed substantial image overlap would justify grouping the involved studies.
Unresolved anatomical or metadata resemblance remains a review candidate. This
audit provides no basis for changing the current fold membership; the copied pair
found later is handled in [repeated series](repeated-series.md#cross-study-repeat).
