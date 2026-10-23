# Public data contract and decoding

The [local metadata audit](../metadata/INDEX.md) records verified CSV schemas,
identifiers, series profiles, and acquisition commands. Label coverage and
class distributions are recorded in [supervision](../modeling/supervision.md#available-supervision).

The [official data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
defines study-level targets and series-level metadata. DICOMs have varying
intensities, orientations and resolutions, multiple transfer syntaxes including
JPEG Lossless/JPEG 2000, and only an allowlisted set of 86 metadata tags. Typical
series contain 20–45 slices, with a longer tail. Test files expose three example
studies; scoring replaces them with about 1,300 studies and their series. Target
prevalence is not guaranteed to match across train/public/private datasets.

A public metadata check found `Fluid_Sensitive == Fat_Suppression` for every
training series, while the host says these represent distinct characteristics
and need not remain equal in other data. Preserve both fields in series
selection and missing-slot handling.
[Metadata clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737312).

Two independently reported failures concern truncated pixel payloads, with only
half the bytes implied by the headers. The [original thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737163)
identifies the following slice paths under `train_series/`:

```text
1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680/1.2.826.0.1.3680043.8.498.31160785238781353848727311763596115703/1.2.826.0.1.3680043.8.498.66460523372602171398246641420761543626.dcm
1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035/1.2.826.0.1.3680043.8.498.39396636671446532796538574315802928348/1.2.826.0.1.3680043.8.498.41947538458330624797291612781685746320.dcm
```

A [later report](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738708)
names the same studies. The local acquisition audit on 2026-09-10 confirmed
both failures with pydicom 3.0.2: the first listed file contains 173,056 pixel
bytes versus 346,112 declared; the second contains 409,600 versus 819,200.
Both files match their archive CRCs, supporting an upstream payload problem
rather than a transfer error. These are two confirmed cases, not an exhaustive
blacklist. Preserve the raw files and handle these failures during preprocessing.
Log decode failures by UID, distinguish missing codecs from malformed payloads,
and explicitly handle missing slices/series while preserving one prediction row
per test study.

**Both files are repaired upstream, verified on 2026-09-20.** The host tried to
replace them on 2026-09-17, several download links broke, and the host said on
2026-09-18 that the update "may have failed"
([thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741916)).
Single-file downloads work again, and re-fetching the two instances returns
complete payloads: 173,056 → **346,112** bytes against 346,112 declared, and
409,600 → **819,200** against 819,200. Both now decode with pydicom 3.0.2, to
(416, 416) and (640, 640) `uint16`. The re-uploaded files carry 2026-09-19
timestamps; a seeded eight-file random sample from elsewhere in the train split
came back **byte-identical** to our 2026-09-10 copy, which is evidence that the
re-upload changed only the repaired files, not proof over 819,640 files.

**Taken on 2026-09-20.** Both repaired instances replaced the truncated files in
`data/raw`, and the guard that skipped them is gone: `exclusions.py` and its two
call sites are deleted, so `index_series` no longer drops the instance by UID and
`build_series_index` no longer subtracts one from those series' image counts.
The two series now index 38 and 36 slices against 38 and 36 on disk, matching
`train_series_image_counts.parquet`, and every slice of both decodes through the
production reader, the repaired ones into finite (224, 224) images. Two slices
return to training out of 819,078, so the effect on a model is not measurable;
the point is that the data and the code now say the same thing.

Removing `exclusions.py` bumps the preprocessing identity — `sources_sha256`
`f9d05ad2` → `65a6123d` — so it applies to fresh runs only. A truncated file in
the hidden test set is still handled, by the generic decode-failure path that
drops the candidate series and records the failure, not by a UID list.

For physical ordering, adjacent-triplet selection and cache tradeoffs, see the
[public geometry evidence](../research/competition-and-prior-art/public-research.md#geometry-sampling-encoders-and-losses)
and [runtime lessons](../research/competition-and-prior-art/public-research.md#data-access-and-runtime-lessons).
