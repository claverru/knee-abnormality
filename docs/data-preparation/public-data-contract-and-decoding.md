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

**The host tried to replace these files on 2026-09-17** and the attempt may have
failed: every competition file now carries a 2026-09-17 22:14 UTC creation date,
several download links broke, and the host answered on 2026-09-18 that "we
attempted to update the dataset recently to fix the corrupted DICOMs, but it
looks like the update may have failed"
([thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741916)).
Our copy predates that upload. Once downloads work again, re-fetch these two
studies and recheck them: `KNOWN_BAD_SERIES` and `KNOWN_BAD_INSTANCES` in
`src/knee_abnormality/data/preprocessing/exclusions.py` name exactly these files,
and a successful repair would make those exclusions discard usable slices.

For physical ordering, adjacent-triplet selection and cache tradeoffs, see the
[public geometry evidence](../research/public-research.md#geometry-sampling-encoders-and-losses)
and [runtime lessons](../research/public-research.md#data-access-and-runtime-lessons).
