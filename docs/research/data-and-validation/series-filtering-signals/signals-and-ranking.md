# Series-filtering signals: ranking and cost

Recorded on 2026-09-17 for series filtering and cleaning. The coil-name rule, the side rule for
a stray other-knee series and the within-series layout guard were implemented on 2026-09-17
([implemented rule](../../../metadata/series-filtering-census/implemented-rule.md#implemented-rule-2026-09-17)); the
other signals here were not implemented, and no rule was validated against disease AUC. Details
per case are in [findings by case](findings-by-case.md); literature and tool rules in
[external evidence](external-evidence.md); the rules considered in
[candidate rules and gaps](candidate-rules-and-gaps.md).

**The best signals here are cheap and header-based, and they differ by target case.**

| Target case | Best signal | Support | Evidence strength |
|---|---|---|---|
| Contralateral knee | Sign of each series' physical field-of-view center X (the patient left–right axis) | `SeriesNumber` blocks | Strong here |
| Non-knee anatomy | A `ReceiveCoilName` anatomy word that differs from the rest of the study | Geometric inconsistency | Strong in one study only |
| Plane and contrast labels | Verification, not filtering | — | See below |

- **Contralateral knee.** The sign of X separates the two knees of every bilateral study found so far.
- **Non-knee anatomy.** The coil-name rule marks exactly the five ankle series of the only confirmed case.
- **Plane and contrast labels** are consistent with headers almost everywhere:
  - The competition plane equals the dominant orientation axis for every series.
  - Header fat-suppression evidence (scan options, description, inversion time) backs the label for 99.0% of fat-suppressed series.
- **Unreliable here:** `BodyPartExamined`, `Laterality`, `ContrastBolusAgent` and `SeriesDescription` keyword rules. The literature and this dataset agree on this.
- **Absent:** localizers and derived images, apart from 53 Philips reformats. Rules for them are defensive, for the hidden test.

Evidence labels used below:

- **Measured here:** scratch counts on the series census table `outputs/analysis/s/series-filtering-20260917/census_series.parquet`. The table has 24,371 training and 15 example-test series, with one header (the middle file) and intensity features from three sampled slices per series. These are counts, not validated rules. The census itself is recorded in [series-filtering census](../../../metadata/series-filtering-census/INDEX.md).
- **Reported:** a number from the cited source, for its own task and data.
- **Claim:** an unquantified statement in a source.
- **Inference:** our reasoning, not measured.

## Cost tiers and tag availability

Rules must run identically in the Kaggle notebook. Rank them by cost:

| Tier | Meaning | Cost at inference |
|---|---|---|
| **H0** | Header fields the loader already keeps: orientation, position, pixel spacing, rows and columns, instance number, frame count | None |
| **H1** | Any other tag of the same header. `read_header` in `src/knee_abnormality/data/preprocessing/dicom.py` already parses every file's full pre-pixel dataset with `read_partial`, and the [tie-break design](../../../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#implementation-2026-09-17) makes the same point. | Code only, no extra I/O |
| **I** | Cheap statistics on a few decoded slices: foreground extent, intensity quantiles | CPU |
| **M** | A small image model, possibly with external weights | T4 time and a licence check |

Coverage below counts **non-empty values** across all 24,371 training series (measured here). The
[tie-break tag inventory](../../../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#why-the-series-tie)
counted element presence in 5,935 repeated-group series. That overstates two tags:

- `Laterality` is present in 78% of those series but non-empty in only 49.3% of all series.
- `ContrastBolusAgent` is present in about 45% but non-empty in only 17 series.

| Tag | Non-empty | Notes |
|---|---:|---|
| `ImageOrientationPatient`, `ImagePositionPatient`, `PixelSpacing`, rows/columns, `SeriesNumber`, `ImageType`, `Manufacturer` | 100% | H0/H1 |
| `RepetitionTime`, `EchoTime`, `ScanningSequence`, `SequenceVariant`, `MRAcquisitionType`, `FlipAngle`, `EchoTrainLength`, `PatientPosition` | 95.1% | |
| `SeriesDescription` | 94.1% | Includes 2,879 `DummySeriesDesc!`, so only **82.3%** carry text |
| `BodyPartExamined` | 90.5% | See [non-knee anatomy](findings-by-case.md#non-knee-anatomy) |
| `ScanOptions` | 85.9% | |
| `ReceiveCoilName` | 76.7% | 3,367 studies |
| `Laterality` | 49.3% | |
| `SequenceName` | 45.9% | |
| `InversionTime` | 5.5% | |
| `ContrastBolusAgent` | 0.07% | 17 series; only 4 name an agent |
| `ProtocolName`, `StudyDescription`, `ImageLaterality`, `DerivationDescription`, `ImageComments` | 0 | |

Hidden test headers may differ from training headers:

- In the 2024 lumbar competition, `SpacingBetweenSlices` worked on train but broke on the hidden test ([discussion](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/537339)).
- The 2025 aneurysm second-place team reports that "test sets had deleted fields" ([write-up](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611867)).

Every rule below must therefore keep a series when its tags are missing.

## Ranked signals

Rank combines usefulness for our target cases with cost.

| # | Signal | Target case | Tier | Coverage here | Reliability |
|---:|---|---|---|---|---|
| 1 | Sign of each series' physical field-of-view center X, with \|X\| > 40 mm, against the rest of the study | Contralateral knee; stray series | H0 | 100% | Measured here: 8 studies have series on both sides. Seven hold two full ~5-series protocols, and their Bulgarian reports describe both knees. One has a single stray series. As a side rule, center sign agrees with tags in 97.72% of 2,151 studies, and 99.23% with \|X\| > 20 mm ([knee-side coordinates](../../../metadata/knee-side/knee-side-coordinate-investigation.md#agreement-and-potential-coverage)). |
| 2 | Inconsistency with a robust study center: residual > 20 mm, and the center projects outside the series' acquired range | Any non-target series | H0 | 100% | Measured: the robust fit (`outputs/analysis/s/study-center-robust-20260915/summary.txt`, deleted) drops 551 series at 10 mm and 159 at 20 mm. 88 series lie outside their range by up to 184 mm. It flags position, not anatomy. |
| 3 | A `ReceiveCoilName` anatomy word that differs from the study majority | Non-knee anatomy | H1 | 76.7% | Measured here: in study `…93672103` the five ankle series are exactly the five `SENSE_ANKLE_8_AC` series, and the six knee series use `SENSE_KNEE_16_AC`. Only 9 of 3,367 coil-bearing studies have more than one coil name. No literature. |
| 4 | Contiguous `SeriesNumber` blocks | Supports rows 1–3 | H1 | 100% | Measured here: each knee of the 7 bilateral studies is one block, e.g. 201–601 and 801–1201. The ankle block is 1101–1501 and the knee block 301–801. Supporting evidence only. |
| 5 | `ImageType` tokens `LOCALIZER`, `DERIVED`, `SECONDARY`, `MPR`, `PROJECTION IMAGE`, plus "MPR" in the description | Localizers, reformats | H1 | 100% | Measured here: no `LOCALIZER` and no `DERIVED`. 53 series are `ORIGINAL\PRIMARY\PROJECTION IMAGE` (Philips "AX PD FS MPR"), so a DERIVED-only rule misses them. Reported: vendors often store derived maps as `ORIGINAL\PRIMARY`, which caused 56% of one classifier's errors ([Gauriau 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7256138/)). |
| 6 | Fat-suppression evidence: exact `FS` token in `ScanOptions`, description keywords, `IR` scanning sequence with inversion time 100–200 ms | Contrast-label check | H1 | 86–95% | Measured here: 99.0% of series labeled fat-suppressed carry such evidence, against 2.6% of the rest. The `FS` token maps to label 1 in 99.57% of its 12,227 series. Inversion time 100–200 ms maps to label 1 in 500 of 500. Reported: a tags-plus-text fat-suppression classifier reached 0.959 accuracy across body regions ([Kujawa 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13219728/)). |
| 7 | Weighting from TR/TE, checking gradient echo first | T1 vs PD vs T2 inside non-fat-suppressed series; T1 inside fat-suppressed | H1 | 95.1% | Public notebook: the non-fat-suppressed label mixes T1, PD and T2. T1 is 57%, 73% and 39% of axial, coronal and sagittal series ([gchauhan EDA](https://www.kaggle.com/code/gchauhan/eda-rsna-knee-what-the-data-tells-us?scriptVersionId=349664172)). Reported: header-only PD accuracy 62–72% (brain; [Cluceru 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9984597/)). |
| 8 | Plane as the dominant axis of the slice normal | Plane-label check | H0 | 100% | Measured here: agrees with the competition plane for 24,386 of 24,386 series. The same EDA reports 24,371 of 24,371. 489 series are oblique by more than 30°, and 15 have a dominance margin below 0.1. |
| 9 | Post-contrast: `+C`/`POST`/`GD` in the description, an agent name in `ContrastBolusAgent`, a series after the first flagged one | Post-contrast series | H1 | 82% text, 17 agent values | Measured here: 9 series in 4 studies. `ContrastBolusAgent` names an agent in 4 series, holds side words in 10 and holds "15" in 3. Reported: contrast is the weakest header call in every study that measured it (e.g. 0.841, [Kujawa 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13219728/)). |
| 10 | Within-series mode filter on matrix, pixel spacing and orientation | Scout slices mixed into a series | H0 | 100% | Reported in the 2025 aneurysm competition: scout slices differ in matrix size in all 38 cases found ([discussion](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/596183)). The winner kept only the majority matrix and spacing ([write-up](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611846)). Measured here: orientation never changes within a series. |
| 11 | `Laterality` tag | Contralateral knee | H1 | 49.3% | Measured here: not a bilateral marker. 25 Philips studies mix L and R tags while geometry keeps one side, and 46 of their 133 series disagree with geometry. No geometric bilateral study has mixed tags. |
| 12 | Body extent or intensity statistics of a mid slice | Non-knee anatomy; blank or calibration images | I | Census computes them | Unmeasured. The ankle series' foreground widths (77–160 mm) overlap the knee series' widths (139–160 mm) in the same study. |
| 13 | Image models: body-part classifier, bone segmentation, laterality classifier | Anatomy, side | M | — | See [external models](external-evidence.md#external-models-and-data-flagged). |
| 14 | `BodyPartExamined` | Anatomy | H1 | 90.5% | Measured here: KNEE 79.4%, EXTREMITY 9.3%, plus WRIST, ANKLE, HEAD, BRAIN, LIVER, HEART and spine values on knee studies; the ankle series say KNEE. Reported: correct in 58.3% of MRI studies ([Raffy 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10407003/)); 15.5% false entries on 580 MR/CT images ([Gueld 2002](https://doi.org/10.1117/12.467017)). |
| 15 | Anatomy or localizer words in `SeriesDescription` | Anatomy, localizers | H1 | 82.3% | Measured here: no ankle, foot, hip, other-joint, localizer or bilateral words in any description, so no recall on known cases. Reported: 10.6–11.4% of descriptions were misleading ([Salome 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10046648/)). Description rules reach 72% on unseen data ([DeepDicomSort](https://pmc.ncbi.nlm.nih.gov/articles/PMC7782469/)). |
