# Series-filtering signals

Recorded on 2026-09-17 for series filtering and cleaning. The coil-name rule, the side rule for
a stray other-knee series and the within-series layout guard were implemented on 2026-09-17
([implemented rule](../metadata/series-filtering-census.md#implemented-rule-2026-09-17)); nothing
else here is implemented, and no rule has been validated against disease AUC.

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

- **Measured here:** scratch counts on the series census table `outputs/analysis/s/series-filtering-20260917/census_series.parquet`. The table has 24,371 training and 15 example-test series, with one header (the middle file) and intensity features from three sampled slices per series. These are counts, not validated rules. The census itself is recorded in [series-filtering census](../metadata/series-filtering-census.md).
- **Reported:** a number from the cited source, for its own task and data.
- **Claim:** an unquantified statement in a source.
- **Inference:** our reasoning, not measured.

## Cost tiers and tag availability

Rules must run identically in the Kaggle notebook. Rank them by cost:

| Tier | Meaning | Cost at inference |
|---|---|---|
| **H0** | Header fields the loader already keeps: orientation, position, pixel spacing, rows and columns, instance number, frame count | None |
| **H1** | Any other tag of the same header. `read_header` in `src/knee_abnormality/data/preprocessing/dicom.py` already parses every file's full pre-pixel dataset with `read_partial`, and the [tie-break design](../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#implementation-2026-09-17) makes the same point. | Code only, no extra I/O |
| **I** | Cheap statistics on a few decoded slices: foreground extent, intensity quantiles | CPU |
| **M** | A small image model, possibly with external weights | T4 time and a licence check |

Coverage below counts **non-empty values** across all 24,371 training series (measured here). The
[tie-break tag inventory](../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#why-the-series-tie)
counted element presence in 5,935 repeated-group series. That overstates two tags:

- `Laterality` is present in 78% of those series but non-empty in only 49.3% of all series.
- `ContrastBolusAgent` is present in about 45% but non-empty in only 17 series.

| Tag | Non-empty | Notes |
|---|---:|---|
| `ImageOrientationPatient`, `ImagePositionPatient`, `PixelSpacing`, rows/columns, `SeriesNumber`, `ImageType`, `Manufacturer` | 100% | H0/H1 |
| `RepetitionTime`, `EchoTime`, `ScanningSequence`, `SequenceVariant`, `MRAcquisitionType`, `FlipAngle`, `EchoTrainLength`, `PatientPosition` | 95.1% | |
| `SeriesDescription` | 94.1% | Includes 2,879 `DummySeriesDesc!`, so only **82.3%** carry text |
| `BodyPartExamined` | 90.5% | See [non-knee anatomy](#non-knee-anatomy) |
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
| 1 | Sign of each series' physical field-of-view center X, with \|X\| > 40 mm, against the rest of the study | Contralateral knee; stray series | H0 | 100% | Measured here: 8 studies have series on both sides. Seven hold two full ~5-series protocols, and their Bulgarian reports describe both knees. One has a single stray series. As a side rule, center sign agrees with tags in 97.72% of 2,151 studies, and 99.23% with \|X\| > 20 mm ([knee-side coordinates](../metadata/knee-side-coordinate-investigation.md#agreement-and-potential-coverage)). |
| 2 | Inconsistency with a robust study center: residual > 20 mm, and the center projects outside the series' acquired range | Any non-target series | H0 | 100% | Measured: the [robust fit](../../outputs/analysis/s/study-center-robust-20260915/summary.txt) drops 551 series at 10 mm and 159 at 20 mm. 88 series lie outside their range by up to 184 mm. It flags position, not anatomy. |
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
| 13 | Image models: body-part classifier, bone segmentation, laterality classifier | Anatomy, side | M | — | See [external models](#external-models-and-data-flagged). |
| 14 | `BodyPartExamined` | Anatomy | H1 | 90.5% | Measured here: KNEE 79.4%, EXTREMITY 9.3%, plus WRIST, ANKLE, HEAD, BRAIN, LIVER, HEART and spine values on knee studies; the ankle series say KNEE. Reported: correct in 58.3% of MRI studies ([Raffy 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10407003/)); 15.5% false entries on 580 MR/CT images ([Gueld 2002](https://doi.org/10.1117/12.467017)). |
| 15 | Anatomy or localizer words in `SeriesDescription` | Anatomy, localizers | H1 | 82.3% | Measured here: no ankle, foot, hip, other-joint, localizer or bilateral words in any description, so no recall on known cases. Reported: 10.6–11.4% of descriptions were misleading ([Salome 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10046648/)). Description rules reach 72% on unseen data ([DeepDicomSort](https://pmc.ncbi.nlm.nih.gov/articles/PMC7782469/)). |

## Non-knee anatomy

**Only one case is confirmed**, study `…93672103`, with five ankle series filed in knee slots.
Its headers show what a rule can use:

- **Coil (strong in this study).** `ReceiveCoilName` is `SENSE_ANKLE_8_AC` for the five ankle series (`SeriesNumber` 1101–1501) and `SENSE_KNEE_16_AC` for the six knee series (301–801).
- **Useless tags.** `BodyPartExamined` is KNEE for all eleven. The descriptions ("SAG T1", "AXI T2 SPIR", "COR STIR") name no anatomy.
- **Geometry.** The robust center fit flags the ankle series as a multi-drop study (five drops, 50 mm center move).

What the coil names look like study-wide (measured here):

- Knee exams also run in body, spine, flex and combined knee–foot coils. Five studies use `BODY_18` plus `SPINE_32` for every series.
- **Implication:** a coil name alone is not an anatomy label. Only a change inside a study is informative.
- Of 3,367 studies with coil names, 9 have more than one:
  - the five whole-study body/spine pairs;
  - the ankle study;
  - three others that need a visual check (`…22538230`, `…85221483`, and a single-series `30AA+40PA` study).

Literature on image-based checks:

- **Body-part classifiers exist mostly for radiographs.** One X-ray body-part model reached 0.889 accuracy, and combining it with the header reached 98.7% recall on a 165-image use case ([Cheng 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12817460/)).
- **Reviewers corrected body-part labels** built from `BodyPartExamined` for 3.9% and 12.8% of radiographs at two institutions ([Mitsuyama 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035548/)).
- **No knee- or MSK-specific series classifier was found.**

Knee bone segmentation (femur, tibia, patella) could confirm knee anatomy from images; see
[external models](#external-models-and-data-flagged). *Inference:* a cheap geometric and coil
screen should come first, and an image check only for the few series it flags.

## Contralateral knee

**The host confirms that bilateral studies exist.** Each was "individually reviewed, and the
released report text or DICOM metadata was adjusted as needed to provide sufficient information
for participants to disambiguate" ([host answer](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826)).

A participant counted about seven training studies with both knees; the post gives no list and no
method ([discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735639)).
A later comment asks, unanswered, which knee a bilateral test study's labels refer to.

Measured here, with physical center X of each series and |X| > 40 mm:

- **Eight studies have series on both sides.** All come from `Philips Medical Systems` scanners with `SENSE_HR_KNEE_AC`.
- **Seven hold two complete protocols** (`…06656162`, `…08545893`, `…30589893`, `…32682289`, `…52975364`, `…62455084`, `…85971577`):
  - 4–6 series per knee;
  - each knee in its own contiguous `SeriesNumber` block;
  - identical descriptions without side words.
- **Their reports cover both knees.** Only the first 700 characters of each Bulgarian report were read. Six of the seven excerpts name a side, and three already mention both knees within that span.
- **Laterality tags in these seven.**
  - In six, the positive-X (left) block carries `Laterality=L` and the right block has an empty tag.
  - In `…08545893`, both blocks are empty.
  - *Hypothesis, low confidence:* this pattern may be the host's disambiguation, or just scanner behavior.
- **`…42012883` has one stray sagittal T1** (series 301, X = +142 mm, tag R) against six series at X ≈ −56 to −103 mm (tag R).

The other side signals fail as bilateral detectors (measured here):

- **Mixed `Laterality` tags:** 25 Philips studies, but geometry puts every series on one side. None of the eight geometric cases has mixed tags.
- **Description side words:** present in 461 studies, never mixed within a study.
- **`ContrastBolusAgent`:** holds "right" in one study and "LEFT" in another, with the same protocol in both. Each value agrees with its study's X sign.
  - *Inference:* this is an extra side field, not a bilateral marker.

Reported evidence for the side rule:

- Sign of center X agrees with explicit tags in 97.72% of studies, and GEHC studies sit near the origin ([knee-side coordinates](../metadata/knee-side-coordinate-investigation.md#coordinate-conventions-and-other-metadata-clues)).
- A public notebook claims about 97% agreement for its own center-X rule. It says the rule is no better than chance within 20 mm of the midline, and shows no numbers ([romanrozen notebook](https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline?scriptVersionId=341519155)).

**Open decision: which knee a bilateral study's labels describe.** Labels are study-level, and the
reports cover both knees. Options (none tested):

- keep both knees (the status quo);
- keep one side group;
- treat each side as its own bag and pool the predictions.

Which option is right depends on how the host consolidated labels, which is undocumented.

## Localizers, scouts and calibration

**None appear in training.** Measured here:

- no `LOCALIZER` image type;
- no localizer, scout, survey or calibration words in descriptions;
- no orientation change inside a series;
- no series with fewer than 11 files.

*Inference:* the host removed them. Test curation is undocumented, so keep cheap defensive rules:

1. Drop a series whose `ImageType` contains `LOCALIZER`, or whose description matches `loc|scout|survey|surview|3.?pl|tri.?pl|calib`.
2. Keep only files matching the series' modal matrix, pixel spacing and orientation.
   - Reported: aneurysm scouts differed in matrix in all 38 found cases (see [ranked signals](#ranked-signals) row 10).
   - One aneurysm participant also dropped files with `PROJECTION IMAGE` ([discussion](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/603094)). Here that would remove the 53 Philips reformats, which the tie-break design keeps as the only fat-suppressed PD axial series in 9 slots.
3. Classifier literature treats localizers as a "junk" class that header models learn well. One brain model scored above 99.9% overall, with 34 of 36 errors in that class ([Liang 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8635782/)).
   - Localizers look PD-like on images, so one study excluded them by description and slice count ([Cluceru 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9984597/)).

## Plane and contrast labels

### Plane

**The competition plane equals the dominant-axis rule** (largest absolute component of the slice
normal) for every series, measured here and in the
[public EDA](https://www.kaggle.com/code/gchauhan/eda-rsna-knee-what-the-data-tells-us?scriptVersionId=349664172).
The label was probably derived that way, so this rule cannot find plane errors.

What remains is obliquity:

- 489 series tilt by more than 30°.
- 15 are within 0.1 of a tie between two axes, so their plane is ambiguous by construction.

Published rules use the same dominant-axis idea:

- [de Almeida 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12187622/): largest direction cosine;
- [dcm-classifier](https://github.com/BRAINSia/dcm-classifier): cosine thresholds 0.5 and 0.707;
- [Kasmanoff 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9361920/): orientation above 99% accurate, with an oblique class.

No paper validates oblique thresholds.

### Fat suppression and fluid sensitivity

**The host defines "fluid sensitive" as fluid-bright *and* fat-suppressed**
([overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)),
and the labels follow that definition (measured here):

- **Fat-suppressed but not fluid-bright is labeled 0.**
  - T1 fat-suppressed series: `T1W_TSE_SPIR`, `AX T1 FS`, `t1_space_fs`, `T1W_TSE_FS+C`.
  - Water-excited cartilage gradient echo: `t1_vibe_we` (35 series) and `3D_WATSc` (48).
- **Water-excited DESS is labeled 1:** `t2_de3d_we` (265 series; TR 18 ms, TE 6.8 ms).
- **Consequence:** a pure fat-suppression detector will disagree with the label on these known groups. A rule must pair fat-suppression evidence with weighting.

Evidence rules (measured here):

- Match `ScanOptions` as exact tokens. GE writes `SAT_GEMS` for spatial saturation; a substring test on "SAT" misfires, as the romanrozen notebook notes. The DICOM defined term is `FS` ([PS3.3 C.8.3.1](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.3.html)).
- Description keywords: `fs`, `fatsat`, `stir`, `tirm`, `spair`, `spir`, `we`, `water`.
- An `IR` scanning sequence with inversion time 100–200 ms (STIR).

Candidate label disagreements for visual review (measured here, not confirmed):

| Direction | Series |
|---|---|
| Labeled 1, header says T1 | `AX_T1W_PRE` and `AX_T1W_POST` (Philips SE, TR 625 ms, TE 30 ms) |
| Labeled 1, no fat-suppression evidence | 35 GE `SMART FAT` series (no `FS` token); single `t2_tse_tra_np`, `t2_fl2d_tra` and `LT.TRA_T2W_FFE` series. A matching `AX T2W_FFE` is labeled 0. |
| Labeled 0, but a fat-suppressed PD | One `SAG DP SPIR`, two `pd_tse_fs_sag_np`, one `Water: SMART FAT` |
| Labeled 0, header ambiguous | Six Philips `DummySeriesDesc!` series with `FS`, TR 1,211 ms, TE 23.5 ms |

The literature expects this call to be imperfect:

- The only measured multi-region fat-suppression classifier reached 0.959 accuracy. For T2 series, 19 of 691 non-suppressed series were called suppressed and 22 of 478 suppressed series were called non-suppressed ([Kujawa 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13219728/)).
- `ScanOptions` had almost no permutation importance in a brain classifier (0.001; [Cluceru 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9984597/)).

### Weighting inside the non-fat-suppressed label

**The non-fat-suppressed label mixes T1, PD and T2**, and a repeated series type is usually a second
contrast rather than a retake. The public EDA reports 76% of 3,037 repeat pairs ([gchauhan EDA](https://www.kaggle.com/code/gchauhan/eda-rsna-knee-what-the-data-tells-us?scriptVersionId=349664172)),
and the [tie-break investigation](../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#why-the-series-tie)
agrees.

Two public rules exist:

- **gchauhan:** TR < 1000 ms and TE < 30 ms → T1; TE < 60 ms → PD; otherwise T2.
- **romanrozen:** description words first, then gradient echo from `ScanningSequence`, then TR < 800 ms → T1, TE > 60 ms → T2, otherwise PD.

Neither was validated. Literature numbers:

- Header-only PD accuracy was 62.1% (rules) and 72.0% (random forest) on clinical brain data; image models reached 88% ([Cluceru 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9984597/)).
- One large classifier merged PD and T2 into a single class ([Liang 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8635782/)).
- *Inference:* treat PD vs T2 as soft information (for example an embedding) rather than a filter.

## Reformats, retakes, post-contrast and derived series

- **Reformats.** Measured: the only reformats are the 53 `PROJECTION IMAGE` series plus description-only cases ("DESS MPR"). The [tie-break design](../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#proposed-deterministic-rule) already ranks originals first.
  - Reported: reformatted 3D post-contrast T1 series caused 43% of one classifier's errors ([Gauriau 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7256138/)).
  - A public aneurysm tip keeps derived images if they are isotropic, long volumes, instead of dropping all derived images ([discussion](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/598083)).
- **Retakes.** Identified by a same-protocol signature with a later `SeriesNumber`; 51 slots have them ([tie-break investigation](../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#why-the-series-tie)). No acquisition times exist to confirm order.
  - GE deep-learning reconstructions appear as series *N* and 40000+*N*: same acquisition, two reconstructions.
- **Post-contrast.** Four training studies. The host labels T1 fat-suppressed post-contrast series 0, so they stay out of fat-suppressed slots. `AX_T1W_POST` is labeled 1 (see the table above).
  - `ContrastBolusAgent` is too sparse and too misused here to rely on.
  - The literature calls it the dominant but inconsistently filled contrast feature: removing it cost 16.2% accuracy in [Cluceru 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9984597/). Reviewers found 9% of "pre-contrast" descriptions were post-contrast in [Kasmanoff 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9361920/).
- **Derived maps (DWI, ADC, T2 maps, Dixon fat images).** None found by description, except GE `SMART FAT` / `Water: SMART FAT` pairs (76 series, 37 studies).
  - *Inference:* these look like Dixon-type water/fat outputs.
  - Check whether any `SMART FAT` series without "Water" is a fat-only image before trusting its label.

## Lessons from other Kaggle competitions

| Competition | What was reported | Measured? |
|---|---|---|
| Lumbar spine 2024 | Host series descriptions were used as given. Multiple axial series were combined: crops from both ([2nd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452)), averaged predictions ([11th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539569)) or position-sorted stacks ([6th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541813)). The 2nd place used a slice-level localizer to detect which levels an axial series covered. | Methods only; no filtering ablation |
| Lumbar spine 2024 | Hidden-test `SpacingBetweenSlices` broke header code that worked on train ([discussion](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/537339)) | Observed through submission failures |
| Aneurysm 2025 | The winner excluded about 60 training series (orientation anomalies, corrupt files, implausible spacing), kept majority matrix and spacing per series, and built an orientation fix that "fixed all identified orientation issues" in training but had no leaderboard effect ([1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611846)) | Training fix count not given; no LB effect |
| Aneurysm 2025 | Scout slices inside series, found by matrix mismatch (38 cases); host: "reflects real world data", left as is ([discussion](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/596183)) | Participant count |
| Aneurysm 2025 | The 2nd place trained a T2-specific orientation classifier because test headers had deleted fields ([2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611867)). The 3rd place found modality and plane auxiliary heads did not help ([3rd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611856)). | Claims |
| Brain tumor 2021 | Participants listed series with orientation or slice-order problems in their DICOM metadata ([discussion](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/267714)) | Participant list, no rate |

In this competition, the public discussions contain no series-filtering method and no measured
filtering result as of 2026-09-17. All 112 topics were searched. The public
[gchauhan EDA](https://www.kaggle.com/code/gchauhan/eda-rsna-knee-what-the-data-tells-us?scriptVersionId=349664172)
keys series by header contrast and flags reformats by `ImageType`. The
[romanrozen notebook](https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline?scriptVersionId=341519155)
fills slots from header-derived fat suppression and weighting. Neither measures the effect.

## External models and data (flagged)

These are notes only. Any use must pass the
[external-data admission criteria](pretraining-and-external-data.md#admission-criteria-for-an-external-data-experiment)
and the competition rules.

- **TotalSegmentator MRI** ([code](https://github.com/wasserth/TotalSegmentator), [paper](https://arxiv.org/abs/2405.19492)).
  - The open `total_mr` task (Apache-2.0) has femur and hip but no tibia, fibula, patella or tarsal bones.
  - Those bones are in `appendicular_bones_mr`, which needs a licence: free for non-commercial academic use. Talus and calcaneus share one "tarsal" class.
  - Weights: about 231 MB × 2 (1.5 mm model), 126 MB (3 mm), 44 MB (6 mm).
  - The README gives no T4 timings.
  - Reported: internal Dice 0.839 over 616 MRI and 527 CT exams.
- **MRSegmentator** ([code](https://github.com/hhaentze/MRSegmentator)). Apache-2.0, 1.15 GB. Its only leg bones are femur and hips.
- **Body-region classifier for MRI/CT** ([Raffy 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10407003/)). No model release was found.
  - 18 MRI classes, including knee, calf and foot.
  - Reported: 92.3% weighted sensitivity on 792 MRI studies from 27 institutions; knee 95.2%, foot 95.3%. The authors note knee–elbow confusion.
  - Reported: `BodyPartExamined` was correct in only 58.3% of MRI studies.
- **BPReg and the original body-part regressor** are CT-only and exclude legs ([BPReg](https://arxiv.org/abs/2110.09148), Apache-2.0 code, CC BY-SA weights). They are not usable here.
- **Header or image sequence classifiers trained on brain, body or prostate data** (no knee training data):
  - [dcm-classifier](https://github.com/BRAINSia/dcm-classifier) (Apache-2.0);
  - [DeepDicomSort](https://pmc.ncbi.nlm.nih.gov/articles/PMC7782469/) (Apache-2.0);
  - [de Almeida 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12187622/) (MIT code).
  - Reported: moving to an unseen centre or vendor cost 15–30 points on some classes. At site B, Philips accuracy was 71.6% ([Gauriau 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7256138/)); leaving one centre out cut F1 by up to 0.18 ([de Almeida 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12187622/)).
  - *Inference:* use them as references for rules, not as drop-in models.
- **No image-based left/right classifier for knee MRI** was found.
  - Radiograph laterality models rely on lead markers, which MRI lacks ([Filice 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6646614/)).
  - A 3D CNN recovers brain MRI left–right orientation at 99.8% ([Nishimaki 2025](https://doi.org/10.1002/brb3.70299)).
- **Existing localizer rules** come from converter code. dcm2niix uses vendor-specific description, sequence-name and private-tag rules, plus "orientation varies within a series" and "fewer than 2 slices" ([dcm2niix](https://github.com/rordenlab/dcm2niix)). No accuracy is reported.
  - DICOM defines `LOCALIZER` as an image type for CT only, not for MR ([PS3.3 C.8.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.2.html)). An `ImageType` rule is therefore vendor-dependent for MR.
  - GDCM calls a plane oblique unless the largest direction cosine exceeds 0.8 ([gdcmOrientation.cxx](https://github.com/malaterre/GDCM/blob/master/Source/MediaStorageAndFileFormat/gdcmOrientation.cxx)).

## Open-source tool rules

These rules were read from code at pinned commits. None of the tools reports accuracy for its rules.

- **dcm2niix `-i y`** ([batch L11455](https://github.com/rordenlab/dcm2niix/blob/8282dbf7d37d34b29ef008061622a44077e24707/console/nii_dicom_batch.cpp#L11455-L11467), BSD-style licence) skips a series when:
  - it is derived: `ImageType` contains DERIVED, or ADC, TRACE or FA;
  - it is a localizer: Siemens `AAHScout` in the description, or `fl2d1` in the sequence name;
  - it is a single 2D image.
  - Orientation varying by more than 1e-4 within a series also invalidates the series.
  - Its weighting guess ([L1361](https://github.com/rordenlab/dcm2niix/blob/8282dbf7d37d34b29ef008061622a44077e24707/console/nii_dicom_batch.cpp#L1361)): spin echo with TE ≥ 45 ms is T2, otherwise PD.
- **Flywheel classification profiles** ([MR intent](https://gitlab.com/flywheel-io/scientific-solutions/lib/fw-classification-profiles/-/blob/94581121e6710d09ac3d8a63eacb36fac2065706/classification_profiles/MR/classify_intent.yaml), MIT):
  - **Localizer:** spacing between slices > 8 mm and fewer than 10 files, with no DERIVED, SECONDARY, REFORMATTED or PROJECTION image type (Philips: fewer than 7 files).
  - **Localizer by name:** `localiz|localis|\bloc\b|3[\s_-]?pl|scout|survey`.
  - **Calibration by name:** `asset.*cal|calibration`.
  - **Plane:** direction cosines must exceed 0.75, otherwise the series is oblique ([orientation](https://gitlab.com/flywheel-io/scientific-solutions/lib/fw-classification-profiles/-/blob/94581121e6710d09ac3d8a63eacb36fac2065706/classification_profiles/MR/classify_orientation.yaml)).
  - **Vendor weighting**, all spin echo ([Siemens](https://gitlab.com/flywheel-io/scientific-solutions/lib/fw-classification-profiles/-/blob/94581121e6710d09ac3d8a63eacb36fac2065706/classification_profiles/MR/SIEMENS.yaml), [GE](https://gitlab.com/flywheel-io/scientific-solutions/lib/fw-classification-profiles/-/blob/94581121e6710d09ac3d8a63eacb36fac2065706/classification_profiles/MR/GE.yaml)):

    | Class | Rule |
    |---|---|
    | PD | TR > 2000 ms, TE < 25–30 ms |
    | T2 | TE > 75 ms, TR > 2400 ms |
    | T1 | TR < 700–900 ms, TE < 15–30 ms |
    | STIR | TI < 200–250 ms, TE > 30 ms, TR > 1000 ms |

  - Its `ScanOptions` FS rule is commented out, and it has no MR contrast rule.
  - *Inference:* knee PD series often use TE of 25–45 ms, so the PD/T2 cut-offs need checking against our data.
- **Retakes** in HeuDiConv/ReproIn ([reproin.py](https://github.com/nipy/heudiconv/blob/11c8769edff2862c86c631fb0366cc231e461f81/heudiconv/heuristics/reproin.py#L690), Apache-2.0) and Dcm2Bids ([sidecar.py](https://github.com/UNFmontreal/Dcm2Bids/blob/7f5ecf18775444d16685624f6c3906ec8b36acf8/dcm2bids/sidecar.py#L28), GPLv3): the last series with the same name wins, ordered by `SeriesNumber` then acquisition time. This matches the tie-break design's retake rule.
- **dicom2nifti** ([convert_generic.py](https://github.com/icometrix/dicom2nifti/blob/d68253ef3b173f43067b6f9db83788bc25cf50cd/dicom2nifti/convert_generic.py#L364), MIT):
  - drops `LOCALIZER` slices;
  - within mixed-orientation series, keeps orientation groups of at least 4 slices;
  - rejects series with 3 or fewer slices.
- **ezBIDS** ([core](https://github.com/brainlife/ezbids/blob/172c6c213dae3987b30bc8f75bf4109a6112b497/handler/ezBIDS_core/ezBIDS_core.py#L1852), MIT): a localizer has "localizer" or "scout" in its description, or was split by orientation.
- **Contrast rules disagree.** [dicomtools](https://github.com/wfwiggins/dicomtools/blob/5bc2af9849dc1924bda3ffb0f448be3605f269cb/dicomtools/series/preproc.py#L43) flags a description matching `(\+-?c|post)`, or any `ContrastBolusAgent` string. The standard makes `ContrastBolusAgent` Type 2 inside a module required only when contrast was used ([C.7.6.4](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.4.html)). Here the second test would misfire on the "right", "LEFT" and "15" values.
- **Vendor conventions.** Siemens marks distortion-corrected (`DIS2D`) and uncorrected (`ND`) copies in `ImageType`, which can look like duplicate series. Siemens deep-learning reconstruction appears as `DRB`/`DRS` tokens, which are present here ([dcm2niix Siemens notes](https://github.com/rordenlab/dcm2niix/blob/8282dbf7d37d34b29ef008061622a44077e24707/Siemens/README.md)).
- **A public competitor repo** ([slots.py](https://github.com/Sibusiso-K/RSNA-Knee-Abnormality-Detection/blob/f1c02d2d7affdc71ddf1a87f50e78425da3b0565/src/data/slots.py#L86)) uses a fat-suppression regex, T1 as TR < 800 ms and TE < 30 ms, and side from center X with a 10 mm dead zone. Its notebook claims 97.31% agreement between header fat suppression and the label, which is unverified.

## Candidate rules to test against the census

Every rule keeps a series when its inputs are missing. None may remove the last series of a
study. Flags go to the visual census first; only confirmed rules become exclusions.

1. **Defensive per-file and per-series filters (H0/H1; expected 0 training hits).**
   - Keep the modal matrix, pixel spacing and orientation within a series.
   - Drop `LOCALIZER` image types and localizer descriptions.
   - Count hits on train and on the three example-test studies; any training hit is a finding.
2. **Side groups (H0).**
   - Compute each series' median physical center X.
   - When a study has at least two series at X > +40 mm and at least two at X < −40 mm, label it bilateral: 7 expected.
   - When exactly one series sits opposite at least three others, label it a stray: 1 expected. Candidates for exclusion.
   - Check that each group is a contiguous `SeriesNumber` block.
   - Report GEHC and near-origin studies separately; the sign convention is weak there.
3. **Coil anatomy (H1).**
   - Normalize `ReceiveCoilName` to anatomy words (KNEE, ANKLE, FOOT, WRIST, HAND, SHOULDER, HIP, SPINE, BODY, HEAD). Combined coils such as `KNEE_FOOT` count as knee.
   - Flag a minority group whose words exclude KNEE when the study majority includes KNEE: 5 series expected.
   - Whole-study non-knee coils are not flagged.
4. **Geometric outlier (H0).**
   - Flag series dropped by the 20 mm robust fit whose center projects outside their acquired range.
   - Count how many rows 2–3 already explain, and send the rest to the visual census.
5. **Label checks, flag only (H1).**
   - Fat-suppression evidence versus label, with the known-0 groups (T1 fat-suppressed, water-excited cartilage) excluded.
   - T1-like weighting (TR < 800 ms, not gradient echo) under label 1.
   - Post-contrast words or agent names.
   - Dominance margin below 0.1.
   - Expect a few dozen series; review them visually before any relabeling.
6. **Decision gates.**
   - Exclude a series only when two independent signals agree (e.g. rows 2 or 3 plus row 4) or the census confirms it visually.
   - Bilateral studies wait for the [open decision](#contralateral-knee).
   - Before an overnight run, measure how many validation studies and triplets each rule changes.

## Open questions and gaps

- Which knee a bilateral study's labels describe, and whether the one-sided `Laterality=L` tags are the host's disambiguation.
- Whether the hidden test contains localizers, reformats or other anatomy that training lacks (unknowable locally). The defensive rules exist for that case.
- Whether GE `SMART FAT` series without "Water" are water or fat images.
- The open-source tool survey was stopped early. What it found is in [open-source tool rules](#open-source-tool-rules); GE conformance statements (the `*_GEMS` scan-option meanings, deep-learning reconstruction numbering) remain unread.
- Knee dataset curation reports give no exclusion counts: fastMRI, MRNet, SKM-TEA and KneeMRI describe manual checks without numbers. The lumbar organizers built per-sequence collages to catch wrong sequence labels but give no counts ([LumbarDISC](https://arxiv.org/abs/2506.09162)).
- No knee or MSK series classifier, bilateral-detection method or MRI knee-laterality model was found. Every reported header-classifier number comes from brain, body, prostate or radiograph data.

## Sources

Kaggle pages were read through Kaggle's public discussion API and rendered notebook versions on
2026-09-17. Notebook links pin the version read. Literature entries give the DOI or PMC link;
numbers marked "reported" belong to each source's own task.

**Header reliability**

- Gueld et al. 2002, *Quality of DICOM header information for image categorization*, Proc SPIE 4685. [doi:10.1117/12.467017](https://doi.org/10.1117/12.467017). Abstract only.
- Salome et al. 2023, MR-Class, Cancers. [PMC10046648](https://pmc.ncbi.nlm.nih.gov/articles/PMC10046648/).
- Raffy et al. 2023, body-region classification, J Digit Imaging. [PMC10407003](https://pmc.ncbi.nlm.nih.gov/articles/PMC10407003/).
- Cheng et al. 2025, BMC Med Imaging. [PMC12817460](https://pmc.ncbi.nlm.nih.gov/articles/PMC12817460/).
- Mitsuyama et al. 2025, Eur Radiol. [PMC13035548](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035548/).

**Sequence classifiers**

- Gauriau et al. 2020. [PMC7256138](https://pmc.ncbi.nlm.nih.gov/articles/PMC7256138/).
- Liang et al. 2021. [PMC8635782](https://pmc.ncbi.nlm.nih.gov/articles/PMC8635782/).
- Kasmanoff et al. 2023. [PMC9361920](https://pmc.ncbi.nlm.nih.gov/articles/PMC9361920/).
- Cluceru et al. 2023. [PMC9984597](https://pmc.ncbi.nlm.nih.gov/articles/PMC9984597/).
- Kujawa et al. 2026. [PMC13219728](https://pmc.ncbi.nlm.nih.gov/articles/PMC13219728/).
- de Almeida et al. 2025. [PMC12187622](https://pmc.ncbi.nlm.nih.gov/articles/PMC12187622/).
- van der Voort et al. 2021, DeepDicomSort. [PMC7782469](https://pmc.ncbi.nlm.nih.gov/articles/PMC7782469/).

**Laterality and segmentation**

- Filice & Frantz 2019. [PMC6646614](https://pmc.ncbi.nlm.nih.gov/articles/PMC6646614/).
- Akinci D'Antonoli et al. 2025, TotalSegmentator MRI. [arXiv:2405.19492](https://arxiv.org/abs/2405.19492).

**Standards and code**

- DICOM PS3.3 [C.8.3 MR](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.3.html), [C.8.2 CT](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.2.html) and [C.7.3 laterality](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.3.html).
- [dcm2niix](https://github.com/rordenlab/dcm2niix), [GDCM orientation](https://github.com/malaterre/GDCM/blob/master/Source/MediaStorageAndFileFormat/gdcmOrientation.cxx) and [dcm-classifier](https://github.com/BRAINSia/dcm-classifier).

**Kaggle, this competition**

- Host answers: [bilateral studies](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826), [overview and definitions](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343).
- Discussions: [two knees](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735639), [six data traps](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741310).
- Notebooks: [gchauhan EDA v349664172](https://www.kaggle.com/code/gchauhan/eda-rsna-knee-what-the-data-tells-us?scriptVersionId=349664172), [romanrozen v341519155](https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline?scriptVersionId=341519155).

**Kaggle, earlier competitions**

- Lumbar 2024: [537339](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/537339), [539452](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452), [539569](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539569), [541813](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541813).
- Aneurysm 2025: [596183](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/596183), [598083](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/598083), [603094](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/603094), [611846](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611846), [611856](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611856), [611867](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611867).
- Brain tumor 2021: [267714](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/267714).
