# Series-filtering findings by case

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
[external models](external-evidence.md#external-models-and-data-flagged). *Inference:* a cheap geometric and coil
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

- Sign of center X agrees with explicit tags in 97.72% of studies, and GEHC studies sit near the origin ([knee-side coordinates](../../../metadata/knee-side/knee-side-coordinate-investigation.md#coordinate-conventions-and-other-metadata-clues)).
- A public notebook claims about 97% agreement for its own center-X rule. It says the rule is no better than chance within 20 mm of the midline, and shows no numbers ([romanrozen notebook](https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline?scriptVersionId=341519155)).

**Unresolved: which knee a bilateral study's labels describe.** Labels are study-level, and the
reports cover both knees. Options considered (none tested):

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
   - Reported: aneurysm scouts differed in matrix in all 38 found cases (see [ranked signals](signals-and-ranking.md#ranked-signals) row 10).
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
and the [tie-break investigation](../../../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#why-the-series-tie)
agrees.

Two public rules exist:

- **gchauhan:** TR < 1000 ms and TE < 30 ms → T1; TE < 60 ms → PD; otherwise T2.
- **romanrozen:** description words first, then gradient echo from `ScanningSequence`, then TR < 800 ms → T1, TE > 60 ms → T2, otherwise PD.

Neither was validated. Literature numbers:

- Header-only PD accuracy was 62.1% (rules) and 72.0% (random forest) on clinical brain data; image models reached 88% ([Cluceru 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9984597/)).
- One large classifier merged PD and T2 into a single class ([Liang 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8635782/)).
- *Inference:* treat PD vs T2 as soft information (for example an embedding) rather than a filter.

## Reformats, retakes, post-contrast and derived series

- **Reformats.** Measured: the only reformats are the 53 `PROJECTION IMAGE` series plus description-only cases ("DESS MPR"). The [tie-break design](../../../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#proposed-deterministic-rule) already ranks originals first.
  - Reported: reformatted 3D post-contrast T1 series caused 43% of one classifier's errors ([Gauriau 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7256138/)).
  - A public aneurysm tip keeps derived images if they are isotropic, long volumes, instead of dropping all derived images ([discussion](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/598083)).
- **Retakes.** Identified by a same-protocol signature with a later `SeriesNumber`; 51 slots have them ([tie-break investigation](../../../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#why-the-series-tie)). No acquisition times exist to confirm order.
  - GE deep-learning reconstructions appear as series *N* and 40000+*N*: same acquisition, two reconstructions.
- **Post-contrast.** Four training studies. The host labels T1 fat-suppressed post-contrast series 0, so they stay out of fat-suppressed slots. `AX_T1W_POST` is labeled 1 (see the table above).
  - `ContrastBolusAgent` is too sparse and too misused here to rely on.
  - The literature calls it the dominant but inconsistently filled contrast feature: removing it cost 16.2% accuracy in [Cluceru 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9984597/). Reviewers found 9% of "pre-contrast" descriptions were post-contrast in [Kasmanoff 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9361920/).
- **Derived maps (DWI, ADC, T2 maps, Dixon fat images).** None found by description, except GE `SMART FAT` / `Water: SMART FAT` pairs (76 series, 37 studies).
  - *Inference:* these look like Dixon-type water/fat outputs.
  - Check whether any `SMART FAT` series without "Water" is a fat-only image before trusting its label.
