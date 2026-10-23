# Series filtering: other competitions, external models and tool rules

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
[external-data admission criteria](../../backbones-and-pretraining/pretraining-and-external-data.md#admission-criteria-for-an-external-data-experiment)
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
