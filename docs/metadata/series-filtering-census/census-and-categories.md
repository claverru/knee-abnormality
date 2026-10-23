# Series filtering census: method and categories

**Series that should not reach the model are rare (measured): 14 non-knee series in 8 of 4,407
training studies, 35 other-knee series in 8 studies (7 of them bilateral exams), 0 localizers,
0 plane labels that disagree with the acquisition normal, 12 contrast slot mislabels and one
study with anatomically swapped planes.** Together they receive about 0.2% of training
triplets, and removing or relabelling them moves fold-0 macro AUC by at most 0.0003 on the current
checkpoints ([inference check](flags-and-inference-effect.md#inference-effect-fold-0)). Budget reallocation is therefore not
at risk of spending slices on junk; the larger structural facts are that 19% of T1/PD-slot series
are T2-weighted without fat suppression (host convention) and 129 are T1 fat-suppressed.

**Implemented (2026-09-17):** the [revised rule](proposed-rule-and-follow-up.md#labelled-findings-in-off-knee-series) drops only
the ankle series (receive-coil change) and the stray other-knee series; far thigh and calf series,
contrast relabels and plane relabels are not implemented
([implemented rule](implemented-rule.md#implemented-rule-2026-09-17)).

Scope: all 24,371 training series (4,407 studies) and the 15 local test-example series (3 studies);
fold 0 of `all_5fold_v3` (904 studies, 5,061 series) for the inference check. CPU only except the
inference check. Scripts: [snapshots](../../history/sources/series-and-sampling/README.md#series-filtering) (`index_all.py`, `geometry_signals.py`,
`contrast.py`, `planes.py`, `flags.py`, `census.py`, `precision.py`, `infer.py`, `analyze.py`,
`sheets.py`, `study_sheets.py`, `verdicts_*.txt`). Outputs:
`outputs/analysis/s/series-filtering-20260917/` (`series_flags_with_triplets.parquet`,
`category_counts.csv`, `precision.json`, `inference_results.json`, `inference_deltas.csv`,
contact sheets in `sheets/`).

## Method

- **Headers and geometry:** the loader's own indexing (`index_series`, `estimate_reference`) for
  every series, one header (the middle file) with 43 tags, and the current evaluation and training
  selections (`select_triplets`, seed 42) for triplet shares. All 24,386 series have valid native
  geometry (`geometry.reason` is null everywhere).
- **Study centre:** the 20 mm robust fit of the [robust study centre](../../history/robust-study-centre.md)
  analysis, reimplemented (identical drops). Per series: in-plane residual, whether the centre
  projects outside the acquired slab (`outside_range_mm`) or the field of view (`fov_outside_mm`),
  and the slab-centre offset from the study centre.
- **Contrast:** a header class (fat-suppression evidence from `ScanOptions` FS/WE tokens,
  inversion recovery, or the description; T1 timing TE < 20 ms and TR < 1,000 ms, excluding DESS
  and TrueFISP) and a gradient-boosted intensity classifier on nine body-mask quantile features of
  three slices (study-grouped out-of-fold AUC 0.994 for the fluid label; headers alone 0.999).
- **Visual review:** contact sheets read by eye: every study with a geometric flag (152) plus all
  47 robust-rejected studies and the 3 test examples (all series of each study), 99 contrast or
  plane candidates with a same-plane reference series, 25 special series, and 40 random studies
  without any flag (217 series). Verdicts are in the `verdicts_*.txt` files.

## Counts

Triplet shares use the training budget (18/18/15/15/15/15 images, mean of four seeded epochs) for
all training studies and the validation budget (multiplier 2) for fold 0. Totals: 111,934 training
triplets; 45,846 fold-0 validation triplets.

| Category | Train series / studies | Train triplet share | Fold 0 series / studies | Fold 0 triplet share |
|---|---:|---:|---:|---:|
| 1. Non-knee, confirmed (ankle 5, thigh 5, calf 4) | 14 / 8 | 0.041% | 1 / 1 | 0.002% |
| 2. Other knee, bilateral exams (7 Philips studies, bilateral reports) | 34 / 7 | 0.076% | 10 / 2 | 0.113% |
| 2. Other knee, stray single series | 1 / 1 | 0% | 1 / 1 | 0% |
| 3. Localizer, scout, survey or < 11 files | 0 / 0 | 0% | 0 / 0 | 0% |
| 3. Pixel spacing > 1 mm (300 mm FOV, 200-slice axials) | 19 / 19 | 0.098% | 2 / 2 | 0.052% |
| 3. Small-FOV patellar axial (< 110 mm) | 93 / 90 | 0.281% | 14 / 14 | 0.209% |
| 3. Thin slab (< 25 mm, 11–12 slices) | 54 / 53 | 0.032% | 10 / 10 | 0.026% |
| 4. Plane label ≠ largest normal component | 0 / 0 | 0% | 0 / 0 | 0% |
| 4. Oblique 30–45° from the labelled axis | 485 / 451 | 0.91% | 113 / 101 | 1.06% |
| 4. Anatomical planes swapped (patient-positioning study) | 7 / 1 | 0.019% | 7 / 1 | 0.089% |
| 5. T1 fat-suppressed in a T1/PD slot | 129 / 103 | 0.27% | 26 / 26 | 0.27% |
| 5. Fluid fat-suppressed in a T1/PD slot (confirmed) | 9 / 9 | 0.030% | 2 / 2 | 0.022% |
| 5. T2/T2* without fat suppression in a fluid slot (confirmed) | 3 / 3 | 0.008% | 2 / 2 | 0.022% |
| 5. T2 without fat suppression in a T1/PD slot (host convention) | 1,946 / 1,484 | 5.56% | 412 / 313 | 5.71% |
| 5. Zero-TE GE sagittal (TE 0.016 ms) | 130 / 88 | 0.25% | 40 / 29 | 0.39% |
| 6. 3D acquisitions | 836 / 659 | 3.06% | 207 / 160 | 3.45% |
| 6. DESS | 268 / 264 | 1.38% | 63 / 62 | 1.57% |
| 6. Post-contrast T1 FS | 12 / 5 | 0.024% | 2 / 1 | 0.022% |
| 6. Philips MPR reformats | 50 / 50 | 0.14% | 8 / 8 | 0.11% |

The three local test examples have no flagged series (all 15 reviewed as target knee).

### 1. Non-knee anatomy (measured)

Five ankle series in `…93672103`; axial slabs of the distal thigh in `…44943102` (thigh tumour),
`…57128226`, `…07160312`, `…86735623` (femoral lesion follow-up; its only axial T1/PD) and
`…52031539` (suprapatellar recess, borderline); calf-level series in `…77308505` and three in
`…63928925` (gastrocnemius lesion). In 4 of these 8 studies an off-knee series is the only series
of its slot. Hip, foot-only and other body parts: none. `BodyPartExamined` remains useless
(ANKLE, WRIST, HEAD, HEART and spine values on knee series).

The current physical sampler already starves most of them: its truncated Gaussian puts little mass
on a slab far from the centre, so these 14 series receive 46 of 111,934 training triplets
(measured); they only take a full budget when they are alone in their slot.

### 2. Other knee (measured)

The x-coordinate spread of series volume centres is bimodal: 4,399 studies ≤ 60 mm, 8 studies
188–237 mm. Seven are Philips bilateral exams with five series per knee, six with a Bulgarian
report that describes both knees, so their study labels mix both knees and neither side is
"non-target". One, `…42012883` (fold 0), is a unilateral study with one stray sagittal T1 of the
other knee; it gets 0 triplets today. `Laterality` tags cannot separate them (only one side tagged
in `…06656162`).

### 3. Localizers and low resolution (measured)

No localizer, scout, survey, calibration, multi-orientation or < 11-file series exists in this
snapshot (the host's series table appears to exclude them). The only coarse series are 19 GE-like
`DummySeriesDesc!` axials with 300 mm FOV, 1.17 mm pixels and 200 slices; 15 reviewed, all contain
the knee. Small-FOV patellar axials (Philips `AXI DP SPIR` and similar, 93) and thin-slab
sagittals (54) are targeted partial acquisitions, not junk; patellar and other partial-coverage
knee series are 91 of the 159 robust-dropped series.

### 4. Plane (measured)

`Anatomical_Plane` equals the largest component of the acquisition normal for all 24,386 series,
so the host labels follow scanner geometry. Two consequences:

- **ACL/PCL-oblique series** (485 at 30–45°) fall into the axial or coronal slot by which side of
  45° they land on: the same Philips `RT.`/`LT.COR ACL T2W FS` protocol sits in the axial fluid slot 17 times
  and the coronal fluid slot 97 times. Reviewed examples look like axial-coronal hybrids in both slots.
- **Anatomically swapped study** `…34605778` (fold 0): the patient was positioned so that scanner
  y is anatomical left–right; its three anatomical sagittals are labelled Coronal and its coronals
  Sagittal (descriptions and images agree with each other, not with the labels). Descriptions
  alone are a poor relabel signal (6 of 14 description-plane mismatches below 25° are real; the
  other 8 are wrong descriptions); requiring at least three mismatching series over two planes in
  one study isolates this study only (7 of 7 correct).

### 5. Contrast (measured)

The table has only two flag combinations, (0,0) and (1,1), and the host's convention is
"fluid-sensitive" = fat-suppressed fluid-sensitive:

- **T2 without fat suppression** (TE ≥ 60 ms, 1,946 series) is labelled (0,0) consistently, so the
  T1/PD slots are 19% T2-weighted by series count. Not a mislabel; budget reallocation should treat
  "T1/PD" as "non-fat-suppressed".
- **T1 fat-suppressed** series (VIBE/WATS water-excitation cartilage GRE, post-contrast and
  arthrogram T1 FS, 129) are labelled (0,0). Fluid-sensitivity is right, fat suppression is not;
  there is no (0,1) slot. Header precision 47/47 reviewed.
- **True slot mislabels:** 9 fluid-sensitive fat-suppressed series in T1/PD slots (PD/T2 SPIR or
  FS, one DESS) and 3 T2/T2* series without fat suppression in fluid slots. Header evidence alone
  has 9/13 and 3/8 reviewed precision (regex misses: GE `SMART FAT`, `frFSE`, and FS options on
  images that are visibly not fat-suppressed); requiring the intensity classifier to agree
  (p > 0.5 or < 0.5) gives 9/9 and 3/3, and the intensity veto was right in all 4 reviewed
  vetoes.
- **Intensity-only disagreements** (210) are not mislabels: 0 of 29 reviewed (23 correct labels,
  3 T2 convention, 2 zero-TE, 1 T2* multi-echo).
- **Zero-TE GE sagittals** (130, `SQPIX_GEMS`, TE 0.016 ms) have flat, near-proton-density contrast
  unlike any other T1/PD series; kept.

### 6. Other oddities

3D acquisitions (836; 268 DESS in the axial fluid slot with 320 mm FOV), 12 post-contrast series
in 5 studies, Philips `AX PD FS MPR` reformats, GE deep-learning reconstruction duplicates and
same-protocol retakes: see the [tie-break investigation](../../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#why-the-series-tie).
666 Philips structural series carry a non-zero `DiffusionBValue`; none is a diffusion series.
