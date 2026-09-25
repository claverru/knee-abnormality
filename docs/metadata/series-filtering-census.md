# Series filtering and cleaning census

**Series that should not reach the model are rare (measured): 14 non-knee series in 8 of 4,407
training studies, 35 other-knee series in 8 studies (7 of them bilateral exams), 0 localizers,
0 plane labels that disagree with the acquisition normal, 12 contrast slot mislabels and one
study with anatomically swapped planes.** Together they receive about 0.2% of training
triplets, and removing or relabelling them moves fold-0 macro AUC by at most 0.0003 on the current
checkpoints ([inference check](#inference-effect-fold-0)). Budget reallocation is therefore not
at risk of spending slices on junk; the larger structural facts are that 19% of T1/PD-slot series
are T2-weighted without fat suppression (host convention) and 129 are T1 fat-suppressed.

**Implemented (2026-09-17):** the [revised rule](#labelled-findings-in-off-knee-series) drops only
the ankle series (receive-coil change) and the stray other-knee series; far thigh and calf series,
contrast relabels and plane relabels are not implemented
([implemented rule](#implemented-rule-2026-09-17)).

Scope: all 24,371 training series (4,407 studies) and the 15 local test-example series (3 studies);
fold 0 of `all_5fold_v3` (904 studies, 5,061 series) for the inference check. CPU only except the
inference check. Scripts: [snapshots](../history/series-and-sampling-sources/README.md#series-filtering) (`index_all.py`, `geometry_signals.py`,
`contrast.py`, `planes.py`, `flags.py`, `census.py`, `precision.py`, `infer.py`, `analyze.py`,
`sheets.py`, `study_sheets.py`, `verdicts_*.txt`). Outputs:
`outputs/analysis/series-filtering-20260917/` (`series_flags_with_triplets.parquet`,
`category_counts.csv`, `precision.json`, `inference_results.json`, `inference_deltas.csv`,
contact sheets in `sheets/`).

## Method

- **Headers and geometry:** the loader's own indexing (`index_series`, `estimate_reference`) for
  every series, one header (the middle file) with 43 tags, and the current evaluation and training
  selections (`select_triplets`, seed 42) for triplet shares. All 24,386 series have valid native
  geometry (`geometry.reason` is null everywhere).
- **Study centre:** the 20 mm robust fit of the [robust study centre](../history/robust-study-centre.md)
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
same-protocol retakes: see the [tie-break investigation](../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#why-the-series-tie).
666 Philips structural series carry a non-zero `DiffusionBValue`; none is a diffusion series.

## Flag precision and recall

Precision is non-target (non-knee or other knee) among reviewed series (`precision.json`).

| Flag | Flagged | Reviewed | Non-target precision |
|---|---:|---:|---:|
| Proposed far rule (below) | 11 | 11 | 1.00 |
| Other-knee side (x spread > 150 mm) | 35 | 35 | 1.00 |
| Robust 20 mm drop | 159 | 159 | 0.16 (91 patellar, 37 target, 6 other session) |
| Centre projects outside slab or FOV | 127 | 127 | 0.32 |
| Slab-centre offset > 80 mm | 43 | 43 | 0.84 |

Recall: all 47 robust-rejected studies were reviewed; 3 non-knee series in `…63928925` cannot be
seen by any centre-based rule (the study has no accepted centre). A non-knee series in an accepted
study that escapes every geometric flag would have to lie within 20 mm in-plane and inside the
slab of the centre. The random sample of 40 unflagged studies (217 series) found no non-knee,
other-knee or plane problem and one missed T1 FS series (Philips `WATS`, now in the pattern);
by the rule of three, the per-study miss rate is below about 7.5% at 95% (hypothesis-level
bound, small sample).

## Inference effect (fold 0)

Measured. Re-scoring of the three 2026-09-17 control checkpoints (`00-control`,
`01-control-seed-43`, `02-control-seed-44`, [overnight results](../validation/results/2026-09-17-overnight-results.md))
with fold-0 series removed from, or moved within, the series index **before preparation**, so the
study reference is re-estimated and same-slot siblings take over the budget; nothing reallocates
budget across slots. Only affected studies are re-scored, at batch size 1 for both the intact and
the modified study (batch-1 intact differs from the saved batch-8 predictions by up to 0.006 in
probability through bf16 batch-composition numerics). Paired report-group bootstrap (852 groups,
2,000 draws), delta of macro AUC averaged over the checkpoints (intact 0.8930 / 0.8898 / 0.8916).

| Variant (fold 0) | Series | Studies | Δ macro ×1000 | 95% CI ×1000 | Per checkpoint ×1000 |
|---|---:|---:|---:|---|---|
| Drop far + stray other-knee (proposed drops) | 2 | 2 | −0.004 | [−0.040, +0.028] | +0.004, −0.012, −0.005 |
| Drop other-knee minor side (incl. bilateral) | 11 | 3 | −0.016 | [−0.096, +0.052] | −0.025, +0.014, −0.038 |
| Drop pixel > 1 mm | 2 | 2 | +0.015 | [−0.001, +0.055] | −0.003, +0.031, +0.017 |
| Drop T1 FS in T1/PD slots | 26 | 26 | +0.088 | [−0.120, +0.369] | +0.009, +0.100, +0.155 |
| Drop zero-TE | 40 | 29 | −0.092 | [−0.207, +0.030] | −0.095, −0.134, −0.047 |
| Drop small-FOV patellar axials | 14 | 14 | −0.061 | [−0.235, +0.103] | −0.058, +0.046, −0.171 |
| Drop oblique > 30° | 113 | 101 | −0.264 | [−1.064, +0.562] | −0.028, −0.287, −0.477 |
| Relabel contrast (moved) | 4 | 4 | +0.009 | [−0.023, +0.042] | +0.000, +0.018, +0.007 |
| Relabel swapped planes (moved) | 7 | 1 | −0.207 | [−0.660, +0.000] | −0.110, −0.243, −0.267 |
| All drops and relabels | 91 | 64 | −0.195 | [−0.689, +0.276] | −0.194, −0.215, −0.176 |

No variant moves fold-0 macro AUC by more than 0.0003, and every interval includes zero (the plane
swap touches zero at its upper bound). The only consistent negative is the plane-swap relabel of
one study: these checkpoints were trained with that study's series in their geometric slots, so a
relabel is off-distribution for them; whether a model trained on relabelled slots gains is
untested. The proposed drops free 0 to 1 triplets on fold 0; the two contrast relabels that fill a
missing fluid slot add 20 triplets. Caveat: checkpoints were trained with every flagged series
present, so removal measures reliance, not the value of training without them.

## Proposed rule: series filtering and cleaning

Superseded by the [revised rule](#labelled-findings-in-off-knee-series) and the
[implemented rule](#implemented-rule-2026-09-17); kept as the design record. Hypothesis, based on
the measurements above. Headers only: every signal comes from tags the loader
already reads (`ImageOrientationPatient`, `ImagePositionPatient`, `PixelSpacing`, rows/columns),
except the contrast relabel, which needs `ScanOptions`, `ScanningSequence`, `SequenceVariant`,
`SequenceName`, TE/TR/TI, `SeriesDescription` (one extra header per series, no pixel decoding) and
a cheap intensity check on three decoded slices.

1. **Drop far series** (precision 11/11, recall 11/14): in a study with an accepted robust centre,
   drop a series whose centre projects more than 15 mm outside its slab, more than 20 mm outside
   its field of view, or whose slab centre is more than 90 mm away, unless its pixel spacing
   exceeds 1 mm.
2. **Other knee:** split studies whose series volume centres spread more than 150 mm in x. Drop a
   side that is a single-plane stray (1 series). For bilateral exams (both sides with at least two
   planes), keep both sides for now: the labels describe both knees; sampling each side around its
   own centre is a separate change.
3. **Relabel contrast** (12 series, precision 12/12): header fat suppression without T1 timing plus
   intensity p > 0.5 in a T1/PD slot → the same plane's fluid slot; no fat-suppression evidence
   (with headers present) plus intensity p < 0.5 in a fluid slot → the T1/PD slot, following the
   host's T2 convention.
4. **Do not relabel planes** for now: the only reliable case is one study (7 series), and moving it
   cost −0.21 ×1000 on all three current checkpoints. Revisit only with a retrained model.
5. **Keep** T1 FS, zero-TE, low-resolution, patellar, thin-slab and oblique series (reviewed as
   target knee; their removal effects are in the table above).
6. **Empty slots:** a filter can empty a slot only for the far rule (3 studies in training). The
   slot then counts as missing; with budget reallocation, the missing-type fallback applies.
   A study left with no series is impossible under this rule (every flagged study keeps knee
   series).

**Interactions.** Filtering runs before `estimate_reference`, so a dropped far series also stops
spoiling the least-squares centre; this overlaps with the robust centre, which would otherwise drop
91 valid patellar series as outliers. Reallocation should treat the T1/PD slots as
non-fat-suppressed (19% T2) and fill missing types after filtering. The tie-break ranking applies
within each cleaned slot; relabelled series join the target slot's ranking. The filter changes the
preprocessing identity (series membership and slot per study) and needs a fresh fold-0 run; at
Kaggle inference it must run on the same tags and thresholds.

## Follow-up measurements (2026-09-17)

Measured, CPU only. Script: `followup_geometry.py` (`041bd16:docs/history/series-and-sampling-sources/series-filtering/followup_geometry.py.txt`), which reruns the
loader's indexing. Tables: `outputs/analysis/series-filtering-20260917/followup/`
(`reference_after_drops.parquet`, `reflection_eligibility_series.parquet`,
`items_1_2_summary.json`).

### Centre rescue by the proposed drops

Rules 1–2 drop 12 series in 8 training studies (2 on fold 0). The loader's own
`estimate_reference` was rerun on the kept series of those 8 studies and of every rejected study.
**The drops rescue no study on net.**

| Scope | Rejected before | Rejected after rules 1–2 | Rescued | Newly rejected |
|---|---:|---:|---:|---:|
| All train (4,407) | 178 (170 RMS, 8 residual) | 178 (170 RMS, 8 residual) | 1 | 1 |
| Fold 0 (904) | 31 (RMS) | 31 (RMS) | 0 | 0 |

- Rescued: `…93672103` (fold 2). Dropping its five ankle series lowers the RMS from 42.05 to
  0.48 mm.
- Newly rejected: `…44943102` (fold 1, thigh tumour). Dropping its thigh-level axial raises the
  RMS from 9.01 to 10.87 mm, above the 10 mm limit.
- The 6 other changed studies were already accepted. Their RMS falls or stays the same; on fold 0,
  `…77308505` goes from 6.40 to 2.99 mm and `…42012883` from 6.72 to 3.57 mm.
- No fold-0 study is rescued, so the checkpoint re-scoring was not run. The existing
  "Drop far + stray" row above already re-scores these two fold-0 studies (−0.004 ×1000).

Rule 1 cannot reach the remaining rejected studies. Of the 178, 47 have no accepted robust fit,
and the other 131 have no far series.

**Range-gated robust centre.** This fit repeatedly drops the series with the largest residual,
but only when that residual exceeds 20 mm and the centre projects outside the series' slab or
field of view (`outside_range_mm` or `fov_outside_mm` > 0). After each drop it refits. A study is
accepted under the loader's condition and RMS ≤ 10 mm checks. The "strict" variant also keeps the
loader's 20 mm residual cap on every retained series. The fit was applied after rules 1–2:

| Scope | Still rejected | 20 mm robust fit accepts | Range-gated accepts | Strict accepts | Series dropped (range-gated) |
|---|---:|---:|---:|---:|---:|
| All train | 178 | 131 | 57 | 52 | 69 |
| Fold 0 | 31 | 21 | 10 | 10 | 14 |

- The 20 mm fit rescues 74 train studies (11 on fold 0) that the range gate leaves rejected.
  Their residual series project inside their own slab and field of view, so the gate never drops
  them.
- The range gate rejects none of the 4,229 currently accepted studies.

### Reflection eligibility on the full data

`reflection_eligibility` (`src/knee_abnormality/data/preprocessing/geometry.py`) was run on the
loader's indexed geometry with each slot's plane. **Every series and study is eligible:**
24,371 of 24,371 training series (5,061 of 5,061 on fold 0) return `eligible`, so no study is
ineligible.

- 2,707 series (636 on fold 0) have more than one distinct native orientation, all within the
  tolerance.
- Not covered: the preparation-time `invalid_selected_pose` rejection, which needs decoded slices.
- Inferred: the plane-swapped study `…34605778` is eligible because its labels follow the scanner
  axes. Its reflection therefore mirrors scanner x, which for that patient is not the anatomical
  left–right axis.

### Labelled findings in off-knee series

Reports and Steven v4 labels of the 8 non-knee studies (measured), plus a rough visual check of
three series (`outputs/analysis/series-filtering-20260917/offknee-review.png`, 27 thumbnails,
`render.py` (`041bd16:docs/history/series-and-sampling-sources/offknee-review/render.py.txt`)):

- The ankle study `…93672103` has a knee-only report and labels; the ankle series carry nothing
  labelled. No non-knee study has a positive Fracture label, and of 28 training reports mentioning
  ankle, hip or shaft terms (rough multilingual search) the only positive Fracture is a proximal
  tibial stress fracture inside the knee images.
- Three off-knee series may show labelled findings: `…52031539` (thigh axial T2 FS, 18.6 mm past
  the slab threshold) visibly shows a large suprapatellar effusion (Effusion 0.95, Synovitis
  0.97); `…77308505` (calf axial, report: leaking Baker's cyst, Baker's 0.96) and `…86735623`
  (thigh axial, ascending Baker's cyst, Baker's 0.99) are inconclusive at thumbnail size.
- Dropping the thigh axial of `…44943102` newly rejects its centre (above), so such series also
  help the fit.

**Revised rule (hypothesis):** exclude far series from the centre fit only and keep thigh and calf
slabs as inputs (the sampler gives them few, knee-nearest slices); drop only clearly different
anatomy (the ankle series, which needs a signal other than distance along the leg, such as the
receive-coil change) and the stray other-knee series.

## Implemented rule (2026-09-17)

Implemented in `src/knee_abnormality/data/preprocessing/filtering.py` and `series.py`, identically
in training, validation and Kaggle inference; the behaviour is described in
[current preparation](../data-preparation/current-preparation.md#series-indexing-guards-and-exclusions).
Preprocessing identity `ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10`.

- **Non-knee coil:** in a study with more than one `ReceiveCoilName` and at least one coil name
  that is not a non-knee part, drop series whose coil name names a non-knee part (ankle, foot,
  wrist, hand, elbow, shoulder, hip, head, neck, breast) and not the knee.
- **Stray other knee:** split studies whose series volume centres spread more than 150 mm in x at
  the largest gap; drop a single-series side when the other side has at least two series.
- **Within-series guard:** keep only instances matching the series' most supported rows, columns,
  pixel spacing and orientation (tolerance 1e-3).
- Missing tags or geometry keep the series; no study loses every series. Far series (thigh, calf),
  patellar, oblique and bilateral series are kept; contrast and plane relabels are not
  implemented (evidence above).

Verification, CPU, all 24,386 series of the 4,407 training and 3 example-test studies (
`index_census.py` (`041bd16:docs/history/series-and-sampling-sources/deterministic-selection/index_census.py.txt`); `outputs/analysis/deterministic-selection-20260917/`
`index_report.json`, `index_series.parquet`, `index_studies.parquet`):

| Check | Result |
|---|---|
| Series dropped | 6: the 5 `SENSE_ANKLE_8_AC` series of `…93672103` (fold 2) and the stray sagittal T1/PD of `…42012883` (fold 0) |
| Body/spine-coil studies (5) and other multi-coil studies (3) | Untouched |
| Instances removed by the guard | 0 |
| Study centres changed | 1: `…93672103`, RMS 42.05 → 0.48 mm, rejected → accepted |
| Example-test studies | Nothing dropped or guarded |
| Volume-centre x against the census | Within 3.0 mm for all series (the census used a slightly different centre); no effect at the 150-mm split |
