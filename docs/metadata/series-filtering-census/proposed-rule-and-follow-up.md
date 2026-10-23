# Series filtering census: proposed rule and follow-up measurements

## Proposed rule: series filtering and cleaning

Superseded by the [revised rule](#labelled-findings-in-off-knee-series) and the
[implemented rule](implemented-rule.md#implemented-rule-2026-09-17); kept as the design record. Hypothesis, based on
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
loader's indexing. Tables: `outputs/analysis/s/series-filtering-20260917/followup/`
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
three series (`outputs/analysis/s/series-filtering-20260917/offknee-review.png`, 27 thumbnails,
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
