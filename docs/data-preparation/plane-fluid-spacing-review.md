# Plane and fluid spacing: statistical and image review

## Recommendation

**Keep the measured 84/93.33/120 mm combination as the baseline.** At 224 pixels, its six
settings are:

| Plane | Non-fluid T1/PD field; spacing | Fluid-sensitive field; spacing |
|---|---|---|
| Axial | 120 mm; 0.535714 mm/px | 120 mm; 0.535714 mm/px |
| Coronal | 84 mm; 0.375000 mm/px | 93.333 mm; 0.416667 mm/px |
| Sagittal | 84 mm; 0.375000 mm/px | 93.333 mm; 0.416667 mm/px |

**Outcome of the axial comparisons (2026-10-02): axial stays at 120 mm.** The image review and
population tissue measurements supported 130 mm / 0.580357 mm/px as a coverage-resolution
compromise. The two
[campaign candidates](../../autoresearch/campaigns/smart-crops/seed-ideas-3.yaml), which widen
both axial types, did not confirm it: 130 mm tied the 84/93.33/120 mm recipe (−0.00003, 80%
[−0.0006, +0.0005]) and 140 mm lost 0.0006, and the axial tokens alone read Baker's cyst
equally at 120, 130 and 140 mm (0.935 to 0.936 at ×1). The Baker's cyst cost of the recipe
comes from the sagittal fluid zoom, not from the axial field
([result](../validation/results/2026-10-02-smart-crops-campaign.md)). The contrast-specific
variant (axial non-fluid 120 mm, axial fluid 130 mm) was therefore not run.

This establishes the best measured configuration, not a global
optimum over six independently varied spacings. Coronal and sagittal have not been independently
swept. The campaign also changes crop placement and view encoding; it does not isolate spacing
alone. Equal-sized contrast mixing uses unregistered prepared grids, so narrowing fluid views
also changes the effective augmentation mismatch.

The settings require the actual campaign placement: coronal at the fitted transverse knee centre,
sagittal 7 mm anterior of the fitted centre, both 10 mm below the image centre, clipped into the
120 mm window and native field. Axial stays image-centred. Retain the 120 mm frame (`data.reference_field_mm`, then `data.spacing_mm = 120/224`)
for slice selection; a view's in-plane spacing must not change sampling depth. Wider axial
in-plane coverage cannot restore a superior or inferior slice that was never sampled.

## Measured validation evidence

Snapshot: 2026-10-02. All three runs use `all_5fold_scanner_v2`, fold 0, training seed
1533697213, 912 validation studies, 851 report groups, ten epochs, 26 training inputs and
224-pixel images. Saved predictions are evaluated at ×4. Intervals use 2,000 paired report-group
draws with seed 20260914. Changes below are point-estimate differences, not bootstrap means.

| Comparison | Macro AUC change | Central 80% interval | 95% interval | Soft macro change |
|---|---:|---|---|---:|
| T1/PD 84 vs all-series 120 | +0.001252 | [+0.000052, +0.002476] | [−0.000613, +0.003153] | +0.000450 |
| Add fluid 93.33 vs T1/PD 84 | +0.000963 | [−0.000023, +0.001967] | [−0.000564, +0.002565] | +0.000859 |
| Complete 84/93.33 combination vs all-series 120 | +0.002214 | [+0.000745, +0.003669] | [−0.000050, +0.004495] | +0.001310 |

The complete combination scores **0.932880** locally versus 0.930666 for the 120 mm control.
The campaign records **S93 0.950 public versus S92 0.947**, and explicit user adoption of the
complete combination at `user-adopt-20261002T073318Z-9ed20e99`. This public result is taken from
that decision record, not a new Kaggle query. The fluid increment originally missed the local
adoption rule; the public result and user decision supersede that automatic rejection.

These intervals exclude training-seed, checkpoint-selection and repeated-search uncertainty.
Per-label intervals are exploratory and unadjusted for twelve labels. The 95% intervals do not
establish a decisive population-wide gain. Do not select six independent settings by combining
the best per-label changes from different runs.

| Label | T1/PD 84 minus 120 | Additional fluid 93.33 minus fluid 120 |
|---|---:|---:|
| ACL | +0.00197 | +0.00152 |
| MCL | −0.00077 | −0.00147 |
| Medial meniscus | +0.00198 | −0.00017 |
| Lateral meniscus | −0.00203 | +0.00296 |
| Medial OA | +0.00038 | +0.00185 |
| Lateral OA | −0.00173 | +0.00363 |
| PF OA | −0.00175 | +0.00287 |
| Effusion | +0.00182 | −0.00020 |
| Synovitis | +0.00356 | +0.00540 |
| Baker's cyst | +0.00222 | −0.00620 |
| Contusion | +0.00158 | −0.00060 |
| Fracture | +0.00779 | +0.00195 |

Baker's is the largest adverse change after fluid narrowing; its 95% interval is
[−0.01679, +0.00180], so loss is plausible, not independently established. Synovitis is the only
fluid-increment label whose unadjusted 95% interval excludes zero, [+0.00035, +0.01078].

## Acquired detail and tissue coverage

The acquisition audit covers all 24,371 training series. Percentages below are conditional on a
usable acquisition matrix, not stored pixel spacing. Missing matrices affect 0–10% of a type;
the CSV also records missing-data bounds and a stricter acquisition-quality subset.

| Type | At 84 mm | At 93.33 mm | At 120 mm | At 130 mm | At 140 mm |
|---|---:|---:|---:|---:|---:|
| Axial non-fluid | 76.0% | 65.7% | 11.1% | 8.2% | 6.1% |
| Axial fluid | 93.0% | 90.8% | 35.4% | 22.1% | 17.7% |
| Coronal non-fluid | 91.5% | 75.6% | 17.9% | 14.0% | 10.0% |
| Coronal fluid | 94.0% | 90.9% | 37.3% | 22.0% | 17.3% |
| Sagittal non-fluid | 89.8% | 73.1% | 23.2% | 18.2% | 14.6% |
| Sagittal fluid | 92.2% | 89.2% | 43.5% | 29.4% | 21.1% |

Each entry is the share whose finer acquired axis is no finer than the output spacing, a header
resolution proxy rather than a guarantee of retained diagnostic signal. The measured detail
settings already reach about 90%; going smaller purchases little acquisition fidelity while
cutting more anatomy. Larger detail fields rapidly sacrifice that fidelity.

The full-cohort window probe processed 24,128 series after 243 cleaning exclusions, at the 25th,
50th and 75th slice ranks. It uses a heuristic tissue mask, not organ or lesion segmentation.
For 4,588 usable **axial-fluid middle slices**:

| Field | Posterior tissue cut >10 mm | Posterior cut >20 mm | Anterior cut >10 mm |
|---|---:|---:|---:|
| 120 mm | 41.9% | 7.3% | 16.2% |
| 130 mm | 21.7% | 1.5% | 5.8% |
| 140 mm | 7.3% | 0.5% | 1.5% |

130 mm reduces large posterior clipping substantially for 8.3% coarser pixels than 120 mm.
140 mm buys another 5 mm per side for a further 7.7% coarsening; it is a legitimate already-queued
comparison, not an established improvement. Axial fluid carries 47.4% of Baker's attention in
the older 140 mm probe, and 70.5% on positive studies, versus 1.7% / 0.5% for axial non-fluid.
Attention supports choosing the series type to investigate; it is not lesion localization and
has not been remeasured on S93.

The sagittal rule deliberately spends field on the anterior joint: at 93.33 mm, the median
posterior skin-to-window cut is 35.1 mm and 97.1% exceed 20 mm. This is not a 97.1% cyst-loss
rate. Similarly, coronal 93.33 mm clips >20 mm of lateral/medial tissue in 29.0%/42.6% of series,
but tissue depth does not identify the ligament or meniscus boundary.

Do not turn the joint-margin CSV into a precise disease-coverage guarantee. Its independent
coronal/sagittal joint-line estimates agree within 10 mm in only 72.2% of 4,075 paired studies;
cross-series consensus reduces noise but does not validate the landmark.

## Images and disease evidence

The reproducible review drew exact crop boundaries through the archived production
`record_transform` and `detail_position` functions. It generated 41 series sheets from eight
studies: two gold-labelled multi-finding cases, two previously reported Baker's crop failures,
and four seeded cases with an axial non-fluid series (seed 20261002). This is purposeful
coverage of failure modes, not a prevalence sample. All six middle-slice overview sheets and
seven selected nine-slice sheets were inspected; not every slice in every generated stack was
manually reviewed. Study labels and reports are case-selection evidence, not pixel annotations.

Artifacts: `outputs/analysis/plane-fluid-review-20261002/`. Study suffixes below resolve to full
study/series IDs, slice indices and crop offsets in `manifest.json`; reports are in
`case_reports.csv`.

| Disease evidence | What is inside the reviewed detail crops | What remains vulnerable |
|---|---|---|
| ACL | Central intercondylar/cruciate region on sagittal 84/93.33 | This establishes anatomical inclusion, not that every tear is visible |
| Medial/lateral meniscal tears | Central joint line and meniscal regions in most reviewed coronal/sagittal windows | Outer compartment clipping in an off-centre case; small/extruded tears need lesion annotations to certify coverage |
| Medial/lateral OA | Articular surfaces and adjacent subchondral signal, notably case …21482192 | Outer compartment and osteophyte margins can be cut |
| PF OA | Patellofemoral articulation in axial 120/130; much of the sagittal articular region remains | Sagittal detail excludes the superior patella/recess in some slices |
| MCL | Proximal medial-side abnormal signal remains in trauma case …55725945 | Entire ligament length and distal attachment are not guaranteed by 84/93.33 mm |
| Effusion | Fluid remains around the joint and in axial views in the large-effusion cases | Sagittal 93.33 clips superior suprapatellar fluid; widening axial in-plane does not extend its sampled slice range |
| Synovitis | Some synovial/joint-fluid regions remain; local AUC increases | These non-contrast images and report labels do not separately localize synovitis well enough to claim complete coverage |
| Baker's cyst | Axial fluid retains the posterior collection that sagittal detail removes | 120 mm axial cuts part of a collection; 130/140 recover increasing amounts |
| Contusion | Much of the proximal tibial marrow abnormality in …55725945 remains in sagittal 93.33 | Its inferior extent is truncated; coronal placement also loses part of an outer compartment |
| Fracture | Reported medial femoral condyle lesion and adjacent abnormal marrow in …21482192 lie within coronal 93.33 | This says nothing about distal/metaphyseal or fibular lesions elsewhere in the cohort |

Concrete inspected examples:

- **…93267826:** the report describes a 4.5 cm craniocaudal Baker's cyst. In
  `93267826_sagittal_fs1.png`, slice 17/40 places most of the bright posterior collection outside
  93.33 mm and part beyond 112/120 mm. In `93267826_axial_fs1.png`, slices 13/32 and 17/32 show
  axial 120 cutting its posterior edge; 130 recovers most and 140 more fully contains it.
- **…11059491:** the report describes a ruptured Baker's cyst and fluid along the MCL. In
  `11059491_axial_fs1.png`, slice 15/36 shows posterior fluid extending through the 120 boundary
  but within the 130 boundary. Other slices retain substantial fluid even at 120.
- **…21482192:** `21482192_coronal_fs1.png`, especially 18/35 and 21/35, retains the reported
  medial-condyle fracture region and adjacent marrow signal in 93.33. The sagittal sheet shows
  that the same window excludes upper suprapatellar fluid and most of the posterior collection.
- **…55725945:** `55725945_coronal_fs1.png`, slices 8/19 and 10/19, cuts through an outer
  tibiofemoral compartment. Geometry is marked trusted; the fluid offset is +32 pixels
  horizontally (about 13.3 mm), while non-fluid is +47 pixels (about 17.6 mm). Its sagittal
  sheet retains the central joint and much of the reported tibial contusion. Spacing alone
  cannot certify safe coverage when placement is off-centre.

The use of axial images to assess the gastrocnemius/semimembranosus anatomy associated with
Baker's cyst has direct support in a [210-patient MRI study](https://pubmed.ncbi.nlm.nih.gov/31811432/).
The crop-retention observations above come from this repository's images, not that paper.

## Reproduction and artifact identity

Checkout at review: `ad51d43f6fbb83e13ba077ac611852084841aeab`, with uncommitted investigation
scripts. The rendering script is
[`inspect_plane_fluid_windows.py`](../../scripts/adhoc/inspect_plane_fluid_windows.py).
Run it with `PYTHONPATH` pointing to the archived `source/src` of
`fluid-detail-93-replacement-20261002T061326Z-c4e4d640`, its
`training/effective_config.json` as `--config`, and
`--output-dir outputs/analysis/plane-fluid-review-20261002`.

The separate population probes are indexed in [ad-hoc investigations](../../scripts/adhoc/INDEX.md):
`measure_series_type_resolution.py`, `measure_view_window_cut.py`,
`summarize_view_window_cut.py` and `analyze_attention_by_series_type.py`. Their artifacts are
under `outputs/analysis/view-spacing-20261002`; the window measurement records its full-cohort
configuration in `window_measurement.json`. Source and input hashes used by this review are in
`outputs/analysis/plane-fluid-review-20261002/review_provenance.json`.

Paired incremental comparisons are the campaign's `comparison/per_label.json` for
`t1pd-detail-84-20261001T220724Z-00bb3343` and
`fluid-detail-93-replacement-20261002T061326Z-c4e4d640`. The direct all-120 comparison was
recomputed with `scripts.tools.compare_fold_runs` and the existing per-label diagnostics;
results, prediction hashes, identities, configurations and copied analysis sources are in
`outputs/analysis/plane-fluid-review-20261002/fluid-detail-vs-120/`.

The comparison permits only name, view policy and output/manifest-path differences, and retains
the campaign's declared `preprocessing_identity` reference exception. No training, inference,
GPU work or test suite was run for this review. Rendering, saved-prediction scoring, targeted
lint and the script-index CLI check completed successfully.
