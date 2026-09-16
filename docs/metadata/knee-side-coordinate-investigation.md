# Knee-side coordinate investigation

**Absolute coordinates contain strong side information in this dataset.** The
earlier 56.98% result measured explicit tags and description text only; it did
not establish the limits of metadata-based inference. The additional CPU audit
on 2026-09-14 examined positions, directions, physical field-of-view centers,
cross-series agreement, coil names and acquisition metadata. No image-based
method was considered or executed; the user explicitly deferred those methods.

For each sampled image, compute its physical field-of-view center using DICOM
position, row/column direction vectors, pixel spacing and dimensions. Take the
median X within each series, then the median across series for the study.
The measured rule is **positive absolute X → left knee; negative X → right
knee**. This uses coordinates before subtracting the study center. Our current
3D token positions are centered and no longer contain this absolute offset.
The field-of-view center is not an anatomical landmark, and the rule is an
empirical acquisition convention rather than a universal DICOM guarantee.

## Agreement and potential coverage

| Coordinate rule | Studies with consistent explicit tags retained | Agreement with tags |
|---|---:|---:|
| Sign of study median physical center X | 2,151 | 2,102 / 2,151 = 97.72% |
| Additionally require absolute X > 20 mm | 2,081 | 2,065 / 2,081 = 99.23% |
| Additionally require absolute X > 40 mm | 2,070 | 2,054 / 2,070 = 99.23% |
| X > 40 mm in magnitude, every series agrees on sign, center range ≤ 40 mm | 1,347 | 1,339 / 1,347 = 99.41% |

The last row requires absolute study X > 40 mm. The other conditions use series
centers. Thresholds 0/20/40/60 mm and the combined gate are exploratory, with no
independent calibration or model validation. Increasing the margin to 60 mm
does not eliminate remaining disagreements: 1,977/1,993 agree. A large absolute
coordinate is useful evidence, not proof that a conflicting tag is wrong.

Among the 360 text-only assignments, center sign agrees with 359; all 346 with
absolute X > 40 mm agree. These remain comparisons with weak metadata references.

Of the **1,863 training studies with neither tags nor descriptive side evidence**,
**1,757 have absolute center X > 40 mm**; 1,756 also have agreement in sign across
all series. Adding their coordinate candidates to the previous 2,511 assignments
would produce **4,268/4,407 = 96.85% potential coverage**, leaving 106 near-origin
cases plus the 33 earlier tag/text conflicts unresolved. This arithmetic keeps
the earlier assignments; it is not an adopted labeling policy or proof that all
4,268 sides are correct. Additional tag-versus-coordinate disagreements also
require investigation. The stricter combined gate admits only 1,165 previously
unknown studies, giving 83.41% potential combined coverage.

Both example-test studies lacking side metadata have negative centers:
**Toshiba −88.88 mm and GE −93.63 mm**, with consistent signs across their five
series. Both pass the exploratory combined gate and are therefore **candidate
right knees**. The explicitly left Siemens example has center +82.59 mm and
also passes. There is no independent side verification for the two inferred
examples, and no access to hidden-test metadata.

## Coordinate conventions and other metadata clues

Raw manufacturer names distinguish important acquisition conventions that a
single broad GE category concealed. The 37 explicitly tagged `GEHC` cases
are near the coordinate origin; center sign agrees in only 14/37. In contrast,
center sign agrees with 204/205 text-labeled `GE MEDICAL SYSTEMS` cases, and
620/663 previously unknown cases in that group have absolute centers > 40 mm.
The initial sagittal-anchor probe pooled sampled headers directly and agreed
with all 205 text labels; median-per-series aggregation and physical centers
change one near-origin case. These are different measured rules, not numerical
parity checks. Coarse manufacturer grouping would obscure this distinction.

Siemens and Philips supply strong evidence for the absolute-position convention.
Fujifilm, Hitachi and `GEHC` include different near-origin behavior, with small
labeled cohorts. Coil names, patient position and phase-encoding direction were
tabulated as possible convention/context clues, but no universal side rule was
established from them. For example, a coil-name `L` can be a hardware descriptor;
it must not automatically become a left-knee assignment.

Signed obliquity supplies a further metadata-only hypothesis for near-origin
cases. A sagittal orientation component has AUC 0.915 for left versus right in
the 37 `GEHC` cases; other components also correlate with side in small vendor
subsets. These are post-hoc descriptive comparisons across eighteen orientation
components, not validated predictors. They justify a bounded orientation
investigation, not automatic assignments. No classifier was fit.

Training reports were used only as an additional textual cross-check. The same
conservative parser finds a single side in 1,164 reports, both sides in 93 and
neither in 3,150. Among otherwise unknown studies with a single parsed report
side and absolute X > 40 mm, coordinates agree in **334/343 = 97.38%**. This
includes 247/256 GE cases and 56/56 Toshiba cases. The nine disagreements in
that subset are all GE cases. Reports, parsing and copied metadata may themselves
be wrong; these are not anatomical accuracy estimates. Reports are never used
as an inference input, and the public examples/hidden test do not require them.

All studies contain multiple recorded frame-of-reference UIDs across series.
Agreement between coordinates therefore remains an empirical acquisition
heuristic; distinct frames cannot be assumed to establish registered anatomy.
Requiring agreement across series can reject some inconsistencies but does not
prove correctness or remove source-dependent failure modes.

## Evidence and next decision

Coordinates make a metadata-only side estimate substantially more plausible
than the tag audit alone suggested. The resulting abstaining rule is the
[knee-side policy audit](knee-side-policy-audit.md); its thresholds were not
tuned on disease validation. Image-based alternatives remain deferred. No labels were installed, training
started, inference changed or GPU used.

The [reviewed summary](../../outputs/analysis/knee-side-geometry-20260914/reviewed/summary.json),
[study coordinates](../../outputs/analysis/knee-side-geometry-20260914/reviewed/studies.parquet),
[agreement table](../../outputs/analysis/knee-side-geometry-20260914/reviewed/agreement.csv),
[report cross-check](../../outputs/analysis/knee-side-geometry-20260914/reviewed/report_agreement.csv)
and [manifest](../../outputs/analysis/knee-side-geometry-20260914/reviewed/manifest.json)
retain the 73,158 successfully parsed headers, field evidence, sources and hashes.
The script uses only CPU metadata and source text; no pixels or models are read.
Reproduce from the enriched snapshot without another DICOM scan:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python -m scripts.adhoc.audit_knee_side_geometry --enriched-headers outputs/analysis/knee-side-geometry-20260914/headers.parquet --output-dir outputs/analysis/knee-side-geometry-reanalysis
```
