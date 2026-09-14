# OA and meniscus disagreement audit

Completed on 2026-09-14. The audit reviewed 48 blinded reports and read one
representative DICOM header for every one of the 4,407 available studies.
It used saved predictions only. No GPU queries, pixel decoding, model execution,
training, inference changes or automated tests were performed.

## Findings and decision

Two concrete vulnerabilities emerged: OA report labels can treat explicitly mild
cartilage disease as a confident positive despite the competition's high-grade
endpoint, and the relative performance of the sharing models varies with scanner
manufacturer. Increasing the weight of Philips studies reverses 1B versus
reflection locally, demonstrating a mechanism through which acquisition/case-mix
shift could change model rankings. The audit does not observe Kaggle's hidden
composition or prove that this mechanism caused the public reversal.

The report-label conflicts do not by themselves explain 1B's public loss.
Excluding the 19 conflicting report-derived cells leaves its local advantage
over reflection almost unchanged. Treat OA supervision as a concrete improvement
opportunity and scanner/case-mix sensitivity as a separate generalization concern.
The results do not justify automatically rejecting sharing, relabeling gold
studies, or selecting a model on this disagreement-enriched review sample.

At the time of this audit, the recommended intervention was an OA-only
supervision experiment after a
training-only random report check confirms the same mechanism outside selected
validation disagreements. Preserve meniscus and other targets, official gold,
sampling and model architecture. Represent explicit low-grade disease separately
from insufficient report evidence; do not turn unmentioned or unquantified OA
into a negative. Keep the frozen validation metric and these predetermined
diagnostics separate. No new labels, training run or architecture change was
implemented as part of this audit. The later [transfer research](research/medial-lateral-transfer.md)
prioritizes weak meniscus classifier coupling as the first architecture experiment
and retains OA supervision as a separate factor; no follow-up is scheduled here.

## Review design and limits

Seed **20260916** selected 48 unique validation studies: two disease families ×
four joint-label groups (neither, medial-only, lateral-only, both) × six cases.
Each group includes two cases favoring 1B's ranking over reflection, two favoring
reflection, one with substantial 1B/1C disagreement and one random control.
Selection uses per-study contributions to pairwise AUC comparisons against
opposite-class validation cases, rather than treating absolute probability
changes as ranking improvements. All selected directional cases have the intended
sign. Shared studies are deduplicated before assignment and the final order is
randomized.

The reviewer saw only opaque case IDs and complete reports. Model identities,
predictions, source/frozen labels, selection reasons, study IDs and acquisition
metadata were hidden until all annotations were recorded. The review covers
five targets: both menisci and medial, lateral and PF OA. There are 46
report-derived studies and two gold studies, in English (28), Turkish (7), Greek
(4), Spanish (3), German (2), Croatian (2) and Dutch (2).

This is an assistant's multilingual report-text review, not an independent
radiologist image adjudication. Model blinding reduces one source of bias; it
does not establish clinical truth or erase the selection bias. The sample is
deliberately enriched for disagreements and cannot estimate dataset error rates.

The fixed rubric distinguishes:

- **Positive:** explicit definite meniscal tear or sufficiently extensive
  high-grade cartilage loss/advanced compartment disease.
- **Negative:** explicit normal relevant anatomy or exclusion of the finding.
- **Below threshold:** degeneration without a definite tear, or explicit
  mild/low-grade cartilage disease without qualifying loss.
- **Uncertain:** ambiguity in the finding, extent, severity or compartment,
  conflicting report statements, or insufficient postoperative evidence.
- **Unmentioned:** no interpretable evidence for that target.

The OA rubric follows the host's requirement for an appreciable region, roughly
1 cm or larger, with cartilage loss exceeding half its thickness. Reports saying
only OA, focal fissuring or unspecified thinning may not establish those details.
For meniscus, definite surface-reaching tears or qualifying morphologic defects
are distinguished from intrameniscal degeneration.
[Official annotation criteria](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343).

The annotation lock was created before model results were inspected. Its SHA-256
is `95641564e5350e0195b27f84dccdc2fa6a6c48a13b0e5680af228232cbf140df`.
Annotations were not changed after unblinding.

## Report-label agreement

The table partitions the frozen-positive target cells in the 48 reviewed cases.
Unknown cases are not counted as wrong labels.

| Target | Frozen positives | Text supports positive | Explicit below-threshold finding | Uncertain evidence |
|---|---:|---:|---:|---:|
| Medial meniscus | 26 | 22 | 0 | 4 |
| Lateral meniscus | 17 | 13 | 0 | 4 |
| Medial OA | 19 | 4 | 4 | 11 |
| Lateral OA | 17 | 2 | 8 | 7 |
| PF OA | 23 | 3 | 8 | 12 |

Among the 81 scoreable meniscus study-target cells, the report review agrees
with the frozen binary labels throughout. Degenerative meniscal signal is
already generally distinguished from definite tears in this sample. This is
counterevidence to a broad claim that all label types are equally problematic.

There are **20 OA text/criterion conflicts across 17 studies**: 19 report-derived
cells from 16 studies and one gold cell. Five conflicts occur among four of the
eight stratified random-control cases. The controls are still balanced by
joint-label group and are too few to estimate population frequency.

Examples, using opaque review IDs:

| Case | Report evidence | Frozen target | Interpretation |
|---|---|---:|---|
| C08 | Superficial lateral tibial chondropathy | Lateral OA 0.91 | Explicitly low-grade report evidence receives a confident positive |
| C19 | Low-grade lateral tibial fissuring | Lateral OA 0.91 | Same severity mismatch; the body and conclusion also differ on other compartments |
| C30 | Mild medial/lateral thinning, grade-II chondromalacia | Both OA targets 0.91 | Low-grade disease becomes bilateral positive supervision |
| C40 | Advanced medial loss, early grade-II lateral and PF disease | Medial 0.995; lateral/PF 0.92 | The labels blur the distinction between the severely affected and mildly affected compartments |
| C44 | Grade-I patellar chondromalacia; other cartilage normal | PF OA 0.91 | The mild finding is strongly positive, while the normal compartments remain negative |

C32 is an important counterexample to treating report review as ground truth:
the report describes early PF OA, but the official gold PF label is 1.
The images may support more disease than the report states, or annotation
interpretations may differ. The audit cannot resolve this and does not override
the gold label. Even otherwise plausible report corrections need image-grounded
validation before being treated as clinical truth.

### Source lineage and prediction effects

All 19 conflicting report-derived cells are positive in both Steven v1/v2 and
lixin. Steven assigns **0.80–0.86**; lixin assigns **1.0 in 18 cases and 0.8 in
one**, producing v4 values **0.80–0.93**, usually around 0.91. The v1→v2 synovitis
repair is irrelevant to these OA cells. The mismatch predates blending and is
shared by both sources. Source agreement, distance from 0.5 and high-confidence
filtering therefore do not solve this mechanism.

Some local improvements move predictions toward the disputed supervision:

| Case/target | Reflection | 1B | Original 1C |
|---|---:|---:|---:|
| C08 lateral OA | 0.188 | 0.385 | 0.187 |
| C19 lateral OA | 0.168 | 0.356 | 0.153 |

Other cases move the opposite way, and the issue affects every architecture.
For C40's reported grade-II lateral disease, all five models predict roughly
0.915–0.946. This is compatible with supervision rewarding general degenerative
disease or co-occurrence, but probabilities alone cannot establish which image
feature a model used.

A diagnostic excludes only the 19 conflicting report-derived target cells,
retaining all gold and all other frozen labels. The 1B-minus-reflection macro
advantage changes from **+0.001047 to +0.001038**; the ranking is unchanged.
Thus the specific conflicts found here do not account for the full public
reversal. This post-hoc sensitivity is not a replacement validation score.

The reviewed OA subset has only four clear medial, two clear lateral and three
clear PF positives. Its report-based AUCs, including perfect medial AUC for all
models, are consequently unsuitable for model selection. Full values are kept
in the artifact for transparency rather than presented as new validation results.

## Acquisition inventory and population sensitivity

One header was read per study, preferentially from a coronal fluid-sensitive
fat-suppressed series, then using deterministic series/file order. All 4,407
reads succeeded. Manufacturer and model name are available throughout; field
strength and patient sex are each missing in 238 studies. No patient names,
patient identifiers or dates of birth were collected. Field-strength strings
are normalized numerically so 1.5 and 1.500000 form the same group.

This representative header supports study-level acquisition proxies. It does
not inventory all sequence variations, verify scanner consistency within a
study, or identify a hospital site. Protocol/description fields are retained for
case inspection; no arbitrary protocol-string groups were promoted to findings.

The three largest manufacturer groups contain 842 of 904 validation studies:

| Vendor | Training studies | Validation studies | Training lateral-OA label rate | Validation lateral-OA label rate |
|---|---:|---:|---:|---:|
| GE | 704 | 201 | 27.98% | 24.88% |
| Philips | 1,046 | 255 | 20.36% | 15.29% |
| Siemens | 1,565 | 386 | 30.80% | 31.87% |

These are effective-label rates, not clinically verified disease prevalence.
Vendor differences may reflect patient populations, report conventions, scanners,
label generation or combinations of them. A diagnostic that assigns each
validation study its training group's positive rate, using no pixels, reaches
**0.617 lateral-OA AUC from vendor alone** and **0.631 from vendor plus field
strength**. The analogous meniscus values are much closer to chance. This
establishes metadata/label association, not reliance on that association by the
image model. No inference metadata prior was added.

### Model changes depend on acquisition group

| Vendor | 1B − reflection, lateral OA | 1B − reflection, PF OA | 1B − reflection, all-target macro |
|---|---:|---:|---:|
| GE | −0.000265 | +0.010415 | +0.001656 |
| Philips | **−0.030983** | **−0.020847** | **−0.003986** |
| Siemens | +0.008316 | −0.001423 | +0.001365 |

The Philips lateral-OA decline appears at both field strengths: **−0.046253**
on 117 studies at 1.5T, and **−0.023053** on 138 studies at 3T. There are only
23 and 16 positives respectively, so finer slicing quickly loses precision.

Exploratory paired IID study bootstrap, 2,000 draws with seed 20260917, gives
Philips 95% intervals of **[−0.059884, −0.003534]** for lateral OA and
**[−0.038061, −0.005352]** for PF OA. These are post-hoc subgroup diagnostics
without multiple-comparison adjustment, group-dependence correction or seed
variance. They strengthen the lead but do not establish a causal scanner effect.

### A population change can reverse the local macro ranking

Hypothetical weighting changes the study population while keeping each model's
predictions and the twelve-target metric fixed:

| Philips study weight | Philips population share | 1B − reflection macro AUC |
|---|---:|---:|
| 1×, original | 28.2% | +0.001047 |
| 2× | 44.0% | +0.000136 |
| 4× | 61.1% | **−0.000939** |

This reproduces the direction, not the full 0.006 magnitude, of 1B's public
loss. Original 1C still trails 1B under these weights. Kaggle's manufacturer
mixture is unknown, and vendor is only a proxy for underlying case/report
differences. It would be unjustified to rebalance training to the hypothetical
61.1% or to infer that the hidden cohort contains more Philips studies.

Sigma-40 1C also cautions against rejecting the architecture: on the Philips
subset its macro AUC is **0.875951**, versus reflection **0.872926**, 1B
**0.868940** and original 1C **0.868938**. Sampling/training trajectory changes
can recover performance within the same compartment-attention architecture.

## What this establishes and what comes next

The audit establishes reproducible report/endpoint conflicts, acquisition-linked
label distributions and model-ranking sensitivity to acquisition mixture.
It does not establish Kaggle's hidden frequencies, that a scanner causes the
errors, or that the reviewed report labels should replace image adjudication.

Before another architecture change, inspect a modest random **training-only**
OA report sample with the same evidence rubric. The current review deliberately
selected validation disagreements; a separate sample is needed before
generalizing the supervision finding. If it persists, prepare an OA-only target
candidate that explicitly separates low-grade disease from high-grade depth and
extent, retaining unknown evidence as unknown and preserving gold overrides.
Sigma-40 1C was the competitive control at the time. The later
[reflection/sigma-40 result](validation.md#2026-09-14-reflection-sigma-40-result)
selects the independent-head model for current development.

For the resulting training comparison, predeclare the full-fold metric, vendor
groups, isolated/co-occurring disease subsets and epoch-selection rule.
Preserve a locked report-evidence diagnostic independently of the training-label
rewrite; do not tune extraction until it reproduces these selected cases.
One matched repeat can then assess whether the subgroup changes survive a new
training realization. Additional report-derived folds address sampling variance,
but cannot by themselves resolve an incorrectly specified clinical endpoint.

## Artifacts and reproduction

The [archived command](history/sharing-sources/audit_sharing_disagreements.py.txt)
is preserved with [runtime requirements](history/sharing-sources/README.md).
All evidence is under
[`outputs/analysis/sharing-disagreement-20260914/`](../outputs/analysis/sharing-disagreement-20260914/):

- `blind_reports.json`, `rubric.json`, `selection_key.json`: frozen sample,
  review instructions and separately stored unblinding information.
- `review_batch_01.json` through `review_batch_06.json`, `reviews.json`,
  `review_lock.json`: model-blinded annotations and their locked identity.
- `study_headers.parquet`: representative metadata for all 4,407 studies.
- `unblinded_review.csv`, `source_label_conflicts.csv`: complete evidence,
  original soft labels, model predictions and source lineage.
- `stratified_auc.csv`, `metadata_distributions.csv`, `summary.json`: subgroup
  counts, metrics, exploratory intervals and weighting sensitivity.
- `conflict_exclusion_sensitivity.json`: diagnostic exclusion of the 19
  conflicting report-derived target cells, retaining official gold.
- `manifest.json`, archived preparation/summary source: input identities and
  reproduction source. Summary source records the empty-subgroup handling and
  numeric field-strength normalization added during analysis; the original
  header artifact and locked reviews are preserved.

The historical preparation command, run in the pinned 1C checkout or archived
analysis runtime, is:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked python -m scripts.adhoc.audit_sharing_disagreements --phase prepare --output outputs/analysis/sharing-disagreement-new
```

After recording all blinded annotations as `reviews.json`, summarize with the
same command using `--phase summarize`. Preparation refuses existing outputs;
summarization rejects annotations changed after the review lock. Running the
summary reproduces tabular calculations, not an independent human/clinical
review. Ruff and catalog checks validate static consistency only.
