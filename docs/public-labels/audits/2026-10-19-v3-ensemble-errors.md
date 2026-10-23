# V3 errors by scanner and report style (2026-10-19)

The strongest remaining opportunities are report-template reliability, lost qualifiers,
and compartment assignment. The earlier v3.2 repairs touched only 3 of the 1,211 cells
with absolute disagreement at least 0.5, and none of the 69 such cells in the worst
well-populated scanner group. The gold-policy diagnostic did not answer this question.

This audit ranks v3 against the three-family image ensemble, groups the errors before
reviewing representative reports, and inspects DICOM images for three ambiguous studies.
This document records the pre-build audit. The subsequent [v3.2 build and evaluation](2026-10-19-v32-labels.md) applied 35 additional silver-cell changes and completed the fixed leaderboard comparison.

## Inputs and ranking

- Baseline: `data/labels/prepared/rebuild-state-v3/labels.csv`, SHA-256
  `d3d8ccccc03479102d6fc2222db74e415eed358666910bf7f45e62d2a77251cb`.
- 4,309 unmasked silver studies, 51,708 observed target cells. Gold and the 40 masked
  silver studies are excluded from error ranking.
- Arithmetic probability mean of ConvNeXt, ResNet50 and NextViT, one out-of-fold
  prediction per family per study. All 15 files have complete, unique and fold-aligned
  coverage of the 4,407 study IDs.
- ConvNeXt and ResNet50 use `gen1` folds 0–4. NextViT uses `gen1` fold 2 and the
  `silver`-named folds 0, 1, 3, 4. All five NextViT effective configs use `rebuild_state_v2`,
  `teacher_table: null` and `teacher_share: 0.0`; the naming difference does not make
  them teacher-trained students. Source revisions/seeds differ and are not claimed identical.
- Primary rank: `abs(v3 - mean_prediction)`. Also retain cross-entropy and each family's
  score. Severe disagreement means absolute error ≥0.5. All three families are on the
  same side of 0.5 for every top-100 cell.
- No exclusion of rule-changed or previously reviewed cells. Old report verdicts are
  secondary triage signals, not truth and not the primary selection gate.
- Scanner means observed DICOM vendor/model/field strength from `study_headers.parquet`.
  Report style comes from the earlier report batches. Neither is a verified institution ID.
  In particular, `en_GE_EXCITE_short` also contains Canon and Toshiba acquisitions; its
  name must not be used as scanner metadata.

Overall mean absolute disagreement is 0.105218; 1,211/51,708 cells (2.342%) are severe.
The ranked top 100 contain 55 old report verdicts `no`, 26 `yes`, nine `silent`, six
`probable` and four `uncertain`. This distribution explains why most extreme disagreements
cannot be solved by trusting either the report or the model automatically.

## Where the errors cluster

Groups below have at least 40 studies. Rates are severe target-cell disagreements, not
estimates of label error prevalence; scanner, pathology mix and reporting style are confounded.

| Scanner / reporting cohort | Signal | Interpretation and review proposed |
|---|---|---|
| Philips Ingenia Ambition X 1.5T, 83 studies | 69/996 cells, 6.93%; ACL 11/83, MCL 10/83, Synovitis 7/83 | Highest overall rate, 2.96× global. ACL/MCL rates are 6.72×/7.63× their target baselines. Review this template family first. |
| Philips Achieva dStream 3T, 136 studies | Medial meniscus 15/136; lateral meniscus 11/136 | Repeated definite tears conflict with images; verify template text versus the imaged compartment instead of blanket tear relabeling. |
| Philips Ingenia 3T, Bulgarian reports, 209 studies | Medial meniscus 24/209, 11.48% | Separate grade terminology, explicit surface contact, and bilateral report assignment. |
| Canon Galan 3T, short English reports, 43 studies | Effusion 5/43, 11.63%; seven low labels opposed by all families ≥0.6 | Repeated terse/default negatives warrant image checks; this is not necessarily a Canon imaging defect. |
| Philips MR 5300 1.5T, 62 studies | Contusion 6/62 and Baker's 6/62 | Mostly short Spanish reports; omitted findings and unsized abnormalities need different treatment from explicit negatives. |

The 83 Ambition X studies are a better first unit of work than an arbitrary next 20
cases per target. Their existing v3.2 repair count is zero. A useful review should include
both disagreement directions and examples that agree with the models to avoid deriving
a scanner-wide correction from selected failures alone.

## Concrete flaws and proposed repairs

Values below are v3 / ensemble probabilities. All examples were unchanged in the earlier 46-change v3.2 artifact at audit completion.
Their applied or unresolved status is recorded in the [v3.2 build](2026-10-19-v32-labels.md#changes-made-in-this-pass). The reviewed candidate CSV includes full study IDs, individual
family scores and source-matched evidence for 17 cells.

| Pattern | Example | Proposed correction |
|---|---|---|
| Old fracture cap misses mixed tags | `batch-27-en_Toshiba [5]`, fracture 0.648 / 0.074; `batch-10-en_GE_EXCITE_short [16]`, 0.617 / 0.103; `batch-42-nl [57]`, 0.639 / 0.168 | F1 uses `only(old_or_healed)`. Adding `avulsion` or `osteochondral` defeats the cap. Represent acuity separately from morphology; lower these explicitly old lesions after checking for a separate acute lesion. |
| Tear keyword overrides surface negation | `batch-39-hr [72]`, medial meniscus 0.674 / 0.116 | The report says horizontal rupture **without opening onto the articular surfaces**. Extract surface contact separately; the bare `tear` answer loses the decisive qualification. |
| Low-grade MCL qualifier disappears | `batch-22-en_Philips_other [53]`, MCL 0.527 / 0.027 | “Healing low grade, grade II MCL sprain” becomes only `grade2_or_partial`. Preserve explicit low severity rather than treating every partial/grade-II injury alike. |
| Contradictory ACL template | `batch-24-en_Siemens [70]`, ACL 0.929 / 0.076 | Body contains `[acute chronic] complete tear`, conclusion says intact ligaments. Detect contradictory clauses and consult images; do not unconditionally prefer either section. |
| Meniscus compartment reversal | `batch-10-en_GE_EXCITE_short [4]`: medial 0.917 / 0.112, lateral 0.046 / 0.921 | Review both targets jointly with anatomy. This case has image support for a side correction; do not swap a whole scanner group. |
| Lost injury context in Dutch reports | `batch-41-nl [17]`, contusion 0.178 / 0.796 | Recent patellar dislocation plus opposing patellar/femoral marrow edema is stronger evidence than an unqualified edema mention. Preserve mechanism and location. Fracture-associated edema in the same location needs separate handling. |
| Depth words and corrupted dimensions | `batch-24-en_Siemens [26]`, PF OA 0.761 / 0.190 | “High grade” conflicts with “less than 50% thickness”; dimensions contain `intact8`. Resolve contradictions explicitly and retain unknown extent. Do not silently turn damaged numerals into guessed measurements. |
| Generic chondropathy becomes a strong positive | `batch-32-es_short_tecnica [38]`, PF OA 0.759 / 0.159 | “Condropatía patelofemoral” supplies neither depth nor area. Revisit source-specific uncertainty calibration; it does not establish a negative or a qualifying positive by itself. |

The mixed-old-fracture screen finds **21 silver studies**, including a case with separate
acute and old findings. Therefore changing `only(old)` to `has(old)` is unsafe without
lesion-level scope. The three clearly old examples above establish the failure without
justifying that global replacement. A fourth high-error German example describes a prior
insufficiency fracture and nearly resolved bruise and remains a temporal-review case.

Malformed `intact`+digit tokens occur in **85 reports**: 39 Achieva dStream 3T, 28 Aera
1.5T and 18 Ambition X 1.5T. This is a report-corruption cohort worth one focused pass.
It is distinct from the mere absence of lesion dimensions; zero/decimal restoration is
not uniquely established by those strings.

Missingness also clusters. Among low labels opposed by all three families ≥0.6, the
short Spanish reports have 17 silent Synovitis, 13 silent Contusion and 10 silent Effusion
cells; Ambition X has 12 silent Synovitis cells. Treat report silence as missing evidence
and validate its calibration per reporting cohort rather than raising every cell to the
ensemble score.

## Image review

All rendering used CPU and local DICOMs. Contact sheets preserve native pixel orientation,
sort by physical slice position and use a shared 1st–99.5th percentile window within each
series. Each sheet has a JSON companion with input-file hashes, positions and orientation.
These are provisional dataset-review findings, not new official gold adjudications.

| Case | Images reviewed | Finding and implication |
|---|---|---|
| Short English meniscus reversal, `[4]` | All 24 coronal and 24 sagittal slices; coronal detail 10–15 | Fibula identifies the lateral side. The lateral meniscal body is markedly deficient compared with the medial side, with supporting sagittal abnormality. This supports investigating a paired side correction. |
| Siemens conflicting ACL, `[70]` | All 26 sagittal PD-FS and 16 ACL-oblique slices; sagittal detail 10–15 | Visible ACL-region fiber continuity; no convincing complete midsubstance gap identified. Extensive medial femoral marrow abnormality agrees with the report injury. Images support reducing confidence in the complete-tear clause; partial-tear severity is not definitively graded. |
| Ambition X normal-MCL report, `[22]` | All 30 coronal PD-FS slices; detail 11–16 | Medial-sided soft-tissue/ligament abnormality is visible, but high-grade disrupted fibers remain uncertain. Keep this unresolved rather than copying the 0.849 ensemble score. |

Further high-priority unresolved cases include the bilateral Bulgarian PF report `[21]`,
degenerative medial-meniscus thinning in Croatian `[53]`, and a 28×7×6 mm Baker's cyst
in Croatian `[93]`. Neither vague morphology nor longest cyst dimension alone proves the
competition threshold.

## Evaluation protocol for the next label build (used for v3.2)

The next build was to report both gold diagnostics and the user's fixed leaderboard
correlation panel. The stored protocol is `.artifacts/v3_ensemble_audit/evaluation_contract.json`.

1. Freeze the existing 26-submission panel from
   `outputs/labels-v3-reading/lb-correlation/local_scores.csv`, with membership and tie order
   from the 182-row `inventory.csv`. Reproduce existing v3 scores before adding a new column.
2. Use mean member probabilities, except S164's normalized per-label weights. Keep the six
   submitted-setting prediction sources (S154, S170–S173, S181) and the other 20 training-time
   prediction sources unchanged.
3. Score all 4,349 silver studies, binarizing targets at **≥0.5**. For the 40 masked v3
   studies, retain v2 values in the evaluation targets, matching the existing analysis.
   Report the number of exact-0.5 cells in each new version.
4. Report macro and per-target AUC, Spearman and Kendall against board goodness, over all
   26 and the 20 submissions scoring 0.963–0.965. Bootstrap differences **new−v3**, with
   new−v2 secondary, using 2,000 submission resamples and seed 0.
5. Carry forward all limitations: two clean/eight mildly leaky/16 leaky submissions;
   v2 training ancestry; 20 inference-setting mismatches; undocumented interpretation of
   within-score ordering; duplicate predictions (S103/S104/S105 and S140/S141); narrow
   score range; fragile per-label correlations and the exact-0.5 convention. Do not use
   the panel to select individual cell changes or tune label values repeatedly.

The subsequent build completed this protocol; see [current v3.2 results](2026-10-19-v32-labels.md#kaggle-submission-correlation). The audit itself remains a record of pre-build candidates.

## Artifacts and reproduction

Everything new is inside `/home/claudio/src/knee-abnormality-labels-v3`; shared `data/` and
`outputs/` paths were read only.

Private directory: `.artifacts/v3_ensemble_audit/`.

- `ranked_cells.parquet`, `top100.csv`: complete ranking and largest disagreements.
- `scanners.csv`, `scanner_targets.csv`, `vendor_field_targets.csv`,
  `report_style_targets.csv`, `scanner_report_targets.csv`, `verdict_targets.csv`:
  denominator-aware grouped summaries. Small groups remain in files but are not prioritized
  without adequate support.
- `report_conflict_candidates.csv`: 164 cells with gap ≥0.4 and conflicting old report verdict.
- `reviewed_candidates.csv`: 17 explicitly reviewed candidates and dispositions.
- `mixed_old_fracture_candidates.parquet`: 21 mixed-tag temporal-review cases.
- `images/`, `image_review.json`: five complete series sheets, three detail sheets and findings.
- `provenance.json`, `evaluation_contract.json`: pinned inputs and next-build evaluation protocol.

With a fresh private destination:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -B -m scripts.adhoc.audit_v3_ensemble_errors --output-dir <fresh-private-directory>
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -B -m scripts.adhoc.render_label_audit_series --series <DICOM-series-directory> --output <fresh-private.png>
```

The ranking rebuild checks study uniqueness, fold coverage and original report identity.
Reviewed evidence spans were matched after whitespace normalization. Static lint and
script-index checks passed. No tests, training or GPU jobs were run.
