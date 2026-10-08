# Ad-hoc investigations

One-off investigations, kept as the record behind their findings. They are frozen once the
investigation finishes and may no longer run against current code; reproduce one from the
commit its findings name, or promote it to `tools/` if it becomes a repeated operation. The
findings doc holds the question, method, command and result.

## Duplicates and series cleaning

| Script | Question | Findings |
|---|---|---|
| [audit_study_overlap.py](audit_study_overlap.py) | Do repeated studies hide behind different IDs (geometry, near-duplicate reports, decoded slices)? | [Study overlap](../../docs/metadata/study-overlap-investigation.md) |
| [audit_repeated_series.py](audit_repeated_series.py) | Which series repeat pixel-for-pixel within or across studies? | [Repeated series](../../docs/metadata/repeated-series.md) |
| [simulate_online_series_dedup.py](simulate_online_series_dedup.py) | How accurate and costly is a per-study online duplicate-series filter? | [Online detection](../../docs/metadata/repeated-series.md#online-detection-per-study) |
| [score_series_cleaning_rules.py](score_series_cleaning_rules.py) | What do candidate series-cleaning rules change in fold-0 predictions? | [Series-cleaning rules](../../docs/validation/results/2026-09-22-series-cleaning-rules.md) |
| [census_series_deletions.py](census_series_deletions.py) | Do the superseded and exact-copy rules delete series that are not the same anatomy or not full copies? | [Deletion-identity census](../../docs/metadata/series-filtering-census.md#deletion-identity-census-2026-10-05) |

## Metadata and reports

| Script | Question | Findings |
|---|---|---|
| [audit_metadata.py](audit_metadata.py) | What do the official CSVs contain (schemas, coverage, report duplicates, gold labels)? | [Official CSV audit](../../docs/metadata/official-csv-audit.md) |
| [normalize_reports.py](normalize_reports.py) | How many distinct reports remain after multilingual normalization? | [Report normalization](../../docs/metadata/report-normalization.md) |
| [infer_training_sites.py](infer_training_sites.py) | Which report languages and sites does training contain, and are the officially listed Italian and Mongolian sites absent? (needs `--with langdetect`) | [Sites and test composition](../../docs/metadata/sites-and-test-composition.md) |
| [audit_knee_laterality.py](audit_knee_laterality.py) | What explicit and textual knee-side evidence do the DICOMs carry? | [Knee-side metadata audit](../../docs/metadata/2026-09-14-knee-side-metadata-audit.md) |
| [audit_knee_side_geometry.py](audit_knee_side_geometry.py) | Can physical geometry alone resolve the knee side? | [Knee-side coordinates](../../docs/metadata/knee-side-coordinate-investigation.md) |
| [census_knee_side_and_centre.py](census_knee_side_and_centre.py) | How is each study's knee side and centre resolved, and do coordinate sides agree with reports? | [Side and centre census](../../docs/validation/results/2026-10-05-code-audit-measurements.md#f04-and-f05-knee-side-and-centre-census) |
| [census_centre_slab_guard.py](census_centre_slab_guard.py) | Does the fitted knee centre project inside each cleaned training series' acquired slab, how far does it miss along the slice normal, and what does the sampler then do? | [Knee side and centre census](../../docs/validation/results/2026-10-05-code-audit-measurements.md#f04-and-f05-knee-side-and-centre-census) |
| [census_reflection_tilt.py](census_reflection_tilt.py) | How far is each cleaned training series tilted from the patient axes that the reflection augmentation and reflection TTA assume, by plane and eligibility? | [Reflection TTA detail coordinates](../../docs/validation/results/2026-10-05-code-audit-measurements.md#f01-reflection-tta-detail-coordinates) |
| [measure_header_window.py](measure_header_window.py) | Does the DICOM header window of a series' slices agree with the pixel-percentile window of its ×1 validation bag, by presence, consistency and vendor? | [Bag-dependent normalization](../../docs/validation/results/2026-10-05-code-audit-measurements.md#f06-bag-dependent-normalization) |
| [measure_slice_budget_shortfall.py](measure_slice_budget_shortfall.py) | How many triplets does the deterministic selection deliver against each study's per-view budget at ×1, ×4 and ×6, and how much of the shortfall could the other views refill? | [Code-audit measurements](../../docs/validation/results/2026-10-05-code-audit-measurements.md) (measured for the audit; no section of its own) |
| [measure_window_estimators.py](measure_window_estimators.py) | Which intensity-window estimator computed from a bag's own slices moves least between bags (validation at ×1, ×4 and ×6, two training epochs at ×1)? | [Bag-dependent normalization](../../docs/validation/results/2026-10-05-code-audit-measurements.md#f06-bag-dependent-normalization) |

## Sampling and preprocessing

| Script | Question | Findings |
|---|---|---|
| [measure_bag_normalization_drift.py](measure_bag_normalization_drift.py) | How much do a series' intensity window and shared pixels change with the bag (×1, ×4, ×6, two training epochs)? | [Bag-dependent normalization](../../docs/validation/results/2026-10-05-code-audit-measurements.md#f06-bag-dependent-normalization) |
| [analyze_physical_sampling.py](analyze_physical_sampling.py) | How does production physical sampling and its fallback behave on the geometry cohort? | [Physical sampling](../../docs/data-preparation/physical-sampling/implementation-verification.md) |
| [audit_sampling_protocols.py](audit_sampling_protocols.py) | What sequence metadata separates repeated plane-contrast groups? | [Sampling cohort and rules](../../docs/data-preparation/physical-sampling/sampling-analysis-cohort-and-rules.md) |
| [analyze_study_center_extents.py](analyze_study_center_extents.py) | How much of each plane lies on either side of the fitted study centre? | [Accurate sampling centres](../../docs/data-preparation/accurate-sampling-centers.md#cohort-wide-center-fit-and-plane-extents) |
| [check_narrow_field_fit.py](check_narrow_field_fit.py) | What does an image-centred 120 or 130 mm field cut (tissue depth per side, tibia below the joint line, overlays)? | [What a narrow field cuts](../../docs/research/input-resolution.md#what-a-narrow-field-cuts) |
| [inspect_plane_fluid_windows.py](inspect_plane_fluid_windows.py) | Which labelled anatomy and peripheral findings remain within the production 84/93 mm detail windows and 120/130/140 mm axial fields? Run against the archived `detail_84_93` source. | [Plane/fluid spacing review](../../docs/data-preparation/plane-fluid-spacing-review.md#images-and-disease-evidence) |
| [measure_series_type_resolution.py](measure_series_type_resolution.py) | Per series type (plane by fluid sensitivity), what is the acquired in-plane spacing distribution, the share of series fully resolved at each candidate spacing, the stored field and the padding each window width brings? Reads the acquisition-header audit. | `outputs/analysis/view-spacing-20261002/report.md` |
| [measure_view_window_cut.py](measure_view_window_cut.py), [summarize_view_window_cut.py](summarize_view_window_cut.py) | Per series type, how much tissue do image-centred and rule-placed windows of 70 to 160 mm leave outside on each side, how much height stays above and below the joint line, and how far is the knee from the image centre? Every training series, three slices each; run the first with `PYTHONPATH` set to the archived `detail_84_93` source. | `outputs/analysis/view-spacing-20261002/report.md` |
| [measure_slice_capacity.py](measure_slice_capacity.py) | Per series type, how many distinct slice positions can the sampler's candidate pool supply per study against the per-type triplet budget at multipliers 1, 2 and 4, how wide is the pool and how far apart are the picked slices? Every training study, headers only. | `outputs/analysis/slice-budget-20261002/` |
| [build_slice_budget_kaggle_run.py](build_slice_budget_kaggle_run.py), [slice_budget_kaggle/](slice_budget_kaggle/) | How far does each label's fold-0 AUC move when one series type's per-study triplet budget is halved or doubled on a trained model, every other type unchanged? Builds the private inputs dataset and the Kaggle notebook; `slice_budget_kaggle/` holds the code the notebook runs against an attached bundle (a `sitecustomize` patch of `KneeStudyDataset._study_budgets` and the thirteen-pass driver), not importable modules. | `outputs/analysis/slice-budget-20261002/sensitivity/` |
| [analyze_slice_budget_sensitivity.py](analyze_slice_budget_sensitivity.py) | From the thirteen prediction files, what are the per-label and macro AUC deltas of each halved or doubled series type, their paired report-group bootstrap intervals, the same per realised triplet changed, and does the Kaggle baseline reproduce the local validation metrics? | `outputs/analysis/slice-budget-20261002/sensitivity/` |
| [measure_intensity_bound_stability.py](measure_intensity_bound_stability.py) | How far do intensity clipping bounds move between draws and triplets? | [Preprocessing profile](../../docs/data-preparation/preprocessing-implementation-and-profile.md) |

## Loading speed and profiling

| Script | Question | Findings |
|---|---|---|
| [profile_training.py](profile_training.py) | Where does preprocessing spend time, and what batch fits? | [Preprocessing profile](../../docs/data-preparation/preprocessing-implementation-and-profile.md) |
| [profile_threaded_loading.py](profile_threaded_loading.py) | Do worker-local slice threads beat process-only loading? | [Threading inside loading](../../docs/data-preparation/threading-inside-dataset-loading.md) |
| [benchmark_output_stride.py](benchmark_output_stride.py) | What does a dilated stride-16 ReXNet cost in training time and memory? | [Dilated stride 16](../../docs/research/multiscale-features-knee-mri.md#a-dilated-stride-16-backbone-costs-three-times-the-backbone) |
| [compare_header_implementations.py](compare_header_implementations.py) | Does the native header parser speed up the training dataloader? | [Loading attribution](../../docs/backlog/loading-speedup/attribution.md#only-inference-benefits) |
| [probe_kaggle_loading.py](probe_kaggle_loading.py), [kaggle_loading_probes/](kaggle_loading_probes/) | How fast is each proposed loading speedup on Kaggle CPUs? The probes are code blocks run in Kaggle kernels, not importable modules. | [Measuring](../../docs/backlog/loading-speedup/measuring.md) |

## Public labels

| Script | Question | Findings |
|---|---|---|
| [audit_label_sources.py](audit_label_sources.py) | Are any public label sources trash? Blind report audit of each source's strong disagreements with the other families | [Source audit result](../../docs/public-labels/label-rebuild.md#source-audit-result-2026-09-28) |
| [audit_public_labels.py](audit_public_labels.py) | How do the Steven, Pilkwang and lixin labels compare with gold? | [Public-label audit](../../docs/public-labels/public-label-audit-overview.md) |
| [compare_public_labels.py](compare_public_labels.py) | How do the 2026-09-13 releases compare with Steven v4? | [New public labels](../../docs/public-labels/2026-09-13-new-public-label-comparison.md) |
| [screen_public_label_tables.py](screen_public_label_tables.py) | Which 2026-09-21 snapshots are usable, and does strict MCL consensus help? | [Rescreen](../../docs/public-labels/2026-09-21-public-label-rescreen.md) |
| [screen_public_label_additions.py](screen_public_label_additions.py) | Do the tables downloaded on 2026-09-22 beat Steven v4 on gold? | [Additions](../../docs/public-labels/2026-09-22-public-label-additions.md) |
| [analyze_label_combination.py](analyze_label_combination.py) | What evidence chose the combined_v1 targets? | [Combined labels](../../docs/public-labels/combined-labels.md#how-the-targets-were-chosen) |
| [check_report_attributes.py](check_report_attributes.py) | What do the report attributes in public tables add? | [Report attributes](../../docs/public-labels/report-attributes.md) |
| [score_jevk5_report_attributes.py](score_jevk5_report_attributes.py) | What structured facts (grades, hedging, silence, location, extra findings) can JevK5 read from every training report? Runs with a `transformers`/`flash-linear-attention` overlay; see the findings doc | [Report attribute extraction](../../docs/ongoing/report-attribute-extraction.md) |
| [analyze_jevk5_report_attributes.py](analyze_jevk5_report_attributes.py) | Where are the extracted answers unreliable (dead options, order bias, parent/child, language), and do the training targets or the official thresholds match gold? | [v3 full-run analysis](../../docs/ongoing/report-attribute-extraction.md#v3-full-run-analysis-2026-09-25) |
| [prepare_grade_aware_labels.py](prepare_grade_aware_labels.py) | Does a severity-aware consensus rule (rejected combined_v2) help on gold? | [Consensus-mean rule](../../docs/public-labels/report-attributes.md#consensus-mean-rule-tested-on-gold) |
| [prepare_tonylin_labels.py](prepare_tonylin_labels.py) | Rebuilds the Tonylin soft columns of the finished label-source experiment | [Tonylin soft labels](../../docs/modeling/supervision.md#training-with-tonylin-soft-labels) |

## Validation and scoring

| Script | Question | Findings |
|---|---|---|
| [score_tta_detail_coordinates.py](score_tta_detail_coordinates.py) | What does mirroring the detail-to-wide map under reflection TTA change, and does it match the training reflection? | [Reflection TTA detail coordinates](../../docs/validation/results/2026-10-05-code-audit-measurements.md#f01-reflection-tta-detail-coordinates) |
| [compare_local_and_public_scores.py](compare_local_and_public_scores.py) | Which local metric tracks the public score across comparable submissions? | [Local and public metrics](../../docs/validation/results/2026-09-18-local-public-metric-comparison.md) |
| [rescore_campaign_comparisons.py](rescore_campaign_comparisons.py) | Which autoresearch verdicts change when past comparisons are re-scored against another frozen label set? | [2026-09-28 campaign re-scoring](../../docs/validation/results/2026-09-28-campaign-rescoring.md) |
| [correlate_submission_metrics.py](correlate_submission_metrics.py) | Which local metric and label set track the public score across S23–S77, and do fixed labels judge label changes? | [2026-09-28 update](../../docs/validation/results/2026-09-18-local-public-metric-comparison.md#2026-09-28-update) |
| [correlate_grouped_auc.py](correlate_grouped_auc.py) | Does any AUC averaged over site, scanner group, scanner model, vendor or field strength track the public score better out of sample, and how do the scanner adversarial runs score on it? | [2026-09-29 grouped AUC](../../docs/validation/results/2026-09-18-local-public-metric-comparison.md#2026-09-29-grouped-auc) |
| [correlate_confidence_pair_auc.py](correlate_confidence_pair_auc.py) | Does the same-group, confidence-weighted pair AUC order submissions like the public leaderboard? | [Confidence-weighted pairs](../../docs/validation/results/2026-09-18-local-public-metric-comparison.md#same-group-confidence-weighted-pair-auc) |
| [compare_across_splits.py](compare_across_splits.py) | How does a run on a new split compare with runs on the old one? | [Combined labels first run](../../docs/validation/results/2026-09-21-combined-labels-first-run.md) |
| [score_temperature_sweep.py](score_temperature_sweep.py) | Which attention temperature and image multiplier score best? | [Combined labels first run](../../docs/validation/results/2026-09-21-combined-labels-first-run.md) |
| [simulate_loss_ensembles.py](simulate_loss_ensembles.py) | Would ensembling the losses-campaign runs beat seeds? | [Loss ensembles](../../docs/validation/results/2026-09-21-loss-ensemble-simulation.md) |
| [validate_calibrated_judge.py](validate_calibrated_judge.py) | Can a gold-calibrated soft AUC judge runs without submitting? | [Calibrated judge](../../docs/validation/results/2026-09-22-calibrated-judge.md) |
| [score_validation_auc_variants.py](score_validation_auc_variants.py) | Which of 85 validation AUC variants tracks the public score? | [AUC variants](../../docs/validation/results/2026-09-23-validation-auc-variants.md) |
| [compare_soft_auc.py](compare_soft_auc.py) | Do two runs differ on hard and soft macro AUC? | [Image-blend labels](../../docs/validation/results/2026-09-23-image-blend-labels-result.md#fold-0-result) |

## Model diagnostics

| Script | Question | Findings |
|---|---|---|
| [check_reference_source_parity.py](check_reference_source_parity.py) | Does a run's checkpoint predict bit-identically under its recorded source snapshot and the current source, so a `sources_sha256` difference can be declared a reference exception? | [RadImageNet campaign](../../autoresearch/campaigns/radimagenet/INDEX.md#reference-parity) |
| [estimate_repeat_pair_noise.py](estimate_repeat_pair_noise.py) | How much training noise does a paired Δ carry beyond the report-group bootstrap? | [Literature mechanisms](../../autoresearch/literature-mechanisms.md#3-calibration-against-repeat-pairs-keep-the-rule) |
| [analyze_campaign_screening.py](analyze_campaign_screening.py) | Do early-epoch deltas predict autoresearch verdicts, where do adoptions arrive, which stop rules lose nothing, and do follow-ups beat new families? | [ScientistTwo lessons](../../autoresearch/scientisttwo-lessons.md#measured-against-our-campaign-records), [literature mechanisms](../../autoresearch/literature-mechanisms.md) |
| [score_axial_only.py](score_axial_only.py) | How much does a trained model lose when it sees only one plane's series? | [Single-plane inference](../../docs/validation/results/2026-09-27-single-plane-inference.md) |
| [diagnose_fusion_branches.py](diagnose_fusion_branches.py) | Which branch of the stride-16/stride-32 fusion do predictions use? | [Stride-16 fusion](../../docs/validation/results/2026-09-22-stride16-fusion-result.md#branch-ablation) |
| [score_sibling_masking.py](score_sibling_masking.py) | Do compartment tokens use the sibling knee side? | [Sibling masking](../../docs/validation/results/2026-09-23-sibling-masking.md) |
| [score_tilted_slot_policies.py](score_tilted_slot_policies.py) | How should slots holding tilted and standard series be handled at inference? | [Oblique groups](../../docs/history/oblique-groups.md) |
| [dump_label_attention.py](dump_label_attention.py), [analyze_label_attention.py](analyze_label_attention.py) | Is the model's per-label attention sharp enough to place a crop, which series types does each label use, and how much attention do fixed and per-study 60 and 80 mm windows hold? Written for a plane-readout checkpoint without the anatomical bias; run with `PYTHONPATH` set to the checkpoint's archived `source/src`. | [Smart-crops prior work](../../autoresearch/campaigns/smart-crops/prior-work.md#where-the-model-looks) |
| [dump_input_saliency.py](dump_input_saliency.py), [analyze_input_saliency.py](analyze_input_saliency.py) | Where, at pixel resolution and knee-centred, are the detail and periphery labels sensitive to the input in each series type, and how much of it do smaller windows keep? Run with `PYTHONPATH` set to the checkpoint's archived `source/src`. | [Smart-crops prior work](../../autoresearch/campaigns/smart-crops/prior-work.md#where-the-model-looks) |
| [analyze_attention_by_series_type.py](analyze_attention_by_series_type.py) | From a label-attention dump, what share of each label's attention sits on each series type, how much of a type's attention lies inside windows of 60 to 140 mm, and where is axial attention relative to the knee centre, over all and over positive studies? | `outputs/analysis/view-spacing-20261002/report.md` |
| [tabulate_view_run_labels.py](tabulate_view_run_labels.py) | What did every recorded run that changed the field or view of a series type do to each label's x4 fold-0 AUC against its reference? | `outputs/analysis/view-spacing-20261002/report.md` |
| [fit_view_width_regression.py](fit_view_width_regression.py) | Put into one ridge model per label, across every finished run with one view width per series type, what does narrowing or widening each type do to each label's fold-0 AUC and to the macro, how far do independent paths to the same effect agree, which widths does each label prefer and which cells can the runs not identify? One line per run in `RUNS`; it reads each run's widths back from its frozen source and picks up queued runs when their predictions appear. No arguments needed. | `outputs/analysis/view-regression-20261002/` |

## Data files

[submission_public_scores.json](submission_public_scores.json) holds the public scores of the
comparable submissions read by `compare_local_and_public_scores.py` and
`validate_calibrated_judge.py`.

## Out-of-fold distillation

| Script | Question | Findings |
|---|---|---|
| [sweep_child_temperatures.py](sweep_child_temperatures.py) | What does each run predict on its validation fold at several image multipliers and attention temperatures, with and without reflection TTA? One backbone pass per fold and multiplier. | [Two-family out-of-fold distillation](../../docs/validation/results/2026-10-04-two-family-oof-distillation.md#attention-temperature-multiplier-and-weights) |
| [score_child_blend.py](score_child_blend.py) | From the swept predictions of two families, which temperature per family and multiplier, which family weight and which multiplier give the best pooled out-of-fold blend, with paired report-group bootstrap intervals? | [Two-family out-of-fold distillation](../../docs/validation/results/2026-10-04-two-family-oof-distillation.md#attention-temperature-multiplier-and-weights) |

## Side-routed shared heads

| Script | Question | Findings |
|---|---|---|
| [render_routing_views.py](render_routing_views.py) | Where do a side-routed run's fitted and learned midline, front boundary, routing weights and landmark weights fall on each series type's middle slice, and how is each crop framed in the 120 mm frame? | [Visual inspection and rotation census](../../docs/ongoing/side-routed-heads.md#visual-inspection-and-rotation-census) |
| [census_knee_rotation.py](census_knee_rotation.py) | How far is each training series rotated about the head-foot axis from the scanner's left-right axis, as external or internal rotation of the knee, and do a study's planes agree? | [Visual inspection and rotation census](../../docs/ongoing/side-routed-heads.md#visual-inspection-and-rotation-census) |
| [inter_series_motion/](inter_series_motion/) | How far does the anatomy shift between two series of one study (same-plane pairs and crossing planes, normalised-gradient-field correlation over ±20 mm, with synthetic-shift and triangle-closure validation)? Run `run_pairs.py` then `analyze.py`; `cross_planes.py`, `analyze_cross.py`, `union.py` and `overlays.py` for the crossing planes and the drawings. | [Patient motion between series](../../docs/ongoing/side-routed-heads.md#patient-motion-between-series) |
| [side_routed_session/](side_routed_session/) | The one-off scripts of the side-routed sessions of 2026-10-05 to 07, kept as written (paths and run names are hard-coded; several load a frozen experiment's source first): per-run tables and rankings (`final_labels.py`, `epoch_table.py`, `rank_runs.py`, `x1_x4.py`, `label_history.py`), attention and cut-force analyses and their drawings (`extract_label_attention.py`, `compare_old_reference.py`, `draw_*.py`, `search_*_division.py`), crop-offset and pattern checks at prediction (`shift_off*.py`, `offsets_off.py`, `incumbent_pattern*.py`, `pattern_*.py`, `alignment_sensitivity.py`, `mcl_*.py`), crops cut by the knee's frame (`frame_crops.py`, `frame_rotation_controls.py`, `rotation_check.py`), the femur lean and the bend (`femur_start.py`, `measure_bend*.py`, `bend_gallery.py`, `draw_femur_*.py`), the spread of the v4 report answers (`answer_information.py`), the bundle export from a frozen source (`export_from_source.py`), checkpoint pruning (`prune_*checkpoints.py`) and the builders of the anatomy targets the recipe trains on, run in this order with the segmenter's weights under `~/.cache/knee-segmenters` (`segmenter_convert_sample.py`, `segmenter_mrannotator.py`, `segmenter_export_targets.py`, `segmenter_export_joint_targets.py`, `mask_measurements.py`) | [Side-routed shared heads](../../docs/ongoing/side-routed-heads.md) |
