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

## Metadata and reports

| Script | Question | Findings |
|---|---|---|
| [audit_metadata.py](audit_metadata.py) | What do the official CSVs contain (schemas, coverage, report duplicates, gold labels)? | [Official CSV audit](../../docs/metadata/official-csv-audit.md) |
| [normalize_reports.py](normalize_reports.py) | How many distinct reports remain after multilingual normalization? | [Report normalization](../../docs/metadata/report-normalization.md) |
| [audit_knee_laterality.py](audit_knee_laterality.py) | What explicit and textual knee-side evidence do the DICOMs carry? | [Knee-side metadata audit](../../docs/metadata/2026-09-14-knee-side-metadata-audit.md) |
| [audit_knee_side_geometry.py](audit_knee_side_geometry.py) | Can physical geometry alone resolve the knee side? | [Knee-side coordinates](../../docs/metadata/knee-side-coordinate-investigation.md) |

## Sampling and preprocessing

| Script | Question | Findings |
|---|---|---|
| [analyze_physical_sampling.py](analyze_physical_sampling.py) | How does production physical sampling and its fallback behave on the geometry cohort? | [Physical sampling](../../docs/data-preparation/physical-sampling/implementation-verification.md) |
| [audit_sampling_protocols.py](audit_sampling_protocols.py) | What sequence metadata separates repeated plane-contrast groups? | [Sampling cohort and rules](../../docs/data-preparation/physical-sampling/sampling-analysis-cohort-and-rules.md) |
| [analyze_study_center_extents.py](analyze_study_center_extents.py) | How much of each plane lies on either side of the fitted study centre? | [Accurate sampling centres](../../docs/data-preparation/accurate-sampling-centers.md#cohort-wide-center-fit-and-plane-extents) |
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
| [compare_local_and_public_scores.py](compare_local_and_public_scores.py) | Which local metric tracks the public score across comparable submissions? | [Local and public metrics](../../docs/validation/results/2026-09-18-local-public-metric-comparison.md) |
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

## Data files

[submission_public_scores.json](submission_public_scores.json) holds the public scores of the
comparable submissions read by `compare_local_and_public_scores.py` and
`validate_calibrated_judge.py`.
