# Side-aware anatomical attention

**Adopted and closed by user decision on 2026-09-21.** The working model now uses
position-rich pooling plus side-aware anatomical attention with two learned
positive family strengths and a fixed 20 mm spatial scale. This is unconditional
model behavior, not an experimental option. `configs/training.yaml` uses seed
2263738815. Appearance-only pooling, fixed-strength-only and learned-width paths
are retired; the original configs and source remain in run artifacts for evidence.

The adopted learned-strength run scores 0.897443 against the new matched-seed,
repaired-data control's 0.895112: gain +0.002331. Its 95% interval includes zero,
and its 80% interval is [+0.000103, +0.004518]. Adoption is a practical
working-recipe decision under the user-approved
[0.001-plus-80%-interval adoption rule](protocol.md#adoption-policy), not statistical confirmation.
The explicit user adoption does not retrospectively alter the older campaigns'
incumbents or verdicts. No confirmation branch or further run is pending here.

The sections below preserve experiment history, including retired flags and launch
commands. They describe archived source, not runnable alternatives in the current
checkout. Start new training with `configs/training.yaml`. Original run configs
and `source_snapshot.zip` files preserve the exact experimental implementations.

## Hypothesis and distinction from prior work

Retain position-rich pooled values and all independent classifier heads. Add a
bounded, anatomically oriented preference to medial/lateral meniscus and OA
attention, so appearance evidence competes within a coarse anatomical frame.
Other targets receive zero added attention bias. There is no auxiliary loss,
classifier sharing, appearance-only pooling, sampling change or new learned parameter.

The [retired attention prior](../docs/validation/results/2026-09-15-attention-prior-result.md)
learned broad Gaussian locations/scales for all targets and stayed nearly flat.
This implementation is an active, fixed, side-conditioned monotonic preference,
with no learned gate that can remain dormant. It also differs from the invalid
anatomical-mirror campaign attempt's zero-initialized shared neural scorer and
coordinate-only side resolver. Its earlier final-inference device-transfer defect
is addressed by including every new input tensor in `MODEL_BATCH_KEYS`.

## Exact implementation

Switch: `model.anatomical_attention_bias`, false for control, true for candidate.

For each token, map its feature-grid anchor back through the existing inverse
augmentation transform to prepared-image pixels, then through the middle slice's
existing patient map to patient LPS X in millimetres. Define:

```text
medial_mm = side_sign * (patient_x_mm - study_center_x_mm)
side_sign = -1 for left knee; +1 for right knee; 0 when unknown
medial attention addition = tanh(medial_mm / 20)
lateral attention addition = -tanh(medial_mm / 20)
```

The 20 mm transition scale and unit maximum logit amplitude are fixed choices
for this experiment, not measured optima or tunable config alternatives. At normal
temperature the bias lies within [-1, 1], permitting at most an exp(2) ratio in
attention preference between opposite extremes before learned appearance scores.
It never removes tokens or makes a compartment inaccessible. Inference attention
temperature scales both the learned scorer and the anatomical term consistently.

The center is the audited median of per-series image centers, not a learned joint
landmark. Patient LPS X is a coarse anatomical approximation; rotation, off-center
acquisitions and overlapping receptive fields remain limitations.

## Side, geometry and reflection

The [audited side policy](../docs/metadata/knee-side-policy-audit.md) is restored from
its archived source with the current geometry import. It resolves explicit laterality
and coordinate consensus, uses description text to detect contradictions, and abstains
on bilateral/conflicting evidence or text alone. It does not infer side from pixels,
labels, reports, identifiers or manufacturer. The old audit's coverage is historical;
coverage of a new training run has not yet been measured.

Both existing header parser paths collect side fields from the headers they already
read. The fast parser decodes side text using the declared DICOM character set.
Per-series side evidence and per-study results are cached with the current indexing;
selection, ordering, filtering, triplet budgets and pixel decoding are unchanged.
Both control and candidate use this same plumbing and repaired dataset.

Unknown side, absent center or invalid triplet geometry yields zero anatomical bias.
Batch metadata carries side reasons, geometry validity, patient maps, signed side
and study center. The inference device-transfer allowlist includes the new tensors.

Training's inverse reflection/affine transform maps each token to its original
content coordinate. Reflected inference composes horizontal reflection into patient
maps for axial/coronal images and reverses their channel dimension for sagittal
triplets. Side and center remain in the original patient frame; no second sign flip
is applied. All three reflected channels retain the existing validity rules.

The encoding diagnostic includes `without_anatomical_bias`; its all-position
intervention removes both positional embeddings and the new bias. Existing
plane-restricted diagnostics are supported but not run. New source identity covers
the side resolver and inference reflection code. Historical checkpoints require
their archived sources.

## Fresh matched control and candidate

- [Control config](../outputs/training/anatomical-attention/control-repeat/input_config.yaml)
- [Candidate config](../outputs/training/anatomical-attention/candidate/input_config.yaml)

Both use seed **2263738815**, fold 0, ten epochs, 26 study triplets, the current
ReXNet and class-balanced BCE recipe. They differ only in name and the new bias
switch. Appearance-only pooling and the rejected adapter are disabled in both.

Train a fresh control from this same source and dataset. Historical control B's
seed matches, but its earlier exclusions and preprocessing do not provide a clean
one-change reference. Keep sources fixed between launches. From the repository
root, the user checks GPU availability and runs the two commands sequentially:

```bash
rtk proxy nvidia-smi
rtk proxy uv run --locked python -m scripts.pipeline.train configs/experiments/anatomical-attention/control.yaml --run-id control &&
rtk proxy uv run --locked python -m scripts.pipeline.train configs/experiments/anatomical-attention/candidate.yaml --run-id candidate
```

After both complete:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python -m scripts.tools.compare_fold_runs \
  --reference outputs/training/anatomical-attention/control \
  --candidate outputs/training/anatomical-attention/candidate \
  --output-dir outputs/analysis/anatomical-attention-seed2263738815 \
  --allowed-config-differences name,model.anatomical_attention_bias
```

Review macro and all per-label AUCs, meniscus/OA family means, exclusive-case
compartment margins, total false positives and cross-talk. A lower cross-talk
fraction alone is not a success. Confirm a promising result with another preselected
matched seed before adoption. A negative result constrains this fixed soft spatial
preference, not every use of knee side or localization.

## Validation and status

Ruff lint/format, syntax parsing of all 14 changed Python files, dependency imports,
configuration/path validation, source fingerprint generation, catalog consistency
and whitespace checks passed. The loaded configs differ only in name and the
anatomical bias switch. All geometry/pooling/IndexedSeries callers were updated.
No tests, training, GPU inference or campaign launches were performed during implementation.
The subsequent user-launched training and measured results are recorded below.
Current metadata coverage has not been re-audited; the prior campaigns and watchdog remain stopped.

## Completed result against control B

Candidate `outputs/training/anatomical-attention/candidate` used seed 2263738815
and selected epoch 10; B selected epoch 9. The comparison reproduces saved selected
predictions and uses 2,000 paired report-group bootstrap draws on the same 904 studies.

| Metric | Control B | Anatomical attention | Difference |
|---|---:|---:|---:|
| Macro AUC | 0.896582 | 0.895529 | -0.001053 |
| Medial Meniscus | 0.926587 | 0.923235 | -0.003352 |
| Lateral Meniscus | 0.897803 | 0.900045 | +0.002242 |
| Medial OA | 0.907763 | 0.912023 | +0.004261 |
| Lateral OA | 0.868603 | 0.865511 | -0.003092 |
| PF OA | 0.869275 | 0.874622 | +0.005347 |
| Meniscus margin AUC | 0.971248 | 0.970062 | -0.001186 |
| OA margin AUC | 0.866116 | 0.872219 | +0.006103 |

Macro 95% interval: [-0.004640, +0.002449]. Meniscus family difference -0.000555
and OA family +0.000584 are both inconclusive. OA margin interval is wide,
[-0.034409, +0.045071], so its positive point estimate does not establish better
compartment assignment. All five meniscus/OA label intervals include zero.

At the prevalence diagnostic cutoff, meniscus lateral false positives fall from
44 to 42 (medial-positive subset 28 to 27); OA falls from 82 to 81 (subset 51 to 50).
These small count changes offer little corroboration for a meaningful localization gain.
Other-eight AUC falls by 0.001587, notably MCL -0.014799 and Synovitis -0.009522;
Synovitis has an exploratory unadjusted interval below zero. Effects outside the
biased labels can arise through shared training and do not identify a causal mechanism.

The anatomical candidate exceeds untied pooling's 0.894777 by 0.000752 and avoids
its large OA-margin decline, but that comparison changes multiple mechanisms and
does not establish a winning spatial prior. No evidence currently justifies adopting
the fixed bias, tuning its constants on this fold, or inferring that a learned
anatomical module would help from this result alone.

Seeds match B, but the earlier repaired-slice/preprocessing difference remains:
this is not a clean same-source control comparison. No fresh control or seed replication
was run. The subsequent within-checkpoint bias ablation is below. Patient bootstrap excludes training-seed,
checkpoint-selection and multiple-comparison uncertainty.

Artifacts: [comparison](../outputs/analysis/anatomical-attention-vs-control-b/comparison.json)
and [per-label diagnostics](../outputs/analysis/anatomical-attention-vs-control-b/per_label.json).

## Bias-off inference ablation

User-authorized validation on the same selected checkpoint, with no retraining.
Ran only intact and without-anatomical-bias variants of the existing encoding
diagnostic, sharing one backbone pass; plane and other embedding interventions
were not run. GPU availability was checked before launch. Saved input/checkpoint
identities passed; intact replay matches saved AUCs exactly and raw probabilities
within 2.98e-8. Serialized replay predictions match the saved CSV exactly.

| Metric | Bias on | Bias off | Off minus on |
|---|---:|---:|---:|
| Macro AUC | 0.895529 | 0.891910 | -0.003619 |
| Medial Meniscus | 0.923235 | 0.919718 | -0.003516 |
| Lateral Meniscus | 0.900045 | 0.872602 | -0.027444 |
| Medial OA | 0.912023 | 0.903936 | -0.008088 |
| Lateral OA | 0.865511 | 0.861130 | -0.004381 |
| Meniscus margin AUC | 0.970062 | 0.958764 | -0.011298 |
| OA margin AUC | 0.872219 | 0.817928 | -0.054291 |

All eight untargeted labels are unchanged. Meniscus lateral false positives rise
42 to 56, with the medial-positive subset rising 27 to 47; OA false positives rise
81 to 87, with the medial-positive subset rising 50 to 63.

An additional paired report-group analysis uses the campaign convention: 904
studies, 852 report groups, 2,000 valid draws, seed 20260914. Macro off-minus-on
95% interval is [-0.005213, -0.002120]; meniscus family [-0.023588, -0.007764];
OA family [-0.010689, -0.002098]. Meniscus-margin interval is
[-0.021439, -0.002010]; OA-margin interval [-0.093806, -0.018079]. These are
fixed-checkpoint, unadjusted diagnostic intervals. The diagnostic's original
study-bootstrap results are retained separately and are not substituted for
report-group intervals.

Interpretation: the spatial prior is active and materially supports compartment
discrimination in this trained checkpoint. This revises the earlier uncertainty
about whether the prior is used. It does not establish that training with the
prior beats training without it: model weights co-adapted to its presence, and
removing it is an inference intervention. The complete candidate still does not
beat B overall. No adoption, code removal, new training or learned-prior implementation
was performed.

Artifacts: [report-group comparison](../outputs/analysis/anatomical-bias-ablation/paired_report_group.json),
[bias-on predictions](../outputs/analysis/anatomical-bias-ablation/intact.csv),
[bias-off predictions](../outputs/analysis/anatomical-bias-ablation/without_anatomical_bias.csv).
The diagnostic directory also retains sources, checkpoint/config/input hashes,
interventions, batch identities and per-target metrics.

## Learned family strengths: next experiment

Implemented after the bias-off ablation; the user-launched run is now complete,
with results below.
`model.learned_anatomical_strength=true` requires anatomical attention to be enabled.
The [candidate config](../outputs/training/anatomical-attention/learned-strength/input_config.yaml)
uses seed 2263738815 and the same recipe/data as the completed fixed-bias run.
Its only recipe change is learning the two anatomical strengths.

For each family, `alpha = exp(log_strength)` with `log_strength` initialized to
zero. The two parameters are ordered meniscus, OA and mapped to labels by target
name. Both start at exactly one, add no random draws at initialization, remain
positive and can approach zero without reversing medial/lateral meaning. They
scale only the anatomical term; the 20 mm tanh shape, existing pooled values,
independent heads, loss and sampling are unchanged. There is no upper bound on
learned strength; the original tanh spatial term remains bounded.

The existing optimizer assigns these head parameters the head learning rate and
zero weight decay because they are one-dimensional. Epoch-end CSV metrics record
`anatomical_strength_meniscus` and `anatomical_strength_oa`; checkpoints retain the
raw log strengths, so the selected epoch's exact values can be recovered. Existing
attention-temperature handling scales the whole anatomical term consistently.
The option defaults off; the fixed-bias and no-bias controls remain available.

User launch from repository root, after checking GPU availability:

```bash
rtk proxy uv run --locked python -m scripts.pipeline.train configs/experiments/anatomical-attention/learned-strength.yaml --run-id learned-strength
```

Compare against `outputs/training/anatomical-attention/candidate`, the completed
fixed-bias run, not historical control B. Use `compare_fold_runs` with allowed
config differences `name,model.learned_anatomical_strength` and allowed input
identity difference `preprocessing_identity` after reviewing the source delta.
Matching seeds reduces noise but does not make training deterministic. Examine
both selected-checkpoint strengths and their epoch trajectories alongside the
macro, per-label and compartment results; a changed coefficient alone is not a win.

Ruff lint/format, syntax parsing, configuration loading and whitespace checks passed.
The loaded candidate differs from the completed fixed run only in name and the
learned-strength option. Archived-source review found changes only in config,
model and Lightning files for this implementation; preprocessing/data code matches
the fixed run. No tests, training or GPU work were run for this implementation.

## Learned-strength result

Run `outputs/training/anatomical-attention/learned-strength` completed with seed
2263738815, selected epoch 10, 4,380 steps and 904 validation studies. Training
took 1,554.72 seconds and final prediction 20.27 seconds. Both learned and fixed
runs selected epoch 10. This comparison keeps the repaired dataset and side plumbing
the same, unlike comparisons with historical control B.

| Metric | Fixed bias | Learned strength | Difference |
|---|---:|---:|---:|
| Macro AUC | 0.895529 | 0.897443 | +0.001914 |
| Meniscus family AUC | 0.911640 | 0.911947 | +0.000307 |
| OA family AUC | 0.888767 | 0.892716 | +0.003949 |
| Medial Meniscus | 0.923235 | 0.922620 | -0.000615 |
| Lateral Meniscus | 0.900045 | 0.901275 | +0.001230 |
| Medial OA | 0.912023 | 0.910633 | -0.001390 |
| Lateral OA | 0.865511 | 0.874799 | +0.009288 |
| Meniscus margin AUC | 0.970062 | 0.969327 | -0.000734 |
| OA margin AUC | 0.872219 | 0.861157 | -0.011062 |

The selected checkpoint learned meniscus strength **1.46533** and OA strength
**1.13049**, from initial values of one. Meniscus rose quickly (1.060, 1.247, 1.360
over the first three epochs); both strengths approached a plateau by epochs 8–10.
They did not collapse to zero or grow without bound in this run. Checkpoint key
`model.anatomical_log_strength` contains [0.38208091, 0.12264736]; epoch CSVs retain
the full trajectories.

Macro learned-minus-fixed paired report-group 95% interval is
[-0.000926, +0.004925], so the overall gain is inconclusive. Lateral OA interval
is [+0.001267, +0.017286], an encouraging unadjusted per-label result. OA lateral
false positives fall 81 to 74, with the medial-positive subset falling 50 to 46.
Meniscus false positives rise 42 to 45. The worse OA margin has interval
[-0.037622, +0.013255]; better lateral-OA ranking does not establish better direct
medial/lateral assignment.

Against historical B, macro is +0.000861 with interval [-0.002575, +0.004456];
OA-family mean is +0.004533, while meniscus-family mean is -0.000248. The repaired
data/source caveat remains for B. Both comparisons use 2,000 paired report-group
draws and exclude training-seed, checkpoint-selection and multiple-comparison
uncertainty.

This is a promising implementation candidate for confirmation, not an adoption.
The next informative training comparison would pair the learned-bias recipe with
the no-bias current-source recipe on another preselected seed, rather than tuning
the spatial constants or assuming the margin improved. No further training or
main-recipe change was made in this analysis.

Artifacts: [learned versus fixed](../outputs/analysis/anatomical-learned-vs-fixed/comparison.json),
[per-label intervals](../outputs/analysis/anatomical-learned-vs-fixed/per_label.json),
[learned versus B](../outputs/analysis/anatomical-learned-vs-control-b/comparison.json).

## Learned strengths reset to one at inference

User-authorized inference-only intervention on the learned-strength checkpoint.
Replayed intact predictions and reset both log strengths to zero in memory, sharing
one backbone pass. No other weights changed, no retraining occurred, and the saved
checkpoint was not overwritten. Intact predictions reproduce saved AUCs exactly,
raw probabilities within 2.98e-8 and serialized prediction values exactly.

| Metric | Learned strengths | Both strengths = 1 | Reset minus learned |
|---|---:|---:|---:|
| Macro AUC | 0.897443 | 0.897213 | -0.000230 |
| Medial Meniscus | 0.922620 | 0.923460 | +0.000841 |
| Lateral Meniscus | 0.901275 | 0.897794 | -0.003481 |
| Medial OA | 0.910633 | 0.910681 | +0.000048 |
| Lateral OA | 0.874799 | 0.874630 | -0.000169 |
| Meniscus margin AUC | 0.969327 | 0.968649 | -0.000678 |
| OA margin AUC | 0.861157 | 0.860776 | -0.000381 |

Other-eight predictions are unchanged. The reset-minus-learned macro 95% paired
report-group interval is [-0.000580, +0.000106]. Lateral-meniscus interval is
[-0.007012, +0.000063]; the other targeted labels and both margin intervals also
include zero. OA lateral false positives and medial-positive subset remain 74 and
46; meniscus total changes 45 to 44, while the medial-positive subset changes 29
to 30. Analysis uses 904 studies, 852 report groups, 2,000 valid draws and seed
20260914. Intervals are exploratory and conditional on the selected checkpoint.

The learned values have a small direct overall effect, concentrated in lateral
meniscus. The large lateral-OA gain over the separately trained fixed-bias run
survives resetting OA strength to one. Thus it mostly resides in the trained
weights/trajectory rather than the final coefficient's direct inference effect.
This does not establish that learning strength caused a repeatable training gain;
the ablation does not recreate fixed-strength training. There is no demonstrated
reason to reset the learned checkpoint or tune coefficients on validation, and
no default implementation or saved checkpoint was changed.

Artifacts: [report-group comparison](../outputs/analysis/learned-strength-reset/paired_report_group.json),
[intervention record](../outputs/analysis/learned-strength-reset/parameter_intervention.json),
[reproducible diagnostic invocation](../outputs/analysis/learned-strength-reset/invocation.py),
[summary calculation](../outputs/analysis/learned-strength-reset/summary_invocation.py).


## Learned family widths

Implemented and evaluated; the user-launched run is complete. This supervised
small-change follow-up is not supported for adoption; its tracking stays here
under the autoresearch documentation rule. No autonomous campaign was resumed.
The hypothesis is that meniscus and OA benefit from different spatial transition
widths, beyond the already learned bias strengths. No improvement is established.

The [candidate config](../outputs/training/anatomical-attention/learned-width/input_config.yaml)
enables `model.learned_anatomical_width` alongside learned strengths. Each family
uses `width_mm = 20 * exp(log_width_scale)`, initialized with a zero log scale,
so the attention addition is `sign * strength * tanh(medial_mm / width_mm)`.
Widths are positive, shared between medial and lateral labels within each family,
and ordered meniscus, OA. Smaller widths sharpen the transition; larger widths
make it more gradual. Initialization adds no random draws and reproduces the
learned-strength recipe's initial bias. Unknown-side or unusable geometry still
contributes zero bias. Other labels receive no anatomical bias.

The two new parameters use the head learning rate and zero weight decay. Checkpoints
retain `anatomical_log_width_scale`; epoch CSVs log `anatomical_width_mm_meniscus`
and `anatomical_width_mm_oa`, in addition to the existing strengths. Width learning
requires anatomical attention and defaults off. The experiment retains seed
2263738815, loss, sampling, data, optimizer and ten-epoch budget. Only the run name
and width flag differ from the learned-strength config. No control rerun is needed
for this requested screen; use the completed learned-strength run as its immediate
reference, with historical B as secondary context and its existing data caveat.

User launch from repository root after checking GPU availability:

```bash
rtk proxy uv run --locked python -m scripts.pipeline.train configs/experiments/anatomical-attention/learned-width.yaml --run-id learned-width
```

Output: `outputs/training/anatomical-attention/learned-width`. Compare macro AUC,
per-label AUC and compartment margins against
`outputs/training/anatomical-attention/learned-strength`. Inspect selected-checkpoint
widths and strengths plus their trajectories: strength and width can partly
compensate for each other, so coefficient movement alone is not evidence of a win.
Matched seeds do not eliminate training variability. Static lint/format, syntax,
configuration loading and recipe-difference checks passed; no tests or GPU work
were run during implementation.


## Learned-width result

The seed-2263738815 run completed all ten epochs and selected epoch 10. Saved
predictions reproduce macro AUC 0.8943017923. Archived source comparison against
learned-strength shows only the new experiment config and the width additions in
config, model and Lightning; the only input-identity change is the source hash.

| Metric | Learned strength | Learned width | Width minus strength |
|---|---:|---:|---:|
| Macro AUC | 0.897443 | 0.894302 | -0.003141 |
| Medial Meniscus | 0.922620 | 0.923301 | +0.000682 |
| Lateral Meniscus | 0.901275 | 0.898248 | -0.003027 |
| Medial OA | 0.910633 | 0.909528 | -0.001105 |
| Lateral OA | 0.874799 | 0.869204 | -0.005595 |
| Meniscus margin | 0.969327 | 0.963904 | -0.005423 |
| OA margin | 0.861157 | 0.860394 | -0.000763 |

The macro paired report-group 95% interval is [-0.006254, -0.000153], favoring
strength-only on these saved predictions. Meniscus and OA family intervals and
both margin intervals include zero. Other-eight mean AUC falls by 0.003581, so
the loss is not confined to anatomically biased labels. Historical B scores
0.896582: width is lower by 0.002280, interval [-0.005798, +0.001127]. Its OA-family
mean is 0.001183 above B but 0.003350 below strength-only; this is not a reason to
prefer widths. The previously documented repaired-data caveat applies to B.

Selected widths are 15.32794 mm for meniscus and 16.72296 mm for OA, down from
20 mm. Strengths are 1.47983 and 1.12414. Width trajectories settle smoothly over
ten epochs; learning a sharper transition did not improve the measured outcome.
This is a training comparison, not an inference intervention that isolates the
final width coefficients. Intervals cover 904 studies in 852 report groups with
2,000 paired draws, seed 20260914, and exclude training-seed/checkpoint-selection
uncertainty. They do not prove widths are universally harmful.

Recommendation: do not adopt learned widths or spend another training on this
refinement; retain strength-only as the stronger candidate in this line. No
model implementation or main defaults changed during this analysis. No training,
tests or GPU inference were run.

Artifacts: [versus learned strength](../outputs/analysis/anatomical-width-vs-strength/comparison.json),
[versus historical B](../outputs/analysis/anatomical-width-vs-control-b/comparison.json).
Each comparison retains configuration/input differences, source snapshots and
prediction/checkpoint hashes. The selected checkpoint retains exact log widths
and strengths; the run's epoch CSVs retain their trajectories.


## Repeated control on current data

The user-launched `outputs/training/anatomical-attention/control-repeat` completed
all ten epochs, selecting epoch 10. Config:
`configs/experiments/anatomical-attention/control.yaml`, seed 2263738815, fold 0,
validation x1, all anatomical options off. It uses the repaired data and current
working-tree code. Saved predictions reproduce macro AUC 0.8951120902.

| Run | Macro AUC | Difference from new control |
|---|---:|---:|
| Historical B | 0.896582 | +0.001470 |
| New control | 0.895112 | 0 |
| Fixed anatomical bias | 0.895529 | +0.000417 |
| Learned strength | 0.897443 | +0.002331 |
| Learned width | 0.894302 | -0.000810 |

New-control minus B has paired report-group 95% interval [-0.004217, +0.001340].
This does not establish that B's seed was lucky: the seed is the same and the
comparison also includes repaired data, intervening code and nondeterministic
training. It does show that B's recorded score was not reproduced by this run.
Earlier experiments remain valid observations under their original conditions;
failed comparisons against B do not establish that each implementation is useless.
Do not retroactively replace historical decision references with a weaker control.

Learned-strength minus new-control is +0.002331, interval [-0.001034, +0.005645].
Meniscus-family mean improves +0.000499 and OA-family mean +0.007352. The OA-family
interval is [+0.000820, +0.014445], exploratory and unadjusted for selection or
multiple comparisons. Per-label changes are medial meniscus -0.001604, lateral
meniscus +0.002602, medial OA +0.003732 and lateral OA +0.010971. Both compartment
margin AUCs decline (meniscus -0.001186, OA -0.011697), with intervals including
zero; better compartment assignment is not established. Overall improvement is
still inconclusive under the campaign macro criterion. Strength-only remains a
promising implementation, especially for OA; widths remain unsupported.

Archived source review against learned-strength found the later width option
(disabled for both compared recipes), its config validation and epoch logging,
and autoresearch seed-policy changes. Preprocessing code matches between these
two runs; recorded input identities differ only in source hash. B retains the
previously documented data/source caveat. Bootstrap uses 904 studies, 852 report
groups, 2,000 draws and seed 20260914, conditional on the fitted models and selected
checkpoints. No training, tests or GPU inference were launched during analysis;
no incumbent, recipe or historical verdict was changed.

Artifacts: [new control versus B](../outputs/analysis/control-repeat-vs-control-b/comparison.json),
[learned strength versus new control](../outputs/analysis/anatomical-strength-vs-control-repeat/comparison.json).


## Lower-confidence interval exploration

At the user's request, the same 2,000 paired report-group draws comparing learned
strength with the new control were summarized at additional central confidence
levels: 80% [+0.000103, +0.004518], 90% [-0.000518, +0.005072], and 95%
[-0.001034, +0.005645]. Point gain remains +0.002331. Thus a >=0.001 gain plus a
positive 80% interval would pass this run; the corresponding 90%/95% gates would
not. The user selected gain >=0.001 plus a positive 80% interval as the fixed future
adoption rule. This is not independent evidence: choosing
confidence after observing this result must not be presented as confirmation.
Intervals remain conditional on the fitted models and exclude training variation
and repeated experiment selection. No model execution was needed.

[Additional interval results](../outputs/analysis/anatomical-strength-vs-control-repeat-ci-levels/comparison.json),
[calculation metadata](../outputs/analysis/anatomical-strength-vs-control-repeat-ci-levels/confidence_levels.json).


The production diagnostic and adoption path reproduces the selected 80% interval
from the saved predictions and returns `provisional_accept` for learned strength
against the new control: [recorded assessment](../outputs/analysis/anatomical-strength-vs-control-repeat-ci-levels/adoption_assessment.json).
This adds analysis evidence without modifying historical campaign decisions.
