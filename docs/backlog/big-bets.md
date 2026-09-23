# Big bets

Investigation-sized plans that need focus sessions with the user: diagnostics, a design, several
runs or a new preprocessing identity. Start one only in a session with the user; when it starts,
move its entry to [ongoing work](../ongoing/INDEX.md) and remove it from this file in the same change.
Entries under [parked](#parked-until-a-specific-need) wait for a concrete need before they become
candidates. Each entry says whether its motivation is measured or a hypothesis.

## OA-only rubric-aligned label repair

**Defect observed; repair and image-model benefit untested.** Prioritized in the 2026-09-20 agenda
review. The [blinded report audit](../history/sharing/oa-and-meniscus-disagreement-audit.md) found
19 report-derived OA cells from 16 studies with explicitly below-threshold report findings but
frozen targets of 0.80–0.93. The sample was deliberately selected: it does not estimate population
error prevalence or establish image-adjudicated truth. No labels were changed. The full-table
[Tonylin swap](../validation/results/2026-09-13-tonylin-label-comparison.md) and
[unmentioned-cell mask](../history/sharing/2026-09-13-label-masking-result.md) were different,
measured interventions and are not candidates for repetition here.

1. **Bounded training-only pilot:** review medial, lateral and patellofemoral OA reports, including
   high-confidence positives. Keep a random panel separate from severity-term enrichment; do not
   restrict review to probabilities in `[0.3, 0.8]`, which misses most known conflicts.
2. **Lock the extraction policy:** distinguish qualifying disease, explicitly below-threshold
   disease, explicit negative, uncertain and unmentioned. Version the mapping to revised soft
   targets and preserve gold labels, the other nine targets and insufficiently supported cells.
   Keep an untouched report panel to assess the fixed rule; do not tune it to flip the known cases.
3. **Compare supervision only:** revised versus original OA targets with identical imaging,
   sampling and the normal training budget, using frozen validation targets. Record the label
   manifest, revision, split, seed, per-label AUC and changed class counts/weights. If weights move
   materially, an original-weight sensitivity separates target repair from altered emphasis.

Use report terms for initial triage. Model-disagreement triage is optional only with a predictor
whose supervised learning excludes outer validation studies; do not train a new triage model.
Advance after independent report review supports the policy and a controlled image-model result
supports adoption. A more faithful report extraction alone is not a score gain. Record disagreement
between frozen-target AUC and rubric diagnostics without rewriting evaluation targets. Stop after
an uninformative bounded pilot; no broad label rewrite, auxiliary severity head or extra epochs.
This entry owns the investigation; training remains user-launched when the user starts it.

Since 2026-09-21, PF OA in both training and evaluation comes from the
[combined labels'](../public-labels/combined-labels.md) consensus, which is stricter than Steven v4.
The split's evaluation targets were rewritten by user decision. Medial and lateral OA are still
Steven v4.

## Richer report supervision: OA attributes

**Accepted for the backlog on 2026-09-21; investigation, implementation and training unstarted.**
Train image features to predict reliable anatomical/severity information from reports beyond
the twelve disease targets. [MedKLIP](https://arxiv.org/abs/2301.02228) motivates structured
report supervision in chest radiographs; the proposed small auxiliary heads are a knee-specific
adaptation, not a reproduction or a measured gain.

**Premise weakened, 2026-09-22.** Public tables already hold per-label severity, so step 1 needs no
LLM run. But where two sources agree on severity (OA, Effusion, Baker's, ACL), the soft training
targets already encode it (Spearman 0.66–0.79 among positives), and the incumbent's predictions
already rise with it. The model ranks worst on low-grade positives, and only about half of those
gold cells are positive: label noise that an auxiliary head cannot fix
([report attributes](../public-labels/report-attributes.md)). Location attributes (horn,
compartment, bruise site) are not in any public table.

1. **Bounded extraction pilot:** start with compartment-specific OA severity attributes, not a
   broad concept vocabulary. Define and audit an assertion-aware extraction policy on training
   reports, with a separate report panel for confirmation. Version the policy, source text
   identity, extracted values and observation masks; report coverage and class counts. Do not
   invent severity from report silence, unresolved wording or the existing disease probability.
   Any learned extractor must exclude the outer validation partition. Stop if reliable attribute
   supervision is too sparse to justify training.
2. **Training-only heads:** reuse the corresponding image-derived pooled features for small
   auxiliary predictors; supervise only supported attributes. Compartment identity routes the
   evidence to the correct target; do not add a constant compartment-classification task where
   the disease head already fixes it. Keep all twelve disease targets, gold overrides and the
   current disease loss unchanged. Fix the attribute encoding and auxiliary loss coefficient
   before the comparison. Discard auxiliary heads at inference; reports are never model inputs.
3. **Controlled comparison:** fresh current-recipe control versus auxiliary supervision only,
   preserving the split, evaluation targets, incumbent seed, image workload and training budget.
   Follow the [controlled-comparison rules](../validation/validation-strategy.md#controlled-comparisons),
   recording macro/per-label AUC, paired uncertainty, auxiliary coverage/loss and runtime.
   Attribute-prediction accuracy is diagnostic, not the adoption metric.

This learns finding content, unlike the existing
[report-state head](small-changes.md#auxiliary-report-state-prediction), which predicts whether
a finding was reported. It is also separate from [OA label repair](#oa-only-rubric-aligned-label-repair):
that investigation changes disease supervision and still excludes auxiliary heads from its own
comparison. Do not combine the interventions. Finding-level text alignment is not part of this
accepted pilot. Start only in a user session; training remains user-launched.

## Position and knee side

**The supervised anatomical-attention sequence is closed:** learned family strengths
with fixed 20 mm width were adopted; appearance-only pooling and learned widths
were discarded. [Evidence](../../autoresearch/anatomical-attention.md) owns these
results. The items below are unstarted independent proposals, not pending branches
of that experiment sequence.

Already measured: the
[knee-side embedding](../validation/results/2026-09-14-knee-side-embedding-result.md) and the
[token graph](../validation/results/2026-09-14-token-graph-result.md) tied and were retired, and
the [attention prior](../validation/results/2026-09-15-attention-prior-result.md) regressed by
0.002177 and was
[retired](../history/model-lineage.md#retired-attention-prior) with the model's
patient points. Remaining (hypotheses), from the
[position research sequence](../research/position-encoding.md#5-recommended-sequence) and the
[knee-side approaches](../research/knee-side-and-location.md):

- **Phase 0 diagnostics:** report laterality read for the persistent wrong-side studies,
  attention and token-position dumps, and frozen token probes. Plane-restricted inference on the
  champion was skipped by user decision.
- **Common knee orientation** (approach 4) and **learned landmarks or localization supervision**
  (approach 7, which depends on a rules-compatible knee segmenter).
- **Side-conditioned sagittal sampling centre** (measured premise, unmeasured payoff). Sagittal
  slices step along patient L, and the compartment labels sit about 31 mm off the study centre
  while the sagittal draw is centred on it. The [spatial importance
  map](../data-preparation/peripheral-coverage.md#spatial-importance-map) shows a central trough on that
  axis alone, flanks minus centre +0.126 against -0.634 coronal and -1.099 axial, carried by the
  compartment labels and symmetric about zero. The symmetry is the problem: pooled left and right
  knees make "medial and lateral matter equally" and "only medial matters, seen twice" produce the
  same picture. The remaining export blocker is that it centres and scales L but never mirrors by
  knee side, so it must be re-exported with left knees mirrored before the lobes can be told apart;
  the per-study utilities for that live outside this repository. The offset would have to flip sign
  per knee side. The [audited side policy](../metadata/knee-side-policy-audit.md) is now
  implemented in preprocessing for anatomical attention, so side plumbing is no longer
  missing; abstentions must take no offset. The 96.14% coverage is the historical audit,
  not a fresh coverage measurement.

Approaches 2 and 3 act through the 3D coordinate branch that approach 1's checkpoint barely used;
the user also declined approach 3 on 2026-09-14. Image-based side inference is deferred by user
request. The [coordinate audit](../metadata/knee-side-coordinate-investigation.md) owns the side
evidence.

## Learned acquisition intensity bounds

Hypothesis. Compare a shared predictor with the current fixed percentile clipping. It must receive
unclipped floating-point input; uint8 clipping has already discarded information. Parameters
indexed by training series UID would not generalize to unseen acquisitions. Other fixed quantile
levels lost: 0.001/0.999 by 0.0003 and 0.01/0.99 by 0.0021
([result](../validation/results/2026-09-17-overnight-results.md)).

- *Rejected source:* the DICOM `WindowCenter`/`WindowWidth` tags (measured 2026-09-20, 30 series):
  median 1.43× wider than the 0.005/0.995 bounds, vendor-dependent (Toshiba 1.77 against Philips
  1.20) and varying slice to slice in 23 of 30 series, so they are neither a tighter nor a
  sampling-independent bound.

## Sampling-independent intensity bounds

**Code observation; size measured, harm on the metric unmeasured.** `prepare_series`
(`src/knee_abnormality/data/preprocessing/preparation.py`) takes the intensity quantiles over the
unique slices of the selected triplets, so the same source slice gets different normalized pixels
when its companion slices change: between training epochs, and between training and the 2×
validation bag. The variability may also act as useful augmentation.

1. **Measured 2026-09-20** on 40 studies over six epochs
   ([bound stability](../data-preparation/preprocessing-implementation-and-profile.md#clipping-bound-stability-across-draws)):
   the high bound moves 0.072 at the median and 0.245 at p90 between draws of one series, against
   the 0.2 relative range of the contrast augmentation, and 14.2% of series exceed it. The spread
   is a small-sample effect concentrated in series with few triplets (0.150 at two or fewer, 0.025
   above six), not a property of the estimator: robust estimators calibrated to the same clip level
   gain about one point, and per-triplet bounds are four times worse. Not measured: normalized-pixel
   and clipping differences on shared source slices, the 1× versus 2× bags, or any effect on AUC.
2. **Only if material:** estimate the bounds from a deterministic sparse reference sample of each
   acquisition, independent of epoch, bag size and selected windows, identical in training,
   validation and inference, at the current quantile levels. Define short-series and
   decode-failure fallbacks. It changes the preprocessing identity; cache scalar bounds only under
   an identity that prevents stale reuse.

- *Distinct from* the quantile-level runs and the learned bounds above: it changes the pixel
  population, not the levels or the estimator.
- *Cost:* extra decoding, against the
  [select-before-decoding](../architecture/data-flow-and-iteration-speed.md#iteration-speed) rule
  that the [loading speedups](loading-speedup/INDEX.md) rely on.
- *Stop if* the variability is negligible, the runtime cost is disproportionate, or a controlled
  run gains nothing.

## Leave-one-scanner-out diagnostic

**Diagnostic, unmeasured.** `all_5fold_scanner_v1` balances scanners across folds, which is right
if the test set shares the training scanners. A split that holds out whole scanner groups would
measure how much the model leans on scanner or site shortcuts. The gap between the two splits is
the estimate. An unverified competitor README claims a random split inflates AUC by about 0.05
through scanner memorization. Run it once, never as a development split, and only if a
scanner-specific failure is suspected.

## Auxiliary-loss campaign

**Requested by the user on 2026-09-22; undefined and unstarted.** An autoresearch campaign on
training-only auxiliary objectives added beside the unchanged disease loss. Its premise is a
hypothesis: no auxiliary objective has trained active in this project. The GPU is free since the
[oblique line](../history/oblique-groups.md) closed on 2026-09-23.

- **Scope:** extra heads or loss terms on the shared features, discarded at inference, so the
  submission's runtime does not change. The class-balanced soft BCE, its twelve targets, gold
  overrides, sampler, budget and evaluation stay fixed. Objectives and coefficients are shared
  across labels, never selected or weighted per label. Each candidate fixes its coefficient before
  launch; coefficient sweeps are not ideas.
- **Seed candidates** (all would transfer to the campaign when it is defined):
  [report-state prediction](small-changes.md#auxiliary-report-state-prediction) (hypothesis, with
  a measured judge risk: it teaches reporting behaviour that fold 0 rewards and the leaderboard
  may not), [OA attributes](#richer-report-supervision-oa-attributes) (premise weakened: agreed
  severity is already in the targets, moderate confidence) and per-view disease classifiers
  (below). Acquisition or geometry pretext tasks (plane, contrast, slice position, knee side) are
  guesses with no measured premise.
- **Per-view disease classifiers** (guess, low-to-moderate confidence; added 2026-09-23 from the
  RSNA 2024 lumbar winner's modality-classifier auxiliaries, whose
  [ablation](https://files.speakerdeck.com/presentations/3a7b5a2c84e14df2a1bce0b41dd609d0/slide_29.jpg)
  bundles them with other objectives and does not isolate them). For each present plane/contrast
  type, pool that type's existing tokens into a bag and predict all twelve study labels from it,
  masking absent types; no view/label subset is selected. The main global head, its attention
  normalization and anatomical bias stay unchanged; reuse the backbone pass and drop the heads at
  inference. Average the auxiliary loss over present views so studies with more types gain no
  extra weight. A positive study label need not be visible in every view, so the auxiliary targets
  carry noise. [Series dropout](../modeling/current-model.md#architecture) already forces
  predictions from subsets of views, which may leave little to gain.
- **Prior evidence:** the [shared-loss campaign](../../autoresearch/campaigns/losses/INDEX.md)
  replaced the main objective rather than adding one (ranking inconclusive, focal negative). The
  medial/lateral exclusivity penalty
  [never became active](../../autoresearch/campaigns/medial-lateral-transfer/prior-work.md#corrected-exclusivity-follow-up)
  (about 1e-6 against a main loss of 0.47), so every candidate logs each loss component and shows
  the auxiliary term is active.
- **To define it:** follow [campaign setup](../../autoresearch/campaign-setup.md) with the
  incumbent at that time as the reference. Start from the losses campaign's policy (the loss
  directory, `training/lightning.py`, `config.py`, `trainer.loss`) and add `modeling/model.py`
  for the heads, plus any label-source access the report-state targets need.

## RadImageNet-pretrained backbone

**Evidence-backed (one external report), untested here; moved from small changes on 2026-09-23 because it needs an off-recipe backbone pair.** Load a RadImageNet-pretrained
ResNet-50 in place of its ImageNet weights, behind one option naming the pretraining source, and
compare the two at identical architecture, normalization and recipe. A participant reports that
going the other way — RadImageNet to ImageNet ResNet-50, same architecture, per-corpus
normalization corrected — cost them **−0.063**, the largest in-competition number anyone has
published on medical pretraining ([sweep](../research/public-research.md#the-2026-09-20-re-sweep));
the public 0.942 stack also carries RadImageNet heads as one of five families. It disagrees with
the [benchmark ranking](../research/backbone-mri-evidence.md) that put RadImageNet ResNet-50 19th
and MedDINOv3 25th, which is what makes it worth one measurement rather than an adoption.

Two caveats price the work. RadImageNet publishes ResNet-50, DenseNet-121, InceptionV3 and
IRV2 only, so neither of our backbones (EfficientNet-B0, ReXNet-200) can take these weights; the
comparison needs a ResNet-50 pair that is off the current recipe, and a win would then have to be
re-measured on a backbone we would actually submit. The weights are a third-party download, so
they need a pinned source, hash and licence check before any run.

## Parked until a specific need

### External pretrained image encoders

**Survey completed 2026-09-20; implementation and training unstarted.** The
[checkpoint review](../research/huggingface-image-encoders.md) owns sources, loading contracts,
revision pins and unresolved terms. Investigate only when a backbone-diversity experiment is
prioritized: DINOv3 ConvNeXt-T first for practical integration, then RadioDINO-S against the
same architecture's general-image weights. OmniRad-S is an alternative after its conflicting
license metadata is resolved. These require spatial mappings or patch-token adapters; they are
not hyperparameter-only runs. Hold the current balanced-BCE recipe and data contract fixed,
compare fresh controls, and record per-label AUC, ensemble contribution, time and memory.
The [RadImageNet comparison](#radimagenet-pretrained-backbone) is the one medical-pretraining
measurement with an in-competition claim behind it.

OrthoFoundation-L is the strongest direct 2D knee-pretraining lead; MRI-CORE and the released
three-plane OrthoDiffusion weights are higher-cost alternatives within this investigation.
Establish loading, source/competition eligibility and input semantics before allocating runs.
The 3D option needs a separate volume pipeline. Existing historical ranks do not settle transfer
under the current recipe. Training remains user-launched; move this entry to ongoing work if
implementation or experiments start.

### Study overlap and anatomical matching

**Follow-up investigation; unstarted.** The user-requested bounded
[study-overlap audit](../metadata/study-overlap-investigation.md) found no confirmed
repeat among its shortlisted report/geometry candidates. To look for separate
examinations of the same knee, retrieve neighbors using stable bone anatomy and
verify with physical registration across multiple slices and another plane.
Use existing prepared data where possible; calibrate against unrelated studies
from the same scanner and protocol. Distinctive trabecular patterns or surgical
features should corroborate shape similarity. For potential split-bilateral
sessions, combine sequence inventories, frequency, coil and numbering evidence;
mirrored anatomy or a shared report template alone cannot establish identity.
Keep candidate, likely acquisition overlap and confirmed duplicate evidence
separate. Change fold groups only with convincing evidence and rerun a baseline
on any revised split; never silently rewrite the existing membership.

### Reformatted fluid planes from 3D series

**Measured premise; value unmeasured, estimated ceiling about +0.0004 macro.** 392 training
studies (8.9%; 83 of fold 0's 905) have no 2D fluid series in one plane but hold an isotropic 3D
fluid series that covers it, so the model leaves that slot empty where the radiologist reads a
reformat. Two site protocols account for them:

- *No sagittal fluid (245 studies):* an axial Siemens DESS water excitation (`T2 DE3D WE TRA
  PATELLA FIT`, 0.6 mm slices, 144 slices, 86 mm slab) next to coronal fluid and two sagittal
  T1/PD series.
- *No coronal fluid (147 studies):* a sagittal 3D PD fat-suppressed series (Philips `SAG 3D VIEW
  PD SPAIR HR`, 0.4 mm spacing and 0.8 mm slices; Siemens `PD SPC RST FS SAG ISO`, 0.89 mm
  isotropic); some of these studies carry an axial reformat, none a coronal one.

Filling the empty slot with a 3 mm thick-slab reformat at the plane's 2D step, resampled straight
to the 0.625 mm grid, is the mechanism. It is parked because the ceiling is small: removing a
present fluid type cost 0.004–0.005 macro after redistribution on the studies holding it
([test-time allocation](../validation/results/2026-09-17-test-time-allocation.md#missing-types)),
and 9% of studies regaining one plane scales that to about +0.0004. The affected fold-0 studies do
not underperform either: `k_05` scores 0.906 on the 54 DESS studies against 0.9015 on the 822
unaffected ones; the 29 coronal-gap studies have too few positives to read. DESS shows marrow
oedema poorly, so its weakest label there, contusion (0.848 against 0.906, 14 positives), may not
be recoverable from a reformat. Revisit only if those subsets turn out to fail on the leaderboard
or a cheaper reformat path appears; the gate would be a frozen-checkpoint fill on fold 0 before
any training. Counts, protocols and subset scores:
`outputs/analysis/acquisition-geometry-20260922/` (`missing_slots`, `iso3d_profile`,
`fill_composition`, `mpr_subset`).

### Other parked work

- **External knee data** (MRNet, fastMRI+, SKM-TEA): only for a concrete missing capability, under
  the
  [admission criteria](../research/pretraining-and-external-data.md#admission-criteria-for-an-external-data-experiment).
- **Image-assisted label refinement:** fold-isolate every learned component that uses evaluation
  labels; earlier refinement gains involved
  [circular reasoning](../modeling/supervision.md#refinement-and-earlier-eligibility-uncertainty).
- **Full-volume 3D training from scratch, reconstruction or denoising systems, dense auxiliary
  tasks and broad leaderboard blend sweeps:** only after a specific failure or controlled result
  justifies the cost. Dense auxiliary tasks also need a reliable annotation source; segmentation
  Dice, SSIM or another competition's rank do not substitute for knee AUC.
