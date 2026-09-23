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

## Medial and lateral disease transfer: training-only shared detector

The original sharing proposals were transferred into the
[medial/lateral autoresearch campaign](../../autoresearch/campaigns/medial-lateral-transfer/INDEX.md).
That campaign's history owns attempted mechanisms, findings and dormant proposals;
this is not a second executable queue. It is stopped on the adopted learned-strength
baseline. Classifier coupling and the pooled adapter were rejected and removed;
[their research history](../research/medial-lateral-transfer.md) remains evidence.
Starting another transfer investigation requires a materially new mechanism and a
session with the user; do not automatically revive the old training-only detector.

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

## Per-label loss selection and learned mixtures

**Research completed 2026-09-20; implementation and experiments unstarted.** Exploit the trusted
external observation that ranking improves ACL while focal works better for MCL. The user wants
to develop the principle independently of the contact's implementation. The
[research note](../research/per-label-losses.md) owns the results, primary sources, candidate
formulas, model fit, pair-coverage calculation and evaluation design.

**Deferred after the 2026-09-20 agenda review.** The shared ranking and focal ingredients have now
been trained: ranking was inconclusive and focal trended negative
([ranking record](../../autoresearch/campaigns/losses/findings.md#completion-20260920t121537z-5b39b418),
[focal record](../../autoresearch/campaigns/losses/findings.md#completion-20260920t124339z-2948d523)).
These results do not settle per-label selection, but lower its priority below OA repair and the
new pooling comparisons. Reconsider a bounded frozen-feature screen before a full training program.

1. **Compare independent heads:** cache the current model's per-label pooled features and train
   balanced soft BCE, focal and ranking classifiers for all twelve labels. Share expensive
   feature extraction, allow larger ranking pools and repeat classifier seeds. Use current
   class-balanced BCE as the control.
2. **Assemble per-label winners:** compare stable selections against the original model and
   refitted-BCE controls. Selected linear rows preserve inference cost. Retain the current loss
   when differences are unresolved.
3. **Evaluate joint continuation:** introduce the selected losses after feature learning,
   comparing with an equally long current-loss continuation. Check every label because shared
   features couple their gradients.
4. **Investigate mixtures:** compare fixed BCE-plus-ranking hybrids with a small per-label
   bilevel controller, first on frozen features. Nonnegative coefficients sum to one per label;
   learn them from a fixed tuning objective after virtual training updates, not by minimizing
   the same weighted training loss. Evaluate warmup, slower coefficient updates and shrinkage
   toward a common prior only as needed. Noise-aware ranking follows only if diagnostics justify it.

Preserve soft-label semantics explicitly, define loss scales and usable pairs, and separate
tuning from evaluation according to the research note. Record configuration, revision, parent,
split/label/feature identities, seeds, coefficients, AUCs and compute. Advance when evidence
justifies the extra complexity; a failed frozen-head screen does not rule out representation
gains. Training remains user-launched. Move this entry to ongoing work when experiments start.

This entry owns loss selection within each label. The parked
[adaptive allocation](#adaptive-training-allocation) proposal concerns weights between targets
and study exposure; [static ranking loss](../../autoresearch/campaigns/losses/findings.md#completion-20260920t121537z-5b39b418) remains a separate
completed comparison.

## Image-model predictions as a label source

**Hypothesis (evidence from chest X-ray only).** Report extractors share each report's gap from
the image reading, so aggregating them cannot fix report-omitted findings. Synovitis is the clearest
case: every text source misses at least 14 of its 27 gold positives
([combined labels](../public-labels/combined-labels.md#literature-behind-the-design)).
VisualCheXbert and Dunnmon 2020 improved report labels by adding an image-side signal. The
knee equivalent is out-of-fold image-model predictions as one more source in the consensus, one
target group per run.

- *Precondition:* every study's image prediction must come from a model that never trained on it.
  That needs all five folds of `all_5fold_scanner_v1`, so the idea waits for fold training.
- *Public signal:* one participant reports that mixing image-model predictions into the extracted
  label beat both replacing and masking
  ([re-sweep](../research/public-research.md#labels-are-the-reported-lever-at-the-top-and-extractor-tuning-is-not)).

## Leave-one-scanner-out diagnostic

**Diagnostic, unmeasured.** `all_5fold_scanner_v1` balances scanners across folds, which is right
if the test set shares the training scanners. A split that holds out whole scanner groups would
measure how much the model leans on scanner or site shortcuts. The gap between the two splits is
the estimate. An unverified competitor README claims a random split inflates AUC by about 0.05
through scanner memorization. Run it once, never as a development split, and only if a
scanner-specific failure is suspected.

## Adaptive training allocation

**Investigation plan, accepted by the user on 2026-09-22; unstarted.** Let the per-label class
weights change during training instead of fixing their strength. Judge every candidate on the
public leaderboard.

**Evidence that motivates it:**

- **Static strength.** At the same seed on the scanner split, public scores are k=0.5 0.945,
  k=0.25 0.944, unweighted 0.943 and k=1 0.940 (S40–S44), while fold-0 validation ranks them in
  the reverse order; k=0.75 is training. The earlier S34→S38 pair moved validation +0.0129 and
  public −0.003 ([result](../validation/results/2026-09-21-class-balance-exponent.md)).
- **Likely mechanism (moderate confidence).** The weights sharpen the report extractor's decision
  boundary on rare labels. The validation deficit against the training labeller is about twice
  the deficit against nine independent extractors
  ([rescoring](../validation/results/2026-09-21-class-balance-exponent.md#rescoring-against-other-label-sources)),
  and on the 58 gold studies reports under-call Synovitis and over-call Effusion, Contusion, MCL
  and ACL. The largest k=1 weights (MCL 17.4, Fracture 14.5) fall on the labels whose report
  positives disagree most with official ones
  ([positive rates](../validation/results/2026-09-21-class-balance-exponent.md#report-derived-and-official-positive-rates)).
- **External.** Best weighting strengths sit below full balancing (focal α ≈ 0.75, square-root
  sampling, seesaw p = 0.8, class-balanced β < 0.9999); progressive sampling beat class-balanced
  sampling in Kang et al. Weighting cannot change population-optimal within-label AUC, so its
  effects come from finite training and gradient variance
  ([weighting evidence](../research/auc_loss_approaches.md#weighting-strength-evidence-2026-09-21)).
- **Old loss-campaign verdicts do not transfer.** Focal (−0.0026), soft-mass balancing (−0.0092),
  biproportional weights (−0.0043) and the prior offset (−0.0119) were judged on report-derived
  fold-0 validation on `all_5fold_v3`, which is the judge the exponent result discredits for
  weighting.

**Design constraint.** The adaptive signal must not reward agreement with report labels:
up-weighting high-loss or lagging rare positives would chase the labeller the leaderboard
penalises. Candidates therefore relax the weights over training or down-weight persistent
disagreements.

**Candidates** (hypotheses; one run each):

1. **Scheduled exponent.** k changes per epoch, balancing early so rare labels are learned and
   relaxing late, e.g. 1 → 0. The reverse direction, deferred re-weighting
   ([Cao et al. 2019](https://arxiv.org/abs/1906.07413)), won in long-tail vision, so the
   direction is two-sided; run the evidence-motivated 1 → 0 first. Needs epoch-dependent weights
   in the training loop and resume checks that record the schedule.
2. **Noise-aware down-weighting.** Reduce the weight of rare-label cells the model confidently
   contradicts over training: [self-paced loss correction](../research/auc_loss_approaches.md#12-self-paced-loss-correction--splc),
   small-loss selection, or bootstrapped targets β·y + (1 − β)·p
   ([Reed et al. 2014](https://arxiv.org/abs/1412.6596)). Aimed at the gold-versus-report
   disagreement; the risk is suppressing genuinely hard positives.
3. **Per-label task weighting.** One adaptive weight per label from training dynamics:
   uncertainty weighting ([Kendall et al. 2018](https://arxiv.org/abs/1705.07115)), which
   down-weights noisy labels, or [GradNorm](https://proceedings.mlr.press/v80/chen18a.html). No
   knee evidence.

**Excluded.** Meta-learning the weights on the gold studies: 58 studies, gold AUC does not track
the leaderboard, and gold weighting is ruled out. Study-exposure allocation (visit frequency) was
dropped from the sampling scope by user decision on 2026-09-21.

**Protocol.** Reference: the current recipe at k=0.5 (`k_05`, S41, 0.945), seed 2263738815. Export
each run at ×4 and τ 0.75 and submit; record fold-0 validation and the nine-extractor rescoring
as secondary diagnostics. Three seeds of one earlier model spanned 0.003 public
([seed controls](../inference/submissions/2026-09-17-seed-control-submissions.md)). An autoresearch
campaign under the shared validation ×1 rule cannot judge this line.

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

OrthoFoundation-L is the strongest direct 2D knee-pretraining lead; MRI-CORE and the released
three-plane OrthoDiffusion weights are higher-cost alternatives within this investigation.
Establish loading, source/competition eligibility and input semantics before allocating runs.
The 3D option needs a separate volume pipeline. Existing historical ranks do not settle transfer
under the current recipe. Training remains user-launched; move this entry to ongoing work if
implementation or experiments start.

### 3D embedding: training-time effect

**Hypothesis; needs multiple seeds.** The retired additive 3D embedding is ignored at inference:
replacing it with the `unknown_3d` vector changed fold-0 macro AUC by +0.000037 [−0.00011,
+0.00017] on the knee-side checkpoint
([result](../validation/results/2026-09-14-knee-side-embedding-result.md)), and the champion scored
0.936 publicly both with and without it
([submission](../inference/submissions/2026-09-15-champion-without-3d-at-inference.md)). Yet the
champion (with 3D, one training) scored 0.936, above all four no-3D trainings of the same recipe
(0.928–0.934, mean 0.931, standard deviation about 0.0025;
[seed controls](../inference/submissions/2026-09-17-seed-control-submissions.md)). The simplest
explanation is a lucky run; a training-time effect is possible but unproven.

- **Possible mechanisms:** scaffolding (position helps attention and the backbone learn
  position-aware features early, after which visual features carry the information and 3D is
  ignored); regularisation (per-study embeddings that flip with reflection change token norms,
  normalisation statistics and attention temperature); a different random stream (adding a
  module changes initialisation draws, so a 3D and a no-3D model at the same seed are effectively
  different seeds, which alone could explain the public gap).
- **Change:** restore the embedding from the champion's source snapshot
  (`outputs/training/reflection-sigma40/source_snapshot.zip`) behind a three-way option: real 3D,
  placebo (same module, every token gets `unknown_3d`, so random-number use and capacity match)
  and none (today's model).
- **Test:** three arms × about three seeds (nine runs, about 4.5 hours, one overnight queue). Real
  versus placebo isolates the information at an equal random stream; placebo versus none isolates
  the module. Fold 0 probably shows a tie again, so the evidence would come mostly from public
  scores (three submissions per arm). To check scaffolding directly, save a checkpoint every epoch
  in the real-3D arm and run the 3D removal on each: removal should hurt early and not at the end;
  logging the gradient norm reaching the embedding shows the same.
- **Priority:** low; do it only to settle the question. Seed-to-seed spread (fold 0 0.0032,
  public up to 0.006) is the context for any result.

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
