# Big bets

Investigation-sized plans that need focus sessions with the user: diagnostics, a design, several
runs or a new preprocessing identity. Start one only in a session with the user; when it starts,
move its entry to [ongoing work](../ongoing/INDEX.md) and remove it from this file in the same change.
Entries under [parked](#parked-until-a-specific-need) wait for a concrete need before they become
candidates. Each entry says whether its motivation is measured or a hypothesis.

## Position and knee side

**The supervised anatomical-attention sequence is closed:** learned family strengths
with fixed 20 mm width were adopted; appearance-only pooling and learned widths
were discarded. [Evidence](../validation/results/2026-09-21-anatomical-attention-result.md) owns these
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
- **Learned landmarks or localization supervision** (approach 7, which depends on a
  rules-compatible knee segmenter). Common knee orientation (approach 4) ran in the
  [knee-geometry campaign](../validation/results/2026-09-24-knee-geometry-campaign.md) as the
  sagittal-normal (+0.0006) and coronal condylar (−0.0022) frames, neither adopted.
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

- **3D position from the computed knee centre** (measured, three ties on fold 1): replacing the
  1D and 2D embeddings −0.0008 and −0.0002, adding beside them +0.0003, no position at all
  −0.0005; Lateral OA is the one robust gain (+0.0073). Three retakes were queued and never run
  (position dropped for 15% of studies, position in the attention path only, wavelengths from
  32 mm); the audit expects little of them
  ([result](../validation/results/2026-10-05-fixes-and-removals-campaign.md#what-the-runs-established),
  [seeds](../../autoresearch/campaigns/fixes-and-removals/position-3d-retakes-ideas.yaml)).

The [knee-geometry campaign](../../autoresearch/campaigns/knee-geometry/INDEX.md), stopped
2026-09-24, left the curve-density sampler and the fallback-safe OA readout untried, and
`meniscus-core-band-contrast-readout` trained but was never compared.

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

## Auxiliary-loss campaign

**Requested by the user on 2026-09-22; undefined and unstarted.** An autoresearch campaign on
training-only auxiliary objectives added beside the unchanged disease loss. Its premise is a
hypothesis: no auxiliary objective had trained active in this project when it was requested.

- **Scope:** extra heads or loss terms on the shared features, discarded at inference, so the
  submission's runtime does not change. The class-balanced soft BCE, its twelve targets, gold
  overrides, sampler, budget and evaluation stay fixed. Objectives and coefficients are shared
  across labels, never selected or weighted per label. Each candidate fixes its coefficient before
  launch; coefficient sweeps are not ideas.
- **Seed candidates** (all would transfer to the campaign when it is defined):
  [report-state prediction](small-changes.md#auxiliary-report-state-prediction) (hypothesis, with
  a measured judge risk: it teaches reporting behaviour that fold 0 rewards and the leaderboard
  may not) and per-view disease classifiers (below). Acquisition or geometry pretext tasks
  (plane, contrast, slice position, knee side) are guesses with no measured premise. Already done:
  [report-attribute heads](../modeling/report-attribute-heads.md) were adopted on 2026-09-25, and
  the [auxiliary-signals campaign](../validation/results/2026-09-27-auxiliary-signals-campaign.md)
  tested attribute-target variants and scanner heads, adopting the scanner confusion head.
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

## Two views per slice

**Untested; the one attempt never switched its second stream on (added 2026-10-01).** Read every
slice at the adopted 120 mm and again at 140 mm, so each label can use detail and the wider
window on the same slice without halving depth density.

- **Evidence for:** taking each label's better field from the single-field runs would be worth
  about +0.002 over the 140 mm reference, against +0.0011 for 120 mm alone: synovitis, Baker's
  cyst and the medial meniscus prefer 140 mm. The 120/160 mix scored well above the average of
  its two fields ([result](../validation/results/2026-10-01-mixed-field-campaign.md#what-the-runs-established)).
- **Evidence against:** the contrast split, which gives each series one of the two fields, only
  tied 120 mm alone, and one backbone per field lost.
- **What the attempt showed:** with the loader reading each slice once and augmentation on the
  middle slice, a ResNet-18 at 120 mm plus a ResNet-10t at 140 mm takes about 173 s per epoch
  (1.4× the reference). Its pool was `narrow + tanh(gate) × (joint − narrow)` with a zero gate,
  which starved the second stream of gradient.
- **Design constraint:** both streams feed the joint pool from the first step, with no gate; the
  activation check must compare the joint score with the narrow-only score, not test for a
  non-zero value.
- **Test:** one run against the retrained control at its fold and seed.

## View table and label routing

**Configuration and small-mechanism runs on main's view table, after the learnable-views campaign
closed without finding a better table by gradient (2026-10-03).** Results behind every item:
[learnable-views campaign](../validation/results/2026-10-03-learnable-views-campaign.md); the
view table itself: [current model](../modeling/current-model.md#view-table). Limits as in the
last two campaigns: no image above 224 px, 26 slices per study, training time within 1.5 times
the control's.

- **Label bias per slice depth** (measured near miss): one zero-initialised attention offset per
  view, depth bin and label beside the per-view bias. One run on the campaign's build: ×4
  +0.0003, 80% [−0.0001, +0.0006], ×1 +0.0003, ten of twelve labels up. Needs porting to main's
  model; frozen source under
  `outputs/autoresearch/learnable-views/experiments/hand-views-label-depth-bias-20261003T201145Z-feb0600b/`.
- **Wider coronal and sagittal sizes, set by hand** (measured lead, weak): the size menu's draws
  settled on 93 to 120 mm and dropped 70 mm everywhere, and gradient-learned widths settle near
  110 mm, yet frozen at those widths the model scores like the 120 mm crop. Untested between:
  T1/PD at 93 mm, coronal fluid at 105 mm.
- **A balanced batch on widths** (six to eight runs): each run gives every type one of four or
  five widths, arranged so each type's response curve per label comes out of the batch,
  including the untried cells. These runs inform; they are not candidates. The width regression
  (`outputs/analysis/view-regression-20261002/report.md`, refit with
  `scripts/adhoc/fit_view_width_regression.py`) proposes the combination; confirm it with a
  rerun, then outside fold 0.
- **A budget batch:** reallocations at 26 slices from the per-type sensitivity measured on
  Kaggle (`outputs/analysis/slice-budget-20261002/sensitivity/`). Evidence against: the learned
  budget moves a quarter of a slice and loses 0.0016 on the hand views.
- **A view added on top of a type's slices, not carved out of them** (guess): a second
  sagittal-fluid view and a zoomed patellofemoral axial-fluid view both lost on the slices they
  took from the first view (−0.0020 and −0.0013), while the zoomed view moved PF OA by +0.0027
  with the label bias routing it as designed. Untested: the same views with the type's six
  slices kept, which raises the study's slice total and the run time.
- **Routing a view to chosen labels** is not built on main: every view's tokens enter every
  label's pool, and the label bias is the only routing. Checked on CPU only on main: a
  knee-anchored 100 mm axial view, a second 84 mm sagittal T1/PD view 15 mm above the image
  centre and a second 120 mm sagittal fluid view of two slices run through the loader and the
  model.
- **Carried over from smart-crops, not run:** coronal T1/PD back at 120 mm with sagittal kept
  at 84 mm, compartment crops routed to the OA labels, the sagittal windows without the 10 mm
  downward shift (with the sizes pinned, learned shifts dropped most of the coronal 10 mm offset and
  kept it on sagittal fluid), separate normalisation statistics per pixel size.
- **Not confirmed outside fold 0:** about 30 view choices have been made on the same 912
  studies; a second fold or a public submission is the check.

## Hard-pair mining phase

**One unverified external report; untested here (added 2026-09-25).** Another participant says
their best setup ends with 3 epochs of hard mining: score the training set, mine the pairs the model
orders worst, batch each pair together and train on those batches. Their loss and mining schedule
are unknown; ask for both before designing anything.

- **Why batching matters:** our only pair term is the within-microbatch ranking loss (batch size 8,
  about 118 usable pairs per step). With random batches most hard pairs never share a batch, which
  may explain why the always-on hybrid
  [was neutral](../../autoresearch/campaigns/losses/findings.md#completion-20260920t121537z-5b39b418)
  (+0.0007, interval spanning zero). Mining changes the pairs the term sees, not its formula.
- **Without a pair term** it reduces to oversampling hard studies, and the nearest measured
  relative, [shared balanced focal](../../autoresearch/campaigns/losses/findings.md#completion-20260920t124339z-2948d523)
  (up-weighting hard cells), lost 0.0026. The losses campaign's `shared-ranking-memory-bank`
  (pairs against stored scores beyond the batch) is the closest unrun relative, still blocked.
- **Design constraints:** one mining rule shared by all twelve labels; mine only training studies;
  training-set scores partly reflect memorization, so the hardness source is an open choice; hard
  pairs under soft labels can be label noise rather than signal, the
  [OA audit](../history/sharing/oa-and-meniscus-disagreement-audit.md) found such cells.
- **Test:** the incumbent recipe plus a final mined phase against a fresh control at the same
  total budget.

## Patellofemoral OA evidence

**Reported diagnosis, not reproduced here; payoff unmeasured (added 2026-10-01).** An analysis the
user shared says the models rank PF OA mostly from correlated findings, not from the
patellofemoral joint. Its scripts and outputs are not in this repository, so the first gate
reproduces it.

- **Reported, kept:**
  - a coronal-only model scores 0.86 on PF OA against 0.84–0.87 for an axial-only one;
  - once each model's Effusion, Medial OA and Lateral OA outputs are regressed out, its PF
    ranking keeps an AUC of 0.71–0.78;
  - PF OA AUC by scanner runs from 0.79 (MR 5300 1.5T) to 0.97 (Ingenia 1.5T, 71 positives), with
    Skyra 3T and Vantage at 0.82–0.83; positives on the weak scanners were not given.
- **Reported, dropped:** each family's PF prediction correlating more with its own effusion
  prediction (ρ 0.32–0.42) than with the labels (ρ 0.25) compares two continuous predictions with a
  rare binary label, whose correlation is capped by prevalence. Reading the report labels' Gold PF
  score (0.933) and the flat public OA score (0.916 → 0.920) as consequences is speculation.
- **Gate 1, existing fold predictions, no GPU:** the same residualisation for Medial and Lateral
  OA, PF OA predicted from the true Effusion, Medial OA and Lateral OA labels alone, and positives
  with paired intervals per scanner. OA is often multi-compartment, so part of the drop is
  legitimate. Stop if PF's residual is in line with the other two compartments.
- **Gate 2, labels before images:** the PF targets over-call low grades. Gold is positive for 3 of
  11 low-grade or softened PF cartilage reports, and a rule at the official threshold scores 0.947
  on gold against 0.842 for "any finding"
  ([gold check](../ongoing/report-attribute-extraction.md), [measured gaps](../public-labels/gold-versus-silver.md#measured-gaps)).
  If the label is the limit, the fix is the label rule and this entry closes.
- **Gate 3, geometry, label-free:** measure how much of the patellofemoral joint the current axial
  and sagittal inputs cover and the in-plane resolution over its cartilage.
- **Bet (guess, low confidence), only if gates 1 and 3 agree:** a PF-targeted axial input, one run
  against the control at its fold and seed. Evidence against: the OA detector, which read PF OA
  over tokens in front of the knee centre, was [removed at a tie](../validation/results/2026-10-05-fixes-and-removals-campaign.md), and
  [per-label crops](../validation/results/2026-09-16-per-label-crops-result.md) moved PF OA by
  +0.0018 against crop224 while trailing `00-baseline` by 0.0028.
- **Tried once without the gates (2026-10-03):** an 84 mm axial-fluid view 21 mm anterior of the
  knee centre, taking three of the type's six slices: PF OA +0.0027 (95% [−0.0038, +0.0097]),
  macro −0.0013 with contusion, MCL and fracture down
  ([learnable-views campaign](../validation/results/2026-10-03-learnable-views-campaign.md#details-of-the-runs-against-the-hand-reference)).

## Pooling head at the inference bag size

**Proposed 2026-10-04; hypotheses, nothing run.** The attention temperature compensates for a head
trained on ×1 bags and used on ×4 to ×6 bags; these two remove the mismatch instead. Tuned τ is
worth +0.0011 over τ 1 on the ten-children blend at ×6
([result](../validation/results/2026-10-04-two-family-oof-distillation.md)), so the expected prize is of that order.
The inference-only alternatives are in
[small changes](small-changes.md#pooling-dense-bags-at-inference); the earlier measurements are
on the `better-than-tau` branch (`docs/ongoing/dense-bag-pooling.md` there, uncommitted).

1. **Refit only the pooling head on dense bags** (Touvron et al. 2019, "Fixing the train-test
   resolution discrepancy"). Freeze the backbone, cache the training folds' tokens at ×4 to ×6
   and retrain the attention scorer, view label bias and head, or in the small form only a
   sharpness per label plus the view label bias (about a hundred parameters). Risk: the cached
   training-fold features are ones the backbone has already fit, so the refit sees easier data
   than the test set. Gate: the refit head at τ 1 against the original head at its tuned τ, on the
   held-out fold.
2. **Dense forward, gradient through a subset** (Bergner et al. 2023, Iterative Patch Selection;
   Katharopoulos and Fleuret 2019, attention sampling). Encode the dense bag without gradient and
   backpropagate through a ×1-sized subset, so the head sees inference-sized bags during training.
   Full retraining at several times the cost; an earlier variable-bag attempt failed and was not
   recorded. Only after item 1 shows the head is the bottleneck.

Not pursued, with the reason: sparse attention (the effective share of attended tokens stays at
about 13% at every bag size, so no tail grows), averaging training-sized sub-bags (worse than
joint pooling here and in the whole-slide literature), further rules from token count to τ (the
log-n rule already takes what the count explains), and dense-to-sparse distillation (the children
already train ×1 bags against ×4 teacher predictions and still want τ 0.7 to 0.9).

## Per-plane teacher readings

**Proposed 2026-10-04 (user idea); hypotheses, nothing run.** With the plane-contrast readouts a
label's logit is a sum over tokens of attention times the evidence of the token's series type,
and `pool_encoded(encoded, view=...)` pools one view alone, so a teacher can give six readings
per label instead of one. No run has saved them; every checkpoint of the
[two-family distillation](../validation/results/2026-10-04-two-family-oof-distillation.md) can
regenerate them at the cost of one ×4 prediction per fold.

1. **First measurement:** the six single-type predictions of each teacher on its validation
   fold: AUC per label and type, how often the types disagree, and whether disagreement marks
   the cells where the silver label and the 58 gold studies differ.
2. **A per-cell teacher weight:** trust the teacher where its planes agree against the silver
   label and keep the silver label where they disagree, in place of the fixed 0.5. A label-table
   change only.
3. **A target per plane:** train each series type's readout alone against the teacher's reading
   from that type, beside the joint target (72 soft targets per study). An auxiliary loss behind
   one option; overlaps the per-view classifiers of the
   [auxiliary-loss campaign](#auxiliary-loss-campaign).
4. **Co-training between planes:** each type's target is the consensus of the other types.
   Labels visible in one plane only have nothing to learn from the others.
5. **Routing labels to their best planes** when building the teacher label; the view label bias
   already learns a version of this.

## Anatomy masks: what is left untried

From the working notes of 2026-10-07 on exploiting a pretrained segmenter offline, after the
[anatomy-masks campaign](../../autoresearch/campaigns/anatomy-masks/INDEX.md) used the bone
masks for the map, the plane, the centre, the sections, each token's position in the bone frame
and the posed crops. All are guesses about the score; the segmenter never runs at prediction.

- **Soft-tissue masks.** Menisci, cartilage and cruciates from `aagatti/nnunet_knee`, PYCAD Knee
  or the SKM-TEA models, in place of bone-only MRAnnotator: the recipe already routes each head
  through sections read from the bone map (femur, joint, tibia, kneecap); these masks would let
  the meniscus, OA and ACL targets read the structures themselves, and give cartilage volume per
  compartment as a target. Licence and training data of each model to be checked first (the bone
  model passed the host's test).
- **Masks for every training knee.** About 400 of 3,500 training knees carry targets; the map
  already reaches 0.90 overlap on femur and tibia with them, so what the rest would add is
  unknown.
- **Segmentation before the labels.** A stage that teaches the backbone to segment knees, or a
  full decoder beside the labels dropped at prediction, instead of the per-token map head.
## Parked until a specific need

### Fused study volume and 3D model

**Hypothesis; prototype fusion measured on one study, value on the metric unmeasured (2026-09-23).**
Fuse a study's 2D stacks into one volume on a shared patient-coordinate grid and train a 3D model
on it: one fused fluid volume, one volume per contrast as channels, or each series resampled onto
the grid as its own channel with missing-series masks. A one-study prototype (study
`…26592250`, three GE 4 mm fluid stacks) registered each stack rigidly on the GPU by exhaustive
±8 mm translation search plus refinement, then fused with a super-resolution solve on a 0.8 mm
grid. Header placement was already correct within the 1.4 mm loop-closure precision, so no motion
correction was needed there; the fused volume streaks where 3–4 mm slices leave gaps and loses the
native in-plane detail. Costs on the 3090: about 26 s per registered pair unoptimised and 3–4 s
per fusion.

Evidence against the main benefit: cross-series 3D position, through the additive 3D embedding and
the token graph, was worth about 0 and was removed
([champion without 3D](../inference/submissions/2026-09-15-champion-without-3d-at-inference.md)).
A 3D network also gives up most of the 2D ImageNet pretraining; the realistic encoders reuse 2D
weights (inflated kernels, a 2D encoder with 3D aggregation, or the
[knee-specific 3D lead](../research/huggingface-image-encoders.md#knee-specific-3d-lead)). Fusion
adds preparation time that counts toward the efficiency prize. The
[volumetric CNN brief](../research/volumetric-cnns.md#6-volume-construction-and-geometry) proposes
the other route first: one native-plane volume per acquisition, fused at the feature level,
without registration or a study cube. This entry is the fused-cube alternative.

- **First gate, no training:** score the frozen incumbent on fold 0 with triplets resliced from
  header-placed fused volumes instead of native slices. A large drop means fusion loses what the
  model uses and a 3D model would start handicapped; about an hour of GPU for 905 studies.
- **Most plausible use:** a deliberately different member of the final ensemble, which belongs to
  the [endgame](endgame.md).

### External pretrained image encoders

**Survey completed 2026-09-20; implementation and training unstarted.** The
[checkpoint review](../research/huggingface-image-encoders.md) owns sources, loading contracts,
revision pins and unresolved terms. Investigate only when a backbone-diversity experiment is
prioritized: DINOv3 ConvNeXt-T first for practical integration, then RadioDINO-S against the
same architecture's general-image weights. OmniRad-S is an alternative after its conflicting
license metadata is resolved. These require spatial mappings or patch-token adapters; they are
not hyperparameter-only runs. Hold the current balanced-BCE recipe and data contract fixed,
compare fresh controls, and record per-label AUC, ensemble contribution, time and memory.
The [RadImageNet comparison](../ongoing/radimagenet-backbone.md), now implemented, is the one
medical-pretraining measurement with an in-competition claim behind it.

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
