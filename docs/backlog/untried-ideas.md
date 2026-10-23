# Untried ideas

Ideas that were never run by the end of the competition and that had a stated reason or evidence
behind them. Each says whether it is **evidence-backed** (a measured premise or a near result in
this project) or **speculative** (a hypothesis, literature or external report only). Ideas that ran
live in their result records ([validation results](../validation/results/INDEX.md), campaign
records under `autoresearch/campaigns/`); hidden-test risks that were measured or guarded live in
[hidden-test risks](../inference/hidden-test-risks.md).

## Position and knee side

- **Side-conditioned sagittal sampling centre** (evidence-backed premise, payoff unmeasured).
  Sagittal slices step along patient L, the compartment labels sit about 31 mm off the study centre,
  and the sagittal draw is centred on it. The
  [spatial importance map](../data-preparation/peripheral-coverage.md#spatial-importance-map)
  shows a central trough on that axis alone (flanks minus centre +0.126, against −0.634 coronal and
  −1.099 axial), but it pools left and right knees, so "medial and lateral matter equally" and
  "only medial matters, seen twice" look the same. The map must be re-exported with left knees
  mirrored (the per-study utilities live outside this repository) before the offset, which flips
  sign per knee side and is zero for side abstentions, can be designed. Side policy:
  [audit](../metadata/knee-side/knee-side-policy-audit.md).
- **3D position retakes** (speculative). 3D position from the computed knee centre tied on fold 1
  (replacing the embeddings −0.0008 and −0.0002, beside them +0.0003, none −0.0005; Lateral OA
  +0.0073). Three retakes were queued and never ran: position dropped for 15% of studies, position
  in the attention path only, wavelengths from 32 mm; the audit expected little
  ([result](../validation/results/campaigns-and-queues/2026-10-05-fixes-and-removals-campaign.md#what-the-runs-established),
  [seeds](../../autoresearch/campaigns/fixes-and-removals/position-3d-retakes-ideas.yaml)).
  Research sources: [position encoding](../research/geometry-and-views/position-encoding.md#5-proposed-sequence-and-what-was-run),
  [knee-side approaches](../research/geometry-and-views/knee-side-and-location.md).

## Views and fields

- **112 mm field with every view at 0.5 mm** (speculative, low to medium). 0.5 mm is the most common
  acquired resolution: half of all series are fully resolved at it against 30% at 0.536 mm. Against
  it: 112 mm cuts more than 20 mm from the front of the knee in about 15% of sagittal series (8% at
  120 mm), and the contrast split with T1/PD at 112 mm only tied 120 mm alone
  ([result](../validation/results/crops-and-views/2026-10-01-mixed-field-campaign.md#what-the-runs-established),
  [geometry](../research/geometry-and-views/input-resolution.md#what-a-narrow-field-cuts)).
- **Two views per slice, ungated** (evidence-backed). Read every slice at 120 mm and again at
  140 mm through two streams. Taking each label's better field from the single-field runs would be
  worth about +0.002 over 140 mm (synovitis, Baker's cyst and the medial meniscus prefer 140 mm).
  The one attempt, `two-stream-paired` (ResNet-18 at 120 mm plus ResNet-10t at 140 mm, about 1.4×
  the reference's epoch time), pooled `narrow + tanh(gate) × (joint − narrow)` with a zero gate, so
  the second stream never received gradient. Both streams must feed the joint pool from the first
  step, and the activation check must compare the joint score with the narrow-only score
  ([result](../validation/results/crops-and-views/2026-10-01-mixed-field-campaign.md#what-the-runs-established)).
- **A view added on top of a type's slices, not carved out of them** (speculative, with a lead). A
  second sagittal-fluid view and a zoomed patellofemoral axial-fluid view lost on the slices they
  took from the first view (−0.0020 and −0.0013), while the zoomed view moved PF OA by +0.0027.
  Untested: the same views with the type's six slices kept, which raises the slice total and run
  time ([learnable-views campaign](../validation/results/crops-and-views/2026-10-03-learnable-views-campaign.md)).
- **Label bias per slice depth** (evidence-backed near miss). One zero-initialised attention offset
  per view, depth bin and label beside the per-view bias. One run on the campaign's build: ×4
  +0.0003 (80% [−0.0001, +0.0006]), ×1 +0.0003, ten of twelve labels up. It needs porting to main's
  model; the frozen source is in
  `outputs/autoresearch/learnable-views/experiments/hand-views-label-depth-bias-20261003T201145Z-feb0600b/`
  ([campaign](../validation/results/crops-and-views/2026-10-03-learnable-views-campaign.md)).
- **Wider coronal and sagittal widths, set by hand** (evidence-backed, weak). The size menu's draws
  settled on 93 to 120 mm and dropped 70 mm everywhere, and gradient-learned widths settled near
  110 mm, yet frozen at those widths the model scored like the 120 mm crop. Untested between: T1/PD
  at 93 mm and coronal fluid at 105 mm ([campaign](../validation/results/crops-and-views/2026-10-03-learnable-views-campaign.md),
  [view sizes](../validation/results/crops-and-views/2026-10-10-view-sizes.md)).
- **A balanced batch on widths** (six to eight runs; they inform, they are not candidates). Each
  run gives every type one of four or five widths, arranged so each type's response curve per label
  comes out of the batch, untried cells included. The width regression
  (`outputs/analysis/v/view-regression-20261002/report.md`, refit with
  `scripts/adhoc/fit_view_width_regression.py`) proposes the combination; confirm it with a rerun,
  then outside fold 0.
- **A slice-budget batch** (speculative): reallocations at 26 slices from the per-type sensitivity
  measured on Kaggle (`outputs/analysis/s/slice-budget-20261002/sensitivity/`, built by
  `scripts/adhoc/build_slice_budget_kaggle_run.py` and `slice_budget_kaggle/`, analysed by
  `analyze_slice_budget_sensitivity.py`). Against it: the
  learned budget moved a quarter of a slice and lost 0.0016 on the hand views.
- **Routing a view to chosen labels** (speculative). On main every view's tokens enter every label's
  pool and the label bias is the only routing. Three views were checked on CPU only: a
  knee-anchored 100 mm axial view, a second 84 mm sagittal T1/PD view 15 mm above the image centre
  and a second 120 mm sagittal fluid view of two slices; each ran through the loader and the model.
- **Carried over from smart-crops, not run** (speculative): coronal T1/PD back at 120 mm with
  sagittal kept at 84 mm; compartment crops routed to the OA labels; sagittal windows without the
  10 mm downward shift (with the sizes pinned, learned shifts dropped most of the coronal 10 mm
  offset and kept it on sagittal fluid); separate normalisation statistics per pixel size
  ([campaign](../validation/results/crops-and-views/2026-10-02-smart-crops-campaign.md)).
- **Caveat on every view choice:** about 30 view choices were tuned on the same 912 fold-0
  studies and none was confirmed on another fold
  ([current model](../modeling/current-model.md#view-table)).
- **Bigger images at the same field** (evidence-backed, weak). Every recipe trained at 224 px; 120 mm
  at 0.536 mm took the resolution gain by narrowing the field (+0.004 public, S90 against S86). At
  that pixel size 140 mm is 261 px and 160 mm is 299 px; at 224 px a 160 mm field added nothing
  in five designs ([result](../validation/results/crops-and-views/2026-10-01-mixed-field-campaign.md)).
  Supporting evidence: per-label crops at 0.5 mm gained on ACL (+0.0111), synovitis (+0.0077) and
  medial OA (+0.0057) over `00-baseline`
  ([result](../validation/results/crops-and-views/2026-09-16-per-label-crops-result.md#per-target-auc));
  0.5 mm keeps the full acquired band in 59% of series against 23% at 0.625 mm; a public thread
  reported 0.936 at 224 px and 0.942 at 288 px for a small ResNet
  ([public research](../research/competition-and-prior-art/public-research.md#validation-and-ensembling)).
  Cost: 1.56× the pixels at 280 px; the estimated 31 GB peak at batch 8 needs accumulation or
  activation checkpointing on a 24 GB GPU (the removed `model.checkpoint_backbone` is at
  `7770ce9:src/knee_abnormality/modeling/model.py`); host memory:
  [larger inputs](../modeling/training-runtime-and-memory.md#host-memory-with-larger-inputs).
- **36 training triplets on the final recipe** (evidence-backed, one pair). On the RadImageNet
  ResNet-18, `data.study_slice_total` 26 → 36 gained +0.0018 at ×4 (0.927806 → 0.929586; 80%
  [+0.0002, +0.0034], 95% [−0.0005, +0.0041]) for 23% more training time; OA rose (lateral +0.0094,
  PF +0.0036, medial +0.0026) and both menisci fell (lateral −0.0049, medial −0.0033). Every later
  recipe kept 26 ([campaign](../../autoresearch/campaigns/radimagenet-rn18/INDEX.md)).

## Pooling and the attention head

- **Refit the pooling head on dense bags** (evidence-backed). The head trains on ×1 bags and
  predicts on ×4 to ×6; one attention temperature compensates. Tuned τ was worth +0.0011 over τ 1
  on the ten-children blend at ×6
  ([result](../validation/results/labels-and-losses/2026-10-04-two-family-oof-distillation.md)), and
  the best τ per label differs between models while holding on unseen studies of the same model
  (+0.00076 on fold 0) ([result](../validation/results/pooling-and-attention/2026-10-12-dense-bag-pooling-on-the-resnet-50.md)).
  Freeze the backbone, cache training-fold tokens at ×4 to ×6, and retrain the scorer, view label
  bias and head, or only a sharpness per label (about a hundred parameters); gate: refit head at
  τ 1 against the original at its tuned τ on the held-out fold (Touvron et al. 2019). Fold-0 tokens
  were cached at ×1 to ×8 in `outputs/analysis/pooling-token-cache/fold0`. A full-retrain follow-up
  is a dense forward with gradient through a ×1-sized subset (Bergner et al. 2023). Ruled out with
  reasons: sparse attention (the attended share stays near 13% at every bag size), averaging
  training-sized sub-bags, further token-count-to-τ rules, dense-to-sparse distillation.
- **Training with the attention temperature** (speculative). τ was only ever an inference override;
  whether training at it moves the optimum was never tested
  ([result](../validation/results/pooling-and-attention/2026-09-17-attention-temperature.md)).
- **Nonlinear readout after pooling** (speculative). A small `256 → 128 → 1` GELU MLP per label in
  place of each linear readout, keeping pooling and supervision; it isolates readout capacity and
  differs from the [rejected compartment adapter](../validation/results/pooling-and-attention/2026-09-20-compartment-adapter-result.md).
- **Plane/contrast-specific feature adapters** (speculative). A zero-initialised residual
  bottleneck per series type after the shared projection, before attention, so it can change
  which evidence is selected (after AnyMC3D's view-specific LoRA,
  [research](../research/backbones-and-pretraining/pretraining-and-external-data.md#adapting-2d-foundation-models-to-volumes)).
  Type-specific readouts after pooling lost −0.0006
  ([near-misses campaign](../validation/results/campaigns-and-queues/2026-09-25-near-misses-campaign.md)).
  Compare against a shared adapter of equal parameter budget.

## Supervision and labels

- **Graded uncertainty weight on Synovitis and Effusion only** (evidence-backed). Weighting every
  report cell by its distance from 0.5 raised Synovitis (+0.0047) and Effusion (+0.0017) but lost
  0.0010 macro through PF OA and labels with few uncertain cells
  ([result](../validation/results/campaigns-and-queues/2026-10-13-targets-and-inputs-campaign.md#targets)).
- **Training-label exclusions** (evidence-backed composition, effect expected below noise). About
  19 studies (0.4%) whose labels likely do not describe their images: one proven cross-study pixel
  copy whose report describes the other knee
  ([findings](../metadata/repeated-series.md#cross-study-repeat)); six empty or cut-off reports
  (four Dutch ending at or before `Bevindingen:`, one `Bevindingen: aa`, one Spanish
  `Resultados: . Impresión:`; 2 in fold 0, none gold; `combined_v1` labels 0.25–0.35); and 13
  non-Bulgarian studies whose report and header name opposite knees (the 13 Bulgarian ones are
  template text, keep them). Lists: `outputs/analysis/label-hygiene-20260922/`.
- **Auxiliary-loss campaign** (speculative; the user requested it on 2026-09-22 and it was never
  defined). Training-only auxiliary heads or loss terms on the shared features, discarded at
  inference so the runtime does not change. Scope rules: the class-balanced soft BCE, its twelve
  targets, gold overrides, sampler, budget and evaluation stay fixed; objectives and coefficients
  are shared across labels, never selected or weighted per label; each candidate fixes its
  coefficient before launch, and coefficient sweeps are not ideas. Log every loss component and
  show that the auxiliary term is active: the medial/lateral exclusivity penalty never became
  active (about 1e-6 against a main loss of 0.47,
  [follow-up](../../autoresearch/campaigns/medial-lateral-transfer/prior-work.md#corrected-exclusivity-follow-up)),
  and the [shared-loss campaign](../../autoresearch/campaigns/losses/INDEX.md) replaced the main
  objective rather than adding one (ranking inconclusive, focal negative). Seed candidates: the
  report-state head below and per-view disease classifiers; acquisition or geometry pretext tasks
  (plane, contrast, slice position, knee side) are guesses with no measured premise. Already
  done: [report-attribute heads](../modeling/report-attribute-heads.md) were adopted on 2026-09-25
  and the [auxiliary-signals campaign](../validation/results/campaigns-and-queues/2026-09-27-auxiliary-signals-campaign.md)
  adopted the scanner confusion head. To define it, follow the campaign setup in
  `autoresearch/campaign-setup.md` with the incumbent as reference; the policy covers the loss
  directory, `training/lightning.py`, `config.py`, `trainer.loss`, plus `modeling/model.py` for
  the heads.
- **Per-view disease classifiers** (speculative, low to moderate confidence; from the RSNA 2024
  lumbar winner's modality-classifier auxiliaries, whose
  [ablation](https://files.speakerdeck.com/presentations/3a7b5a2c84e14df2a1bce0b41dd609d0/slide_29.jpg)
  bundles them with other objectives and does not isolate them). For each present plane/contrast
  type, pool that type's existing tokens into a bag and predict all twelve study labels from it,
  masking absent types; no view/label subset is selected. The main global head, its attention
  normalisation and anatomical bias stay unchanged; the backbone pass is reused and the heads
  dropped at inference. Average the auxiliary loss over present views so studies with more types
  gain no extra weight. A positive study label need not be visible in every view, so the
  auxiliary targets carry noise. Series dropout, which also forced predictions from subsets of
  views, was turned off on 2026-09-29 (+0.0023 without it) and removed on 2026-10-14
  ([adoption history](../modeling/current-model.md#adoption-history)).
- **Auxiliary report-state head** (speculative, with a measured risk). Three extra logits per label
  for `{not reported, negative, positive}` from Steven v1, beside unchanged disease supervision
  ([motivation](../public-labels/uncertainty-and-confidence-encoding.md#report-state-audit-and-implications-for-supervision)).
  Risk: on fold 0 the worst-ranked negatives are unaddressed findings and 14% of such gold cells
  are positive ([report attributes](../public-labels/report-attributes.md#agreement-with-the-incumbents-predictions)),
  so a gain may reward reporting behaviour rather than disease.
- **Balanced target columns by masking** (speculative). Per label, keep all minority cells and an
  equal random sample of majority cells in the loss each epoch, keeping soft values. Seeded as
  `balanced-target-columns` in the targets-and-inputs campaign and retired unrun.
- **Per-label teacher weights from out-of-fold AUCs** (speculative). When several teachers are
  blended into targets, weight each label by each teacher's out-of-fold strength; models trade
  labels (plane-contrast readouts: Lateral OA +0.0054, Fracture −0.0024). Never fit them on gold:
  gold-fitted member weights lost
  ([result](../validation/results/validation-and-ensembles/2026-10-16-gold-fitted-stacking.md)).
- **Per-plane teacher readings** (speculative; proposed by the user on 2026-10-04, nothing run).
  With the plane-contrast readouts a label's logit is a sum over tokens of attention times the
  evidence of the token's series type, and `pool_encoded(encoded, view=...)` pools one view alone,
  so a teacher can give six readings per label instead of one. No run saved them; every checkpoint
  of the [two-family distillation](../validation/results/labels-and-losses/2026-10-04-two-family-oof-distillation.md)
  can regenerate them at the cost of one ×4 prediction per fold. In order:
  1. First measurement: the six single-type predictions of each teacher on its validation fold,
     AUC per label and type, how often the types disagree, and whether disagreement marks the cells
     where the silver label and the 58 gold studies differ.
  2. A per-cell teacher weight: trust the teacher where its planes agree against the silver label
     and keep the silver label where they disagree, in place of the fixed 0.5. A label-table change
     only.
  3. A target per plane: train each series type's readout alone against the teacher's reading from
     that type, beside the joint target (72 soft targets per study); an auxiliary loss behind one
     option that overlaps the per-view classifiers above.
  4. Co-training between planes: each type's target is the consensus of the other types; labels
     visible in one plane only have nothing to learn from the others.
  5. Routing labels to their best planes when building the teacher label; the view label bias
     already learns a version of this.
- **Hydra-style student, one head per teacher** (speculative, literature). A shared body with one
  head per teacher family, each trained on its own teacher's out-of-fold predictions (0.25 silver +
  0.75 teacher, gold rows keep gold), heads averaged at inference (Hydra, Tran et al. 2020,
  arXiv 2001.04694). Expected gain small: model disagreement is weaker than human disagreement.

## Augmentation, sampling and intensity

- **Intensity and noise augmentation on fluid series only** (evidence-backed). Removing intensity,
  noise, GIN or resolution degradation lowered Effusion at ×4 in all four runs (−0.0019 to
  −0.0035) and raised Fracture in all four (+0.0012 to +0.0079)
  ([result](../validation/results/campaigns-and-queues/2026-10-13-targets-and-inputs-campaign.md#inputs-and-augmentation)).
- **Transform ordering and interactions** (speculative). Nine image families run in a fixed order
  and the [augmentation campaign](../validation/results/campaigns-and-queues/2026-09-26-augmentation-campaign.md)
  showed their gains do not add. Inspect transforms with the
  [augmentation viewer](../data-preparation/reflection-geometry-and-previews.md#reflection-preview-evidence)
  before changing a family.
- **Per-slot budget split between a slot's own series** (speculative). The one part of the
  repeated-series evidence that does not starve other types; every other budget question closed
  in the [triplet-budget investigation](../history/triplet-budget-allocation.md).
- **Learned acquisition intensity bounds** (speculative). A shared predictor on unclipped float
  input in place of fixed percentile clipping; other fixed quantiles lost (0.001/0.999 −0.0003,
  0.01/0.99 −0.0021, [result](../validation/results/campaigns-and-queues/2026-09-17-overnight-results.md)).
  Rejected source: DICOM `WindowCenter`/`WindowWidth` (30 series, 2026-09-20) are a median 1.43×
  wider than the 0.005/0.995 bounds, vendor-dependent (Toshiba 1.77, Philips 1.20) and vary slice
  to slice in 23 of 30 series.

## Recipe removals never tested alone

Queued by the [fixes-and-removals campaign](../validation/results/campaigns-and-queues/2026-10-05-fixes-and-removals-campaign.md)
and never run (speculative: each was adopted on an earlier backbone or in a bundle):
the **half-knee loss** and the **meniscus own-side attention mask**
([seeds](../../autoresearch/campaigns/fixes-and-removals/seed-ideas-3.yaml)).

## Anatomy masks

From the [anatomy-masks campaign](../../autoresearch/campaigns/anatomy-masks/INDEX.md) notes
([run notes](../../autoresearch/campaigns/anatomy-masks/run-notes.md)); all speculative about the
score:

- **Region membership over the token's footprint** (source-backed): membership is decided at the
  token centre though a token is 12 to 13 mm wide (`regions-integrate-the-token-footprint`).
- **Across axis from the joint mask's covariance** (`joint-map-covariance-sets-the-across-axis`).
- **One plane and centre per knee in two-knee studies** (source-backed): they are fitted over both
  knees pooled; about eight training studies.
- **Joint gap as a class or relative to the other side**: in millimetres the head recovered a
  fifth of a target that varies by under a millimetre.
- **Structure ablation** (read-only diagnostic): drop one predicted anatomy class's tokens at
  validation and see which targets fall; where attention falls is already in
  `outputs/analysis/f/frame-and-targets/anatomy-sections-audit/`.
- **Cleaning rules checked against the masks**: series where no knee is found point at wrong body
  parts or bad scans; rule-based checks only, no trained model in the loader.
- **Masks for every training knee**: about 400 of 3,500 knees carry mask targets and the map
  already reaches 0.90 overlap on femur and tibia.

## Patellofemoral OA

A diagnosis the user shared (not reproduced; scripts outside the repository) says models rank PF OA
mostly from correlated findings: a coronal-only model scores 0.86 on PF OA against 0.84–0.87
axial-only; residualised on Effusion, Medial and Lateral OA outputs the PF ranking keeps 0.71–0.78;
PF AUC by scanner runs 0.79 (MR 5300 1.5T) to 0.97 (Ingenia 1.5T). Gates in order (speculative):
the same residualisation for Medial and Lateral OA on existing fold predictions; the label rule
first, since gold is positive for 3 of 11 low-grade PF reports and a rule at the official threshold
scores 0.947 on gold against 0.842 for "any finding"
([measured gaps](../public-labels/gold-versus-silver.md#measured-gaps)); then PF joint coverage of
the axial and sagittal inputs; only then a PF-targeted axial input. The one ungated try is the
zoomed axial view above (PF +0.0027, macro −0.0013).

## Ensembles

- **A mask-trained (heated) ResNet-50 member** (evidence-backed on one fold). On the ResNet-10t an
  equal blend of standard-weights and heated-weights runs scored +0.0011 to +0.0012 over the
  standard run, though the heated runs alone were 0.0017 and 0.0022 below it; a second
  standard run added 0.0000 to 0.0005
  ([result](../validation/results/campaigns-and-queues/2026-10-05-pretraining-input-campaign.md#findings)).
  The two ResNet-50 runs on these weights were stopped before completion
  ([result](../validation/results/backbones/2026-10-04-refined-radimagenet-campaign.md#what-stayed-open)).
- **Rank-blend with the public stack** (speculative, external). Top public notebooks reached 0.945
  by rank-blending the 0.943 public stack with one private model at weight 0.45–0.6; prize
  eligibility of using others' public weights was unanswered
  ([re-sweep](../research/competition-and-prior-art/2026-09-28-public-resweep.md#public-0945-the-public-stack-rank-blended-with-a-private-model)).

## Inputs beyond the 2D recipe

- **Reformatted fluid planes from 3D series** (evidence-backed premise, ceiling about +0.0004).
  392 training studies (8.9%; 83 of fold 0) lack 2D fluid in one plane but hold an isotropic 3D
  fluid series (245 axial Siemens DESS without sagittal fluid; 147 sagittal 3D PD FS without coronal
  fluid). Removing a present fluid type cost 0.004–0.005 macro
  ([test-time allocation](../validation/results/sampling/2026-09-17-test-time-allocation.md#missing-types)),
  and the affected fold-0 studies do not underperform (`k_05` 0.906 on 54 DESS studies against
  0.9015 on 822 others). Gate: a frozen-checkpoint fill on fold 0. Data:
  `outputs/analysis/a/acquisition-geometry-20260922/`.
- **Fused study volume and 3D model** (speculative). A one-study prototype registered three GE 4 mm
  stacks (header placement already within the 1.4 mm loop-closure precision) and fused them on a
  0.8 mm grid, streaking in slice gaps; about 26 s per registered pair and 3–4 s per fusion on the
  3090. Against: cross-series 3D position was worth about 0 and was removed, and 3D networks lose
  2D pretraining ([volumetric brief](../research/backbones-and-pretraining/volumetric-cnns.md#volume-construction-and-geometry)).
  First gate, no training: the frozen model on triplets resliced from fused volumes.
- **Study overlap by anatomical matching** (speculative). The bounded
  [overlap audit](../metadata/study-overlap-investigation.md) found no confirmed repeat; the
  follow-up would retrieve neighbours by stable bone anatomy, verify by registration across slices
  and planes, and change fold groups only on convincing evidence.

## Inference loading

Measured evidence behind these: [loading cost attribution](../inference/profiling/loading-cost-attribution.md).

- **Central-window neighbour resampling in GPU preparation** (evidence-backed, bit-exact): the
  loader version gained 13% of loader throughput
  ([profile](../inference/profiling/2026-10-12-middle-slice-preparation-and-loader-speed.md)).
- **Intensity bounds from the middle slices only** (speculative, changes images): with
  `repeated_middle` the model reads only the middle slice, so neighbours would not be read at all,
  up to two thirds of slice reads; needs a retrain.
- **Cheaper first look** (speculative, changes the pose): the pose pass was about 73 ms of a
  317 ms profiled training step.
- **`PREFIX_BYTES` 12288 → 2048–4096** (evidence-backed): the pre-payload header was at most
  1,758 bytes over 20,000 files and a 2,048-byte prefix truncated 0 of 1,200; 219 → 37 MiB over
  NFS; extend-and-retry must stay.
- **`truncnorm._ppf` in place of `truncnorm.ppf`** (evidence-backed): bit-identical on 2,000 random
  inputs, 0.15 s per 100 studies; private API, so pin scipy.
- **Cache the per-series header decoding** (evidence-backed, bit-exact, identity-changing). Side
  fields, character set and `header_side` are decoded per file with pydicom helpers; headers are
  about 30% of loader CPU and the cache is worth about 1 to 2% of it. Key it on the raw bytes so it
  stays exact.
- **Vectorise `summarize_series`** (speculative, bit-exact only if float order holds). 16.1 µs per
  file serial, 15% of serial `index_series`; worth 1 to 1.5 s. Audit poses and bounds for
  float-order differences.
- **One file access per selected frame** (speculative, unquantified). Indexing already records the
  payload offset, yet the payload read opens the file again and calls `fstat` to revalidate size
  and mtime. The `fstat` is 1.6 µs per frame locally (24 ms per 14,882 frames), so only the NFS
  round trip could justify the change, and the attribute cache may already absorb it. A payload
  read must still fail closed when the file changed. No probe exists.
- **Bounded ordered payload window in `decode_series`** (speculative, memory-motivated). Submit a
  window of reads, decode in record order and release consumed bytes: `pool.map` submits eagerly and
  the tuple retains every payload through the decode loop, so raw payloads and resampled outputs
  are live at once (about 83 MB of payloads per worker at ×4, about 1.3 GB across 16). Up to −80%
  transient RAM; `buffersize` is not available on the bundled Python, so the window must be
  explicit, and throughput must be measured, not only RSS.

## Hidden-test guards

Faults with zero training hits (speculative; details in [hidden-test risks](../inference/hidden-test-risks.md)):

- **A study with no usable series** raises in `KneeStudyDataset.__getitem__` and stops the run;
  predict training prevalences for it instead.
- **Contrast flag combinations other than (0,0) and (1,1)** are dropped with a warning; map them to
  the (0,0) slot, the host's training convention
  ([census](../metadata/series-filtering-census/census-and-categories.md#5-contrast-measured)).
- **Localizers and scouts as separate series**: drop series whose `ImageType` holds `LOCALIZER` or
  whose description names a localizer, scout or survey
  ([signals](../research/data-and-validation/series-filtering-signals/findings-by-case.md#localizers-scouts-and-calibration)).
- **Within-series consistency**: keep a series' most common TE, TR and `ImageType` and drop
  repeated positions; all 24,371 training series were already consistent
  ([audit](../metadata/repeated-series.md#detection)).
