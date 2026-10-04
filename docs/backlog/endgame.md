# Endgame

Final-phase work, deliberately deferred by the [fold-0 iteration
strategy](../validation/validation-strategy.md); bigger images joined on 2026-09-16 and weight
averaging on 2026-09-17. These entries wait for the final phase; when one starts, move it to
[ongoing work](../ongoing/INDEX.md) and remove it from this file in the same change. Fold-0 results that
bear on them are recorded here as evidence. Each entry says whether its motivation is measured or a
hypothesis.

## Ensembling

**Round one is executed and is no longer backlog:** ten members, +0.0061 OOF over ReXNet alone,
**0.943 public**, recorded in the
[fold and ensemble results](../validation/results/2026-09-19-fold-ensembles-and-sigma-sweep.md),
the [submissions](../inference/submissions/2026-09-19-ensemble-submissions.md) and the
[project state](../ongoing/state.md).

**What remains here is a second round on the current recipe.** Every member of round one trains
with ordinary BCE, which class-balanced BCE beats by 0.0116, so the whole bundle is built on the
superseded loss; seed ensembles and diverse runs are still untried. Train the remaining folds under
the [validation strategy](../validation/validation-strategy.md), then combine models under the
[ensemble interpretation](../validation/ensemble-interpretation.md) rules.

- *Bundling mechanism (implemented 2026-09-19):* `export.py` and `predict.py` now support a
  multi-checkpoint bundle that averages members' per-label probabilities on one shared decode
  pass, so building the eventual fold/seed ensemble is a matter of exporting with the trained
  checkpoints, not further engineering
  ([bundle contract](../inference/model-bundle-and-offline-environment.md#multi-checkpoint-ensemble-bundles),
  [risk guard](risks.md#numerics-and-scores)). What remains here is which folds/seeds/backbones to
  train and combine and how to weight them, not the export/predict mechanism.

- *Full-data fits:* one model trained on all training studies, with its epoch count fixed from
  cross-validation, is the single-model alternative to a fold ensemble. It is also the natural
  fast candidate for the efficiency prize.
- *Fold models:* besides being ensemble members, they give out-of-fold predictions for choosing
  ensemble weights.
- *Local evidence:* averaging the champion with another fold-0 run gained +0.0028 to +0.0041 over
  the better single run, and a crop model added about as much as a second full-slice run
  ([two-crops averaging](../validation/results/2026-09-15-two-crops-result.md#per-target-comparison-with-the-champion),
  [per-label ensembles](../validation/results/2026-09-16-per-label-crops-result.md#ensembles-with-the-full-triplet-models)).
- *Logit averages:* champion + `00-baseline` scored 0.896032, and champion + `00-baseline` +
  per-label crops scored 0.897562.
- *Seed versus loss diversity:* on saved fold-0 predictions from the losses campaign, three control
  seeds averaged gained +0.0044 over one run. Loss-variant trios of equal member quality added
  about +0.0009 more, led by `shared-bce-ranking` and `shared-shift-projected-bce`. Subset search
  did not beat averaging every run once selection optimism was removed
  ([simulation](../validation/results/2026-09-21-loss-ensemble-simulation.md)).
- *Scale diversity:* on fold-0 ×4 predictions, pairs of a stride-32 run with a stride-16 or fused
  run gained about +0.001 more over the pair mean than two stride-32 runs (+0.0040 against
  +0.0030). `fusion-sum`, `fusion-scaleheads` and `stride16` are candidate members; their model code
  exists only in their run's `source_snapshot.zip`
  ([pair ensembles](../validation/results/2026-09-22-stride16-fusion-result.md#pair-ensembles),
  [history](../history/stride16-feature-fusion.md#implementation-notes)).
- *Three architectures and many folds (measured, 2026-10-02):* ReXNet-200 + Refined ResNet-50 +
  the ResNet-18 with the 84/93 mm views scores 0.936926 on fold 0 at ×4, +0.0009 over the S89
  pair (80% [+0.0003, +0.0016]), and 0.953 public as S96 against 0.951 for the pair. A second
  ResNet-18 with other views adds nothing on the public set (S95), and no other fold-0 model
  improves the trio on fold 0. Adding models of other folds and two more architectures, 28
  networks with mirror TTA and a fixed share per architecture, scored **0.954 public as S97**;
  which addition carried the step is not separated, and members trained on different label
  versions are averaged without calibration
  ([blends](../inference/submissions/2026-10-02-blend-submissions.md)).
- *Public-stack blend (external evidence, 2026-09-28):* the top public notebooks reach 0.945 by
  rank-blending the free 0.943 public stack with one independent private model at weight 0.45–0.6,
  and a participant's 0.942 CoAtNet did the same. Each blended model scored at or below the stack
  alone; ours (0.948) scores above it. Candidate: our final bundle rank-blended with the stack, at
  one global weight fixed in advance. The stack needs roughly 1.5 h or more on 2×T4, and prize
  eligibility of submissions using other participants' public weights was unanswered on 2026-09-28
  ([re-sweep](../research/2026-09-28-public-resweep.md#public-0945-the-public-stack-rank-blended-with-a-private-model)).
- *Per-target member weights (external evidence, 2026-09-28):* one public pipeline measured
  per-target top-3 member selection at +0.0034 over a flat mean on held-out halves of 84
  out-of-fold files, mostly on MCL, Synovitis and ACL. That disagrees with our loss-ensemble
  simulation above; per-label choices are allowed again, fixed before the run
  ([re-sweep](../research/2026-09-28-public-resweep.md#measurements-embedded-in-the-s75-private-model)).

## Bigger images

**Finer resolution at the same field of view** (hypothesis; resolution at a fixed field of view
has never been tested, see [input resolution](../research/input-resolution.md)). Train the
full-slice model at 0.5 mm views and `data.image_size: 280`, which keeps the 140 mm field (`data.reference_field_mm: 140`)
and the sampler's 70 mm in-plane radius.

*Update, 2026-10-01:* the ResNet recipes now read 120 mm at 0.536 mm and 224 px, which took the
resolution gain by narrowing the field (+0.004 public, S90 against S86). A bigger image would
now buy the field back at that pixel size: 140 mm is 261 px and 160 mm is 299 px at 0.536 mm.
At 224 px a 160 mm field added nothing in five designs, but there it cost resolution
([result](../validation/results/2026-10-01-mixed-field-campaign.md)).

- *Motivation:* per-label crops at 0.5 mm gained over `00-baseline` on ACL (+0.0111), synovitis
  (+0.0077) and medial OA (+0.0057)
  ([result](../validation/results/2026-09-16-per-label-crops-result.md#per-target-auc)). Learned
  zoom, whose ReXNet-200 evidence was in effect at 0.875 mm, lost heavily on the lateral meniscus
  (−0.1057), MCL (−0.0671) and ACL (−0.0223)
  ([result](../validation/results/2026-09-16-learned-zoom-result.md#per-target-auc)). Across the
  crop runs, score followed slice coverage and a spacing of 0.625 mm or finer
  ([investigation](../history/crop-selection-investigation.md#what-the-runs-established)). The
  architectures differed, so none of this is a controlled resolution test.
- *Acquisition evidence:* 0.5 mm keeps the full acquired band in 59% of series against 23% at
  0.625 mm, and finer than about 0.5 mm adds almost nothing, because stored images are
  interpolated about 2×.
- *Public claims (unreplicated):* one discussion thread reports 0.936 for a small single-fold
  ResNet at 224 px and 0.942 at 288 px
  ([public discussion](../research/public-research.md#validation-and-ensembling)).
- *Cost:* 1.56× the pixels (280² / 224²). ReXNet-200 measured 2.462 against 1.526 GMACs per
  triplet (1.61×), and its token grid grows from 7×7 to 9×9, which gives 81 tokens instead of 49
  ([budget](../modeling/backbone-swaps.md#current-budget)).
- *Memory:* scaling `00-baseline`'s 20.02 GB peak gives an estimated 31 GB at batch 8, so the
  24 GB GPU needs gradient accumulation (for example batch 4 × 2) or activation checkpointing. The
  model allows `model.checkpoint_backbone` only with `model.backbone_eval: true`, which also
  changes BatchNorm behaviour.
- *`model.backbone_eval` alone:* moved here from the [overnight queue](overnight.md) on
  2026-09-16 (the user's decision) because it only matters as the gate for
  `model.checkpoint_backbone` above; an isolated on/off comparison at the current image size is
  not worth a queue slot. `baseline0` (`resnet50d.ra4_e3600_r224_in1k`) and `baseline1`
  (`resnet50d.ra2_in1k`) both trained with `backbone_eval: true` and `checkpoint_backbone: true`,
  but neither has a recorded comparison against `backbone_eval: false` on the same recipe.
- *Host memory:* loader memory grows with pixels
  ([host memory](../modeling/training-runtime-and-memory.md#host-memory-with-larger-inputs)).
- *Inference:* the backbone forward is about a quarter of Kaggle prediction time at 16 loader
  workers ([profiling](../inference/2026-09-15-kaggle-inference-profiling.md)).
- *Dominated alternative:* a dilated stride-16 ReXNet (14×14 tokens, no new pixels) measured 3.0×
  the backbone time and a full 24 GiB
  ([measurement](../research/multiscale-features-knee-mri.md#a-dilated-stride-16-backbone-costs-three-times-the-backbone)).

## Test-time augmentation

**Mirror TTA evaluated 2026-09-19: +0.001 public for +181 s** on five b0 members at ×4 (0.941
against 0.940), and +0.00046 on fold 0 locally. Both sit inside the ±0.006 public noise, so it is a
marginal win on score and a clear loss on efficiency, where 181 s costs 0.0056 of the runtime term —
on for the score bundle, off for the efficiency bundle
([submissions](../inference/submissions/2026-09-19-ensemble-submissions.md)). Other transformations
remain unevaluated, and the mirror gain should be remeasured once members train with class-balanced
BCE. Evaluate any further transformation with consistent geometry and target handling, recording
total inference cost.

The implemented [reflection TTA mechanism](../inference/model-bundle-and-offline-environment.md#reflection-test-time-augmentation-tta2)
is already used by the submitted bundles; the pending work is its value under the next recipe.

**More evaluation triplets are adopted, not pending.** ×4 at τ 0.75 scores 0.939 public against
0.938 at ×3 and 0.933 at ×2; the per-count fold-0 and public table, and the reason sharper pooling
is what makes larger counts pay, are in the
[submissions record](../inference/submissions/2026-09-18-temperature-and-count-submissions.md).
Counts above ×4 were dropped by user decision on 2026-09-24; the one submission above ×4,
`oblique` at ×5 τ 0.70 (S47), scored 0.945 against 0.944 at ×4 τ 0.75 (S46)
([history](../history/oblique-groups.md)). External evidence on density is in
[public research](../research/public-research.md#slice-density-and-the-board-2026-09-18-re-sweep).

**More training triplets per study are deferred here for iteration speed** (user decision,
2026-09-29). In the `radimagenet-rn18` campaign, `data.study_slice_total` 26 → 36 on the
RadImageNet ResNet-18 without series dropout gained +0.0018 macro AUC at ×4
(0.927806 → 0.929586; 80% [+0.0002, +0.0034], 95% [−0.0005, +0.0041]; one pair, weak). The gain
moved between labels: OA rose (lateral +0.0094, medial +0.0026, PF +0.0036) and both menisci fell
(medial −0.0033, lateral −0.0049). Training time rose 23% (1,202 → 1,477 s), leaving later
campaigns about 10% headroom under their 1,627 s ceiling. A 40-triplet, 9-epoch follow-up scored
−0.0015 against 36, confounded by the lost epoch. The 36-triplet model scored 0.944 public (S79);
no 26-triplet version was submitted, and a 0.002 gap is below one submission's resolution. The
committed recipe (`configs/training.yaml`) therefore keeps 26. Denser per-type budgets (`slices` above 6/6/5/5/5/5) belong to the same
deferral; on ReXNet the cheaper side lost monotonically (−0.0041, −0.0051). Revisit training at 36
with the final recipe.

## Weight averaging

**EMA or SWA as the final recipe's default** (measured on fold 0; moved here on 2026-09-17, the
user's decision). Both are implemented behind `optimizer.weight_averaging` (`null`, feature off, in
`configs/training.yaml`) and were measured by `03-ema-0.999` and `04-swa-3` of the 2026-09-17
queue ([result](../validation/results/2026-09-17-overnight-results.md#weight-averaging)): averaged
weights beat the live ones by about +0.00015 at the best epoch and +0.0005 at the last, and lower
BCE; EMA costs +153 s per run and SWA +369 s. The EMA checkpoint scored 0.929 public, +0.001 over
its own live weights (0.928). The gain is small against seed noise, so fold-0 experiments train
without it until the final phase. A public CoAtNet pipeline reports the same null for SWA over six
pairs: 0.9167 against 0.9150 on the 58 gold studies and 0.938 against 0.937 on the board
([public research](../research/public-research.md#slice-density-and-the-board-2026-09-18-re-sweep)).

- *Before adopting it:* validate the live weights only at the last epoch. The second validation
  pass is all of EMA's +153 s (about 17 s per epoch; training steps are unchanged), so this cuts
  the overhead to about 17 s.
- *Follow-ups:* EMA decays 0.998 and 0.9995, EMA with `minimum_lr_ratio: 0.1`, and SWA paired with
  a non-vanishing tail learning rate (`minimum_lr_ratio: 0.1`), which `04-swa-3` does not test.
Implementation, original design choices and measured costs live in
[the modeling guide](../modeling/schedule-and-augmentation.md#weight-averaging).

## Training-label exclusions

**Measured composition; untested as a change. Set aside on 2026-09-22 by user decision.** Three
groups of training studies whose report-derived labels likely do not describe their images.
Excluding them changes training only, not inference. They total about 19 studies (0.4%), so any
fold-0 effect is expected to be below noise. Evidence:
`outputs/analysis/label-hygiene-20260922/` (scripts, study lists and matched phrases).

- **Cross-study copy (proven):** `…99725523560254732094432244556623376711` (fold 2). Its images
  are a pixel copy of the right-knee study `…613209`, but its report describes the left knee
  ([findings](../metadata/repeated-series.md#cross-study-repeat)).
- **Empty or cut-off reports (6 studies; 2 in fold 0, none gold):** four Dutch reports end at or
  before `Bevindingen:`, one reads `Bevindingen: aa`, and one Spanish report is
  `Resultados: . Impresión:`. Their `combined_v1` labels stay between 0.25 and 0.35.
- **Report and image on opposite knees:** the report states one side and the header the other
  in 26 of the 1,715 studies where both are known.
  - **13 at the Bulgarian site** (11.4% of its 114). Its template reads "the bones forming the
    right knee joint", so these are probably unedited template text; keep them.
  - **13 elsewhere** (0.4–2.2% by language), including the cross-study copy. They are typos or
    swapped reports, which cannot be told apart without reading the images; review before
    excluding.

The reference is the production knee side: explicit tags, or coordinates when tags are absent.
Where both exist, they agree 99.4% of the time
([coordinate investigation](../metadata/knee-side-coordinate-investigation.md)).

## Defensive guards for faults absent from training

**Hypothesis; zero training hits, kept in case the hidden test differs.** Two header-only
guards for faults the training data does not contain, so a fold-0 run cannot measure them. Decide
with the final bundle.

- **Localizers and scouts:** drop series whose `ImageType` holds `LOCALIZER`, or whose description
  names a localizer, scout or survey. None exist in training
  ([signals](../research/series-filtering-signals.md#localizers-scouts-and-calibration),
  [risk](risks.md#silent-input-changes)).
- **Within-series consistency:** keep the most common TE, TR and `ImageType` of a series, and drop
  repeated-position instances, so that a dual-echo series cannot mix contrasts inside one triplet.
  The 2026-09-22 check of all 24,371 training series found no series with more than one TE, TR or
  `ImageType`, no repeated slice position and no spacing gap
  ([audit data](../metadata/repeated-series.md#detection)).

## Out-of-fold self-distillation

**Train a second five-fold round on targets blended with the first round's out-of-fold
predictions** (filed 2026-09-21; first test 2026-09-23, a public tie). Every non-gold training cell
becomes `0.7 · label + 0.3 · teacher`, where the teacher is that study's out-of-fold prediction. Gold
labels and the evaluation targets stay unblended. The aim is robustness to report-label noise; the
same option also serves speed (*Speed* below).

- *Two-family round ([result](../validation/results/2026-10-04-two-family-oof-distillation.md),
  2026-10-04):* five Refined ResNet-50 and five ReXNet-200 children at 0.5 silver, 0.25 and 0.25
  of the two families' out-of-fold predictions. Every child beats its teacher (pooled +0.0034 and
  +0.0090, optimistic), and the ten together score 0.955 public (S103), the best. Open: a second
  round with the children as teachers (their blend agrees with the 58 gold studies at 0.9375
  against 0.9288 for the teachers'; one ResNet-50 on fold 0 first, against today's child), and
  the teacher share of the labels (0.5 was a convention; the ResNet-50 teacher alone is 0.9354 on
  gold against 0.9358 for the blended table and 0.9075 for the silver labels).
- *First test ([result](../validation/results/2026-09-23-image-blend-labels-result.md)):* one
  fold-0 child at 0.5/0.5, with the on-hand teacher described below. It gave:
  - gold agreement of the label table +0.020 [+0.006, +0.036];
  - fold-0 hard AUC +0.0035 and soft AUC +0.0065, both leaky;
  - S50 0.945 against 0.942 for the same-source control S48, a tie by the rule fixed in advance.

  The gains sat on the eight Steven v4 targets. PF OA lost on every measure, because the teacher
  learned Steven v4's looser definitions and ordinary BCE.
- *When to resume:* once the final phase's fold training on the current recipe exists (its
  labels, loss and split). Its out-of-fold predictions are the matched teacher. On 2026-09-24 the
  user dropped combining the then-adopted model with the existing image-blend labels, so the
  entry waits for that matched teacher. That removes the label-definition and loss mismatches and gives a clean fold-0 gate.
  Build the table with
  [`prepare_image_blend_labels`](../../scripts/tools/prepare_image_blend_labels.py), whose teacher
  runs, split and blend weight are constants to point at the new folds. Fix the weight before the
  first run: 0.7/0.3 per the precedent below, or the tested 0.5/0.5. Do not choose it on gold.
- *Per-label teacher weights (user, 2026-09-30):* when several teachers are blended, weight each
  label by that teacher's measured per-label strength rather than one global weight. Models trade
  labels: plane-contrast readouts gained Lateral OA +0.0054 (95% [+0.0009, +0.0104]) and lost
  Fracture −0.0024 against the control in the `revisits` campaign. Every campaign comparison saves
  the per-label ×4 AUCs in `comparison/per_label.json`; the campaign monitor
  (`scripts/tools/serve_campaign_monitor.py`) shows them per run. Fix the per-label weights from
  out-of-fold AUCs before the child run, never from fold-0 or gold results.

- *External evidence:* the Plant Pathology 2020 first-place solution used this recipe: five-fold
  out-of-fold predictions and 0.7 truth + 0.3 prediction
  ([write-up](https://hamonk.github.io/2020/12/05/plant_pathology.html); no isolated gain given).
  Same-architecture children beat their parents on CIFAR
  ([born-again networks](https://arxiv.org/abs/1805.04770)). For linear and logistic models under
  label noise, theory gives when the child beats the parent, and under heavy noise the optimal
  teacher weight exceeds the label's ([Das and Sanghavi, ICML 2023](https://arxiv.org/abs/2301.13304)).
  A participant 26th on the board reported, in a discussion reply of about 2026-09-18 with no
  numbers, that mixing predictions into the extracted labels beat both replacing and masking them.
  No measured gain was found for multi-label AUC on report-derived labels; the closest setting,
  [chest X-rays with NLP labels](https://link.springer.com/chapter/10.1007/978-3-030-87358-5_57),
  is paywalled.
- *Forum reports (2026-09-28, mixed):* one participant says only truly out-of-fold teachers help
  and that more than 50% teacher weight helped more. Another calls multi-source teachers at 50/50
  their main lever. Two others saw gold and local gains with no public gain, as S50 did
  ([re-sweep](../research/2026-09-28-public-resweep.md#pseudo-labels-and-out-of-fold-label-blending)).
- *Why an independent teacher (2026-09-23):* teachers from the same run add nothing here. The
  model does not memorize its labels in ten epochs, so bootstrapped, running-average and EMA targets
  tied or lost, while two independent seeds averaged gained +0.0028
  ([noisy-labels campaign](../validation/results/2026-09-23-noisy-labels-campaign.md#why-the-family-barely-acts)).
- *Settings:* 0.7/0.3 is the only weight with a winning precedent; one weight for all labels.
  Children keep their
  augmentation, since removing student noise removed the student's gain in
  [Noisy Student](https://arxiv.org/abs/1911.04252). One round only: a few rounds reduce
  overfitting and more underfit ([Mobahi et al., NeurIPS 2020](https://arxiv.org/abs/2002.05715)).
- *Teacher on hand:* round one's out-of-fold predictions cover all 4,407 studies, one ReXNet + B0
  pair per study ([results](../validation/results/2026-09-19-fold-ensembles-and-sigma-sweep.md)).
  They come from split `all_5fold_v3`, ordinary BCE and the labels before `combined_v1`, and carry
  the folds' calibration drift into the targets.
- *Local-score leak:* child k's target for a study in fold j comes from parent j, which trained on
  fold k, so child out-of-fold scores are biased upward against the parents'. The hidden test is
  unaffected. A clean fold-0 gate: four teachers cross-validated inside folds 1–4, then one child on
  folds 1–4, against the fold-0 incumbent. Otherwise the public board decides.
- *Comparison:* child ensemble against parent ensemble, and parents plus children. Models trained on
  improved labels were reported to be more correlated and to add less to ensembles
  ([supervision](../modeling/supervision.md#downstream-benefit-changed-over-time)).
- *Cost:* a training option that blends a predictions table into the targets, then a second
  five-fold round; the fold-0 gate is five runs.
- *Speed (hypothesis; no speedup or score measured):* the same blending option distills the
  ensemble into one fast model, so that inference costs one model. Point the teacher at the full
  ensemble's predictions, raise the teacher weight, and train one full-data student, possibly on a
  smaller backbone. The idea recurs in the
  [unverified notes](../modeling/unverified-modeling-notes.md#loss-and-ensemble-diversity) and the
  [runtime leads](../inference/runtime-evidence-and-leads.md#runtime-leads).
