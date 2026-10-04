# Learnable-views campaign contract

## Objective

Improve the learnable recipe (user, 2026-10-02): a ResNet-18 that learns, by gradient, where and
how wide each series type's view is, which view each label listens to, and how a study's slices
are shared between its series types, starting from the 120 mm centre crop
([build and runs](build-notes.md)).

The reference and starting incumbent is `views-budget-shift-centre`
(`outputs/training/views-budget-shift-centre`, the recipe in `configs/training.yaml`, seed 1533697213,
fold 0 of `all_5fold_scanner_v2`): ×4 0.932489, ×1 0.9273, 1,096 s of training, 0.947 public as
S94. It ties the hand recipe at ×4 (0.932880 on main, 0.932627 frozen on this build) and trails
it by 0.002 at ×1 and by 0.003 on the public set (S93 0.950)
([prior work](prior-work.md#where-the-learnable-recipe-stands)). The campaign looks for changes
to how the views, the routing and the budget are learned that make the learnable recipe better;
closing the ×1 gap is the first behaviour to target.

Fourteen forced seeds each change one thing of the recipe. The first six open the user's named
directions: two views on one series type, a learned knee centre, a 3D position measured from
it, and a learned spread of the slices along the stack. The next three change how the views
are learned: on held-out studies, on a longer gradient average, and starting late. After them the researcher's main work is to combine what acted
([evidence requirements](#evidence-requirements)), and it is free to try any idea on how these
quantities are learned, inside the three [limits](#limits). The research does not end by
itself (user, 2026-10-02): the researcher never returns an empty refill, opens a new family
when one closes, and only the user stops the campaign.

## Evaluation

The [shared evaluation and adoption rule](../../protocol.md#shared-evaluation-and-adoption)
applies unchanged against the campaign incumbent.

## Limits

1. **Time:** training time inside the [budget](#budget) range.
2. **Slices:** the configured `slices` of all views sum to 32, and a study keeps its slice
   total: 26, or the configured sum of its present series types when that is larger
   (`data.study_slice_total`). How types and views share it is open and may be learned.
3. **Image size:** no backbone input above 224 px on either side. The canvas the loader ships
   (384 px at 0.375 mm) is not a backbone input and stays as it is.

Standing project rules still apply and are listed under [Excluded](#excluded).

## Learnable recipe

What the incumbent does, and what every idea keeps unless it states the one thing it changes:

- **View table.** `data.series_types` lists the views of each series type. A view states
  `spacing_mm`, `slices` and, per axis, an anchor (`image` or `knee`) and an `offset_mm`. A type
  may list several views; each selects its own slices of the type and is cut by the model from
  the canvas. The recipe has one view per type at 0.5357 mm (120 mm) on the image centre, with
  6/6/5/5/5/5 slices.
- **What is learned.** `data.learn_views: [spacing_mm, offset_mm, slices]` makes those fields
  starting values. Spacing and offsets are model parameters trained at `optimizer.view_lr`
  (0.006) with no weight decay and outside gradient clipping, clamped after every step to
  `data.view_spacing_range_mm` (0.3125–0.625) and `data.view_shift_limit_mm` (20 mm). A field
  left out of the list stays at its configured value; an empty list is a frozen recipe.
- **Gradient.** A view value receives gradient only through the image samples of its windows.
  Token positions and validity are detached.
- **Jitter.** In training the window centre moves by a uniform draw of up to
  `data.affine_augmentation.shift_mm` (8 mm) per axis and the spacing is divided by a draw from
  `scale_range` (0.9–1.1); only the rotation stays in the resampling affine. Validation and
  prediction use none.
- **Clipping.** A window is shifted, never shrunk, to stay inside the stored image.
- **Pixels.** Windows are resampled from the canvas with bicubic sampling and a low-pass matched
  to the spacing ratio, and rescaled per series and view from the window's own quantiles.
- **Label bias.** `model.view_label_bias`: one learned attention offset per view and label,
  starting at zero.
- **Slice budget.** With `slices` learned, one logit per view; their softmax shares the study's
  slice total between its present views. In training the counts are rounded at random with a
  fixed total, `data.slice_budget_shift` (0.5) of share moves between two random present views
  first, and one marked slice per view carries the gradient, computed over the views present in
  the study. Validation and prediction round the learned shares by largest remainder.
- **Deterministic evaluation.** The views and counts of every validation study are a fixed
  function of the study and the checkpoint; the validation dataset receives no training seed.
- **Prediction reads the checkpoint.** Learned values are model parameters. A quantity the
  loader needs is returned by the model's `loader_state()`, handed to the datasets each epoch,
  saved in the checkpoint and passed to the loader by the final prediction and the Kaggle
  prediction worker.
- **Knee centre.** A knee-anchored axis is placed on the study's knee centre, the point closest
  to the centre lines of its series with every plane weighted equally; slice selection draws
  around the same point, and the meniscus mask, the OA detector regions and the half-knee loss
  measure medial, lateral and anterior from it, outside a 10 mm band around the midline. A
  study whose fit is rejected takes the existing fallback.
- **Fallbacks are per series:** a series without trusted geometry is placed on the image centre
  on any knee-anchored axis.

## Seeds

Forced, in this order (user, 2026-10-02): the user's two-view direction first, then the
single-change seeds, configuration-only before the one that needs code. It is not the order of
expected gain.

1. `two-views-sagittal-fluid`: sagittal fluid has two learned views, 120 mm and 93 mm, sharing
   its five slices 3 and 2.
2. `two-views-every-type`: every type has two learned views, 120 mm and the hand recipe's width
   for its contrast (84 mm T1/PD, 93 mm fluid).
3. `knee-anchored-learned-offsets`: every view anchored on the study's knee centre on both
   axes, offsets learned from zero. The control of seed 4.
4. `learned-centre-fit`: seed 3's views, and the knee centre fit with a learned weight per
   series type. One centre: it places the windows and the slices, both of which teach it, and
   the meniscus mask, the OA detector regions and the half-knee loss measure medial, lateral
   and anterior from it, with their hard cut-offs.
5. `knee-frame-3d-position`: seed 4's recipe with one side-normalized 3D position per token,
   measured from the learned centre and teaching it, in place of the 1D and 2D positions.
6. `learned-slice-spread`: the width of the Gaussian that places a view's slices along the
   stack, learned per view from 0.867, with today's centre.
7. `views-learned-on-held-out-studies`: a tenth of the training studies never train the
   weights and are the only ones that update the view values; the split is made inside each
   batch, so every study is still visited once per epoch.
8. `view-optimizer-long-average`: the view values stepped on a gradient averaged over about a
   hundred steps in place of ten.
9. `views-start-late`: view values kept at their start for three epochs, then learned.
10. `label-bias-off`: no per-view label bias.
11. `normal-jitter`: jitter of 4 mm and ±5% in place of 8 mm and ±10%.
12. `views-only-fixed-budget`: slices fixed at 6/6/5/5/5/5, no budget shift.
13. `start-from-hand-recipe`: learning starts from the hand views (84 and 93 mm, knee anchors).
14. `views-frozen-late`: view values stop moving after five of ten epochs.

Seeds 1 to 3 are configuration-only on the merged view table: a second view takes slices from
its type, so the cost is the reference's. Seed 4 is the learned centre as agreed with the user:
no free offset in the centre, so it cannot take over a view's offset and stays inside what the
study's series say, and no hard-coded bound. Seed 5 retakes the 3D position encoding, worth
about zero three times with the computed centre, on the learned one: it states seed 4's
complete recipe plus the position change, so seed 4 is its exact control. Seed 6 learns the
spread with the same per-slice weight and also tests that estimator. Seeds 7 to 9 answer three
symptoms of the learned runs: views that do not generalise (their frozen values score like the
centre crop), views that wander and end elsewhere in each run, and views that move most while
the model is untrained. Seeds 10 to 12 each remove or narrow one mechanism of the recipe, so
that its own effect is known. Seed 13 changes only where learning starts. Seed 14 is a guess on
the schedule. Seeds 4 to 9 and 14 need a source change. A seed that is accepted becomes
the incumbent the later ones are applied to; seed 4 states its complete view table, so it does
not depend on seed 3 being accepted, and is read against seed 3.

Six user seeds were added while the campaign ran (2026-10-02 and 03), forced right after seeds 4
and 5 and read with them: seed 5's recipe with the OA detector started first-order (zero readout,
positive gain), from an audit of why it stays dead beside the 3D position
([OA detector](oa-detector-ideas.yaml)); seed 5's recipe with a random displacement of the centre in training, as the
origin of the positions and axes ([centre noise](centre-noise-ideas.yaml)); seed 5's recipe with the centre weights stepped on a long gradient average,
from the audit of seed 5 ([centre long average](centre-long-average-ideas.yaml)); the 3D position from the computed centre on the incumbent, with nothing about
the centre learned ([computed centre](computed-centre-3d-ideas.yaml)); the 3D position added beside the 1D and 2D embeddings instead of replacing
them ([both encodings](position-both-ideas.yaml)), and seed 5's recipe with an EMA at decay
0.999 through the existing option, BatchNorm statistics averaged with the weights and no
refresh ([EMA seed](ema-ideas.yaml)). The EMA seed lifts the weight-averaging exclusion below
for that idea only; the researcher still proposes no weight averaging. It carries one source
change: under weight averaging the validation loader and the checkpoint take the averaged
model's loader state, since the averaged model is the one evaluated.

Rules for every idea, the seeds' and the researcher's:

- **One change against the incumbent at launch.** An idea that changes one field applies it to
  every view the incumbent has; an idea that states a view table replaces the incumbent's.
- **A learned quantity starts as the incumbent.** At its starting value it selects the same
  slices and places the same windows as the incumbent; the run logs that check at step 0.
- **The training loader is not changed.** The harness checks that every training study is
  visited once per epoch; an idea that treats studies differently does so inside the step.
- **Per-slice weights are not score-function estimators.** A quantity taught by the slice
  selection adds its log-density, minus its detached value, to the slice's pooling log-weight;
  the loss is never multiplied by a log-density.

## Budget

Wall-clock range: each candidate's training time must lie between 75% and 125% of the
reference's 1,096 s, that is 822–1,370 s (user, 2026-10-02). It is the only budget rule. Ten epochs, batch size 8
and the study's slice total stay; a candidate that would pass the upper bound removes duplicate
reads, decodes and resamples before anything else and is otherwise blocked, never trained for
fewer epochs.

## Fixed

Split, fold, evaluation labels (`rebuild_state_v2`), class-balanced BCE and its exponent 0.5, the
half-knee, report-attribute and L2-SP terms and their weights, the pool building, source
ranking, 60 mm radius and 0.5357 mm resolution floor of slice selection (the centre and spread
of the target depths are open), the six series types (their views are open), the study's
slice total, the canvas, epochs,
batch size, the photometric augmentations and their middle-slice path, and the evaluation and
prediction multipliers.

## Excluded

- **Frozen-view recipes as candidates** (user, 2026-10-02). A recipe whose view values are set by
  hand and not learned (`data.learn_views` empty) is not an idea of this campaign, and neither
  is a hand sweep of a width,
  a shift or a budget. A frozen run of learned values is a diagnostic the user runs outside the
  campaign. Starting values and ranges of learned quantities are open.
- Anything past the three [limits](#limits).
- Fewer epochs to buy time.
- A trained model inside the loader.
- Ensembling and test-time augmentation (endgame).
- Weight averaging (SWA, EMA; except the user's EMA seed above), reflection-probability changes, the class-balance exponent,
  label-source changes, gold-study weighting, multi-seed or confirmation runs.

## Evidence requirements

- Every candidate logs per epoch what the build logs for each view (spacing, width, both shifts,
  their gradients, the share of clipped windows, the attention each label gives it, the ×1
  validation AUC of its tokens alone) and the budget of every view, next to its own mechanism
  scalar.
- **Activation signatures prove the mechanism acts, not that a value is non-zero.** A run that
  learns its views shows at least one width more than 10 mm from its start by epoch 3; a run
  that fixes or removes a learned quantity shows it constant or absent.
- **×1 is reported beside ×4.** Every result gives the ×1 validation AUC of the selected epoch
  and of the last epoch next to the ×4 score, against the reference's 0.9273 and the hand
  recipe's 0.9293. The adoption rule reads ×4; the ×1 gap is the behaviour under study because
  the public score of S94 followed it (one submission).
- Every result is reported against the incumbent, against the reference and against the hand
  recipe on this build (`view-frozen-control`), with the paired 80% interval, and label by
  label.
- Every result records where the six views and the budgets ended. Learned view values differ
  between runs of the same recipe, so a difference in end values between a candidate and the
  reference is not by itself an effect of the candidate.
- One run per idea does not separate differences under about 0.002 macro AUC. Ideas are
  therefore also read on the logged diagnostics.
- The researcher starts each refill from the per-epoch curves and the view columns of every
  decided run and the reference, and names the behaviour each proposal targets.
- **Combinations** (user, 2026-10-02). A seed rejected by the rule is not a dead mechanism: one
  run does not separate 0.002. The researcher keeps one table of every decided run (what it
  changed, whether its activation held, its ×4 and ×1 deltas, the labels it moved, the
  diagnostics it changed) and, after the seeds, proposes combinations of mechanisms that acted
  on different labels or parts of the recipe. A combination states its complete recipe, the
  decided experiments it draws on and what it should show that its parts did not, and is
  reported against each of its parts as well as the incumbent.
