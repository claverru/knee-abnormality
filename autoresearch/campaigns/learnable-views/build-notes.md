# Learnable-views build and review notes

Working notes of the learnable view setup, written from 2026-10-02 to 2026-10-03 while it was
built and while its [campaign](INDEX.md) ran; moved here from `docs/ongoing/view-optimisation.md`
when the line closed on 2026-10-04. They describe the code of the `view-optimisation` branch,
which is not on main: main keeps the hand-set view table and the label bias
([view table](../../../docs/modeling/current-model.md#view-table)). Results and conclusions:
[2026-10-03 learnable-views campaign](../../../docs/validation/results/2026-10-03-learnable-views-campaign.md).
Statements such as "uncommitted" or "no run yet" are as written at the time. It followed the
[smart-crops campaign](../../../docs/validation/results/2026-10-02-smart-crops-campaign.md);
detailed evidence behind the smart-crops statements here:
[view effects by target](../smart-crops/view-effects-by-target.md).

## Goal

One recipe that keeps every per-label gain found in smart-crops, searched as an optimisation
problem and not as one hand-built idea at a time. Starting point: the recipe in force (coronal
and sagittal T1/PD at 84 mm, fluid at 93 mm, axial at 120 mm; ×4 0.932880, 0.950 public as S93).

## What we carry over

- Replacing a view by a finer one gains; adding views whose tokens enter every label's pool does
  not (fifteen ties).
- Effects add up across series types (fluid two-by-two: +0.0003, +0.0005, +0.0010 together), so
  types can be tuned in parallel.
- The conflicts are between labels inside one type: sagittal fluid must be fine for five labels
  and wide for Baker's cyst, and no single width between 93 and 120 mm serves both.
- Placement by rule is enough; the sagittal forward shift earns its place; axial width between
  120 and 140 mm does not matter.
- Supply of slices never limits the budget: every type has four to five times what it is given.
- Inside the tested widths the surface is nearly flat around the recipe in force: the regression
  predicts every single-type change 0.0001 to 0.0005 lower. One run resolves about 0.001 on the
  macro.
- Much of the space is untried: axial finer than 120 mm, fluid finer than 93 mm, T1/PD finer
  than 84 mm, any budget other than 6/5/5 slices per type.

## Decision: learnable first, configuration as the fallback

User, 2026-10-02: try the full learnable setup, except sampling; configuration experiments
remain available on the same code.

| Thing | How it is learned |
|---|---|
| Spacing of each view | one scale per view of each series type, shared by all studies, bounded (about 0.31 to 0.63 mm) |
| Shift of each view | two offsets per view in anatomical millimetres from its anchor, bounded (about ±20 mm) |
| Which labels read which view | a bias per label and view on the attention score, starting at zero |

Stays configuration: how many views a series type has, the image size (up to 224 px), the
slices per type, and whether a second view reads the same slices as the first or its own. Every
learned value can also be frozen from the configuration, which is what makes the fallback free.

The parameters are global per series type, not predicted per image: per-image placement is what
failed in smart-crops and in the literature.

## What is built

Code: `src/knee_abnormality/modeling/views.py` (the view parameters, the window placement and
the resampling), `modeling/model.py` (`encode_batch`), `data/dataset.py` and
`data/preprocessing/spatial.py` (`window_frame`), `training/lightning.py`.

- **Canvas.** The loader ships every slice once, centred on the stored image, at
  `data.canvas_spacing_mm` and `data.canvas_size` (0.375 mm and 384 px, 144 mm; 0.3125 mm
  and 448 px, 140 mm, until the two first runs). 0.375 mm is the finest pixel of the recipe
  in force and keeps the acquired detail of about nine in ten coronal and sagittal series;
  a finer view is interpolated from it. The extent holds the widest window; at a 120 mm
  start it leaves 12 mm of room to shift on each side.
- **View table.** Every series type lists its views (`data.series_types[].views`): spacing,
  and per axis an anchor (`image` or `knee`) and an offset in millimetres, plus
  `learn_spacing` and `learn_shift`. `data.views.policy`, `data.spacing_mm` and the loader's
  window placement are gone. A type may list several views; they read the same slices.
- **Directions.** Horizontal offsets point medially on axial and coronal series and
  anteriorly on sagittal ones; vertical offsets point posteriorly on axial series and
  inferiorly on coronal and sagittal ones. An offset from the knee centre is measured in
  patient millimetres, one from the image centre along the image axis.
- **Fallback.** A series without trusted geometry is placed on the image centre; only a
  vertical offset from the image centre still applies, along the rows.
- **Clipping.** The window is moved back inside the stored image and the canvas; a window
  larger than the image is centred. It is never shrunk.
- **Intensity.** Every window is restretched between the 0.5% and 99.5% quantiles of its
  own pixels inside the image, per series and view, as main does over its window; the
  loader clips the canvas at a fifth of those tails (0.1% and 99.9%). Added after the audit
  of the first budget run; the three learnable-views runs before it used canvas-wide
  limits.
- **Learned values.** One spacing per view, stored as a fraction of the log range
  `data.view_spacing_range_mm` (0.3125 to 0.625 mm), and two offsets, stored as a fraction
  of `data.view_shift_limit_mm` (20 mm). They are clamped to their bounds after every
  optimizer step, train at `optimizer.view_lr` without weight decay, and stay out of the
  gradient clipping norm. A frozen view keeps its configured value exactly.
- **One resampling step.** The window, the study reflection and the augmentation's rotation
  are composed into one map from input pixels to canvas pixels; the augmentation's shift and
  scale act on the window itself. Each input pixel is the mean
  of two bicubic reads per axis, placed so that the total blur equals main's single
  triangle filter: their distance grows from zero, when the input pixel is no coarser than
  the stored or the canvas pixel, with the input pixel's size (the loader ships each
  series' stored pixel size for this). The result is clamped to [0, 1]. The other
  augmentations run on it as before. Contrast mixing now happens on the canvas, before the
  window is cut.
- **Gradient.** Only the image samples carry gradient to the view values. Token positions,
  validity, the meniscus own-side mask and the OA regions are recomputed every step from
  the current window but detached, and so is the scale embedding's input. Through the
  photometric augmentations the gradient is passed straight through.
- **Tokens.** The 2D position is the token's place in the canvas frame, in units of 60 mm
  from the image centre, mirrored with the image under reflection. A small network of the
  log spacing gives every view an additive embedding, zero at the start.
- **Label routing.** `model.view_label_bias` adds a bias per view and label to the
  attention score, zero at the start.
- **Slice selection.** `data.selection_radius_mm` (60) and
  `data.selection_resolution_floor_mm` (0.5357) replace what followed `data.spacing_mm`.
- **Prediction.** The learned values are part of the checkpoint, so Kaggle prediction reads
  them from it. Mirror TTA passes the study mask to the model, which mirrors inside the
  resampling; `inference/reflection.py` is gone.
- **Logged every epoch.** Per view: spacing, width and both offsets, their gradients, the
  share of clipped windows, the attention each label gives it, and the validation AUC of its
  tokens alone (`val_macro_auc_view_<view>`, `val_auc_view_<view>_<label>`).

Not built: a second view with its own slices, and a view restricted to chosen labels by
configuration.

### Checks so far (CPU, 2026-10-02)

- **Placement.** With the recipe in force frozen, the window centres of 245 coronal and
  sagittal series (60 training studies) match main's rule to within 0.28 mm (median
  0.13 mm, the old rounding to whole pixels) in 243; the other two were held inside the
  120 mm frame by main and are now free to follow the knee centre by 6 and 10 mm.
- **Gradient.** On the resampling path alone the analytic gradient of every spacing and
  offset agrees with finite differences. Through the whole model it does not and is not
  meant to, because the position and validity paths are detached.
- **Loader cost.** Sixteen workers alone on the CPU, seconds per batch of eight: 0.121 on
  main (47 MB), 0.226 with the 448 px canvas (188 MB), 0.179 with the 384 px canvas
  (138 MB). In training the 448 px canvas ran at 0.35 s per step with the GPU 44% busy and
  took 1.9 times the incumbent's time, and the 384 px canvas ran no faster; see
  [step time](#step-time-causes-and-fixes-2026-10-02) for the causes.
- **Detail against main (fixed 2026-10-02).** For the same slices and windows (12 fold-0
  validation studies, 64 series; scratch script, not kept), spectral energy above half the
  Nyquist frequency, new over main's single resampling from the stored pixels:

  | View | First build (0.3125 mm canvas, bilinear reads averaged 2x2) | Now |
  |---|---:|---:|
  | Coronal T1/PD, 0.375 mm | 0.64 | 0.95 |
  | Sagittal T1/PD, 0.375 mm | 0.65 | 0.95 |
  | Coronal fluid, 0.417 mm | 0.72 | 0.98 |
  | Sagittal fluid, 0.417 mm | 0.75 | 0.98 |
  | Axial fluid, 0.536 mm | 0.95 | 1.01 |

  The first build blurred twice whenever the input pixel was close to the canvas pixel, so
  a fine view was worth less than on main. The two first runs carry that defect.
- **Preview.** `scripts/tools/preview_augmentations.py` renders the windows of the first
  view of each type from the canvas.

The new resampling path cannot reproduce S93 bit for bit.

### Parity with main, frozen recipe (CPU, 2026-10-02, after the fixes)

Same studies through main's code and this build with `configs/training.yaml` (scratch
scripts, not kept).

- **Evaluation mode, 8 validation and 8 training studies:** the selected slices are
  identical in all 16; token positions differ by 0.003 of the ±1 range on average (0.012
  at most, about 0.7 mm); medial and anterior axes by 0.35 mm at most; token validity and
  trusted-geometry flags are identical. Image detail and intensity are in the checks above.
- **Training mode, 12 studies, four draws each; differences that remain:**
  - Valid tokens: about 90% on main (89 to 94% by type), 100% here. Main drops a token
    whose anchor leaves the 224 px window under the affine augmentation; here the window
    is resampled from the canvas, so that token has real content and stays.
  - Zero pixels: 6 to 15% of the input on main, 2 to 12% here (about four points fewer on
    every type), for the same reason.
  - Coronal token positions under reflection: main gives a mirrored coronal 84 or 93 mm
    view its unmirrored positions (spread across images 0.09 and 0.18), while its axial
    views get the displayed grid; here every view gets the position of what is displayed
    (spread 0.02 to 0.05).
  - Axial token positions: on main the fixed grid whatever the augmentation; here they
    follow the affine jitter (spread 0.01 to 0.05).
  The frozen control ties main, so none of these moves the score measurably.

## First runs

The reference is the recipe in force as trained on main, `fluid-detail-93-replacement`
(×4 0.932880, S93 0.950 public; user, 2026-10-02).

**Frozen control on this build** (`outputs/training/view-frozen-control`, 2026-10-02,
`configs/training.yaml` with nothing learned, fold 0, seed 1533697213, after the detail,
intensity, hand-off and logging fixes): ×4 0.932627, −0.0003 against the recipe in force on
main, where one run carries about 0.0008 of training noise; ×1 0.929247 at epoch 9, and
within 0.0002 of the incumbent at each of the last five epochs. Per label against main:
Baker's cyst +0.0025, ACL +0.0023, lateral OA +0.0023; MCL −0.0041, lateral meniscus
−0.0022, fracture −0.0022. Training 1,419 s, 1.25 times main's (133 s per epoch), final
prediction 76 s. So the build reproduces the recipe in force, and this run is the control
for everything learned on it: against it the budget-only run is −0.0016, the combined run
−0.0010, and the third learnable-views run −0.0007 (that one before the intensity fix).
The training-mode differences from main listed under parity (all tokens valid, displayed
positions, no black borders) therefore cost nothing measurable.

1. **Learnable from a centre crop** (`configs/learnable-views.yaml`, run
   `view-learnable-centre`, launched 2026-10-02; start agreed with the user): every type
   starts at 120 mm on the image centre with no offset, and learns its spacing, both offsets
   and the label bias. If the views move towards what smart-crops found by hand, learning
   works. If they stay, the search goes back to configuration experiments on the same code.
   At `optimizer.view_lr` 0.0006 (the head rate) the views barely moved in the first three
   of ten epochs: every offset stayed under 1 mm with no trend towards 10 mm down; coronal
   and sagittal T1/PD narrowed to 116.9 and 117.1 mm and coronal fluid widened to 124.0 mm,
   about 1 mm per epoch. Validation tracked the 120 mm control (0.9150 against 0.9168 at
   epoch 3, ×1). A training step took 0.35 s with the GPU 44% busy, so the loader limits it;
   the incumbent's whole epoch took about 113 s against roughly 155 s of training steps here.
2. **The same at `optimizer.view_lr` 0.006** (user, 2026-10-02; the tenfold is a guess; run
   `outputs/training/view-learnable-centre-lr6e-3`, fold 0, seed 1533697213, finished):
   ×4 0.931622, −0.0013 against the recipe in force and +0.0010 against the 120 mm control
   (`rn18-field-120-control`, 0.930666); ×1 0.927186 at epoch 9. One run, and a single run
   resolves about 0.001, so it sits between the two without separating from either.
   Training took 2,158 s, about 1.9 times the incumbent's, with the loader as the limit.

   | View | Width, mm | Horizontal, mm | Vertical, mm | By hand |
   |---|---:|---:|---:|---|
   | Axial T1/PD | 117.5 | +2.6 | +5.2 | 120, centred |
   | Axial fluid | 121.8 | +3.8 | −0.3 | 120, centred |
   | Coronal T1/PD | 92.4 | −0.4 | −6.4 (up) | 84, knee centre, 10 down |
   | Coronal fluid | 109.0 | +6.3 | −1.0 | 93, knee centre, 10 down |
   | Sagittal T1/PD | 139.2 | −2.2 | +8.1 (down) | 84, 7 forward, 10 down |
   | Sagittal fluid | 106.3 | +3.5 (forward) | +0.2 | 93, 7 forward, 10 down |

   - Agrees with the hand search: coronal T1/PD narrowed and settled near 92 mm from epoch
     3; both fluid views narrowed part of the way; sagittal fluid moved 3.5 mm forward;
     sagittal T1/PD moved 8 mm down.
   - Disagrees: sagittal T1/PD, the zoom worth most by hand, widened from epoch 4 until it
     hit the 140 mm bound, where the window fills the canvas, is always clipped and its
     offsets stop receiving gradient (frozen at −2.2 and +8.1 mm). Coronal T1/PD moved 6 mm
     up where the rule says 10 mm down (it reached 11.6 mm up at epoch 4).
   - Most values swung by several millimetres in epochs 1 to 5 and settled as the rate
     decayed, so the final values of the views that zigzagged (coronal fluid, the axial
     ones) are where the walk stopped, not a measured optimum.
   - Label bias stayed small: largest +0.10 (coronal fluid for MCL, axial fluid for Baker's
     cyst).
   - Per label, ×4 against the recipe in force: MCL +0.0052, Baker's cyst +0.0020, PF OA
     +0.0015, lateral OA +0.0011, medial meniscus +0.0008; fracture −0.0122, synovitis
     −0.0048, contusion −0.0035, effusion −0.0023, lateral meniscus −0.0015, ACL −0.0008,
     medial OA −0.0007. Against the 120 mm control: synovitis +0.0042, lateral OA +0.0030,
     MCL +0.0029, ACL +0.0027, medial meniscus +0.0026, PF OA +0.0026, medial OA +0.0016;
     fracture −0.0025, contusion −0.0025, Baker's cyst −0.0019.
   - Tokens of one view alone at ×1, last epoch: sagittal fluid 0.865, coronal fluid 0.858,
     axial fluid 0.828, sagittal T1/PD 0.779, coronal T1/PD 0.734.
   - Lesson: the gradient through the resampling finds some of the hand-found directions
     and misses the most valuable one. The run is also confounded by the
     [detail lost by the second resampling](#checks-so-far-cpu-2026-10-02), which makes a
     fine view worth less here than on main and may be why sagittal T1/PD went wide; that
     is not shown.
3. **The same after the detail fix, on the 384 px canvas** (run
   `outputs/training/view-learnable-centre-sharp`, same fold, seed and `view_lr` 0.006,
   launched by the agent at the user's request, finished 2026-10-02): ×4 0.931976, −0.0009
   against the recipe in force, +0.0013 against the 120 mm control and +0.0004 against run
   2; ×1 0.926334 at epoch 9. One run each, so none of these gaps is resolved.

   | View | Width, mm | Horizontal, mm | Vertical, mm | By hand |
   |---|---:|---:|---:|---|
   | Axial T1/PD | 129.1 | +3.3 | −6.3 | 120, centred |
   | Axial fluid | 119.3 | +2.6 | −0.5 | 120, centred |
   | Coronal T1/PD | 117.4 | −2.2 | +0.4 | 84, knee centre, 10 down |
   | Coronal fluid | 112.2 | −4.5 | +7.3 (down) | 93, knee centre, 10 down |
   | Sagittal T1/PD | 84.1 | +8.1 (forward) | −11.8 (up) | 84, 7 forward, 10 down |
   | Sagittal fluid | 108.6 | +0.1 | −3.8 | 93, 7 forward, 10 down |

   - Sagittal T1/PD narrowed every epoch to 83 mm by epoch 5 and stayed at 84 mm, and moved
     8 mm forward: the hand-found width and forward shift, where the blurred build of run 2
     sent the same view to the 140 mm bound. Its vertical offset went 12 mm up, against the
     rule's 10 mm down, which smart-crops never tested alone.
   - Coronal T1/PD is not reproducible between runs: 92 mm in run 2, here 106 mm at epoch 5
     and back to 117 mm. Fluid widths ended at 109 and 112 mm (106 and 109 mm in run 2).
   - Label bias stayed small (largest ±0.09; axial fluid for Baker's cyst +0.09 again).
   - Per label, ×4 against the recipe in force: Baker's cyst +0.0036, MCL +0.0022, medial OA
     +0.0012, PF OA +0.0011, lateral OA +0.0009; fracture −0.0060, contusion −0.0038,
     synovitis −0.0033, lateral meniscus −0.0023, medial meniscus −0.0021, ACL −0.0013,
     effusion −0.0011. Against the 120 mm control: synovitis +0.0057, fracture +0.0037,
     medial OA +0.0035, lateral OA +0.0028, ACL +0.0022, PF OA +0.0022; contusion −0.0028,
     lateral meniscus −0.0014.
   - Tokens of one view alone at ×1, last epoch: sagittal fluid 0.864, coronal fluid 0.860,
     axial fluid 0.827, sagittal T1/PD 0.754, coronal T1/PD 0.741.
   - Time: 2,110 s of training, inflated by a profiling job that shared the GPU from about
     epoch 3; before it, 0.35 s per step, the same as on the 448 px canvas, so the canvas
     size is not what limits the step.
   - Lesson: with the detail restored, learning from a centre crop finds the sagittal T1/PD
     zoom and forward shift on its own, and its score sits between the 120 mm control and
     the hand recipe. It does not find the coronal T1/PD zoom reliably, and it cannot reach
     the per-study knee anchor.

### Step time: causes and fixes (2026-10-02)

Seconds per training step, 300 steps of fold 0 on an idle GPU, real loader, full logging
unless noted (scratch scripts, not kept):

| | Main | New build before | New build after the fixes |
|---|---:|---:|---:|
| Learnable views | | 0.351 | 0.277 |
| Frozen views | 0.239 | | 0.265 |
| Logging reduced to the loss and validation score | 0.206 | 0.314 (learnable), 0.309 (frozen) | |

The canvas size was not the cause, and the main thread waited for batches only 17 to 25 ms
per step. Two defects, both fixed without changing a pixel the model sees:

- **Three slice channels handed over where one is read.** The ResNet recipe reads only the
  middle slice, but the loader shipped all three to the main process: 141 MB per batch
  against 48 MB on main, 87 ms of pinning against 24 ms. The training and validation
  loaders, and the run's final prediction, now ship the middle channel only
  (`KneeStudyDataset(middle_slice_only=...)`, set from `model.slice_channels`); the batch
  is identical to the middle channel of the old one on four studies. Measured alone, with
  logging reduced: 0.311 to 0.271 s per step.
- **Logging volume.** The per-view, per-label attention shares and the per-view gradients
  were logged with one `self.log` call each, every step: 196 calls against 136 on main,
  about 33 ms per step. They are now summed on the GPU and logged once per epoch, with the
  same names and the same epoch means.

Left as it is: the gradient through the resampling to the view values costs about 22 ms
per step (backward 93 ms against 71 ms with frozen views, synchronised timings on
pre-collated batches), and cutting the windows about 5 ms more than main's augmentation.
Logging still costs main about 33 ms per step (0.239 against 0.206).

Full runs after these two fixes took 1,419 s (frozen control) and 1,521 s (run 3 of the
learned sampling) against 1,134 s on main, with the final ×4 prediction at 76 and 91 s
against 32 s.

Second round, same day; its first full run (run 4 of the learned sampling) took 1,096 s
of training and 44 s of final ×4 prediction, against 1,134 s and 32 s on main:

- **Only the middle slice is loaded** when only it is read. Selection is unchanged and
  still works on triplets; the two neighbours are no longer read, resampled or counted in
  the canvas clipping limits, and a neighbour that fails to decode no longer drops the
  triplet. Against the build before the change, the model's inputs on 16 studies differ by
  0.29 grey levels of 255 on average, with 0.002% of pixels off by more than two levels;
  token validity and axes are identical. The loader alone delivers a batch every 0.097 s
  with sixteen workers (0.179 s before, 0.121 s on main). It applies to the training and
  validation loaders, the run's final prediction and, since the S94 bundle, Kaggle
  prediction.
- **The remaining per-step logging is summed and logged once per epoch** (OA detector,
  half-knee loss, report attributes, augmentation activity, gradient statistics); the same
  names and epoch values reach the metrics file. `train_loss` and `grad_norm` stay per
  step.
- **Measured, 300 steps on an idle GPU:** 0.235 s per step with learnable views (0.277
  before) and 0.216 s with frozen views (0.265 before), against 0.239 s on main. On a tiny
  complete run the final ×4 prediction took 42 s (75 to 91 s before).

From the image centre the model can learn the average displacement of each view. It cannot
learn to centre every study on its fitted knee centre, which is the anchor and stays
configuration. With Adam, a higher view rate scales the steady drift and the random wander of
a value by the same factor, so it makes the movement larger, not cleaner: at 0.006 pure noise
alone is estimated to move a value by about a quarter of its range (about 5 mm of shift).

The learned values themselves are the reading, not the score alone: the effect to be found is
about +0.001 to +0.002, at the edge of what one run shows.

## Learned slice budget (2026-10-02)

The user's idea, relayed from the view-config session: learn how the study's slices are
split between series types, where today the split is fixed at 6 slices per axial type and 5
per coronal or sagittal type.

- **Value.** `model.learn_slice_budget` turns one real budget per series type into a learned
  value. The six budgets are a softmax share of their configured sum (32), so only
  differences between types can move; they start at the configured 6/6/5/5/5/5 and train at
  the head rate.
- **Training, first version (runs 1 and 2).** The loader reads the budgets at the start of
  every epoch. A type with budget 5.4 loaded 6 slices, one of them, picked at random per
  draw, at weight 0.4, added as a log-weight to the attention score of that slice's tokens
  for every label. Replaced by the [second version](#second-version-of-the-learned-sampling-and-the-view-jitter-2026-10-02).
- **Study total.** Every study keeps the total its present types have under the configured
  budgets, or the 26-slice minimum; the learned shares only split that total (since the
  first run, which did not hold it).
- **Evaluation and prediction.** Budgets are rounded to whole slices by largest remainder,
  with no partial slice. With the configured budgets this reproduces the previous rule in
  every combination of present types at ×1, ×2 and ×4 (189 cases).
- **Logged.** `slice_budget_<type>` per epoch.
- **Checked on CPU.** With the configured budgets three studies load 26 slices each with
  five marked at weight 1 and a non-zero gradient on every present type. With budgets set
  to 6.4/5.8/5.2/4.9/4.6/5.1 they load 28, with weights 0.89, 0.28, 0.98, 0.67 and 0.18.
- **Kaggle prediction (2026-10-02, for S94).** The prediction worker reads the budget and
  `model.slice_channels` from the checkpoint file before it builds its loader
  (`checkpoint_sampling` in `scripts/pipeline/predict_worker.py`), so Kaggle samples with
  the learned budgets and loads only the middle slice, as the run's own final prediction
  does; an ensemble bundle uses the mean of its members' budgets and loads the middle
  slice only when every member reads only it.
- **Names.** Names differ from the view-config worktree, which calls the per-view budget `slices` and
  the study minimum `data.study_slice_total`; here they are still `max_images` (in channel
  appearances) and `data.study_triplet_total`.
- **Run 1.** `outputs/training/budget-learnable-recipe` (2026-10-02, launched by the agent
  at the user's request): `configs/training.yaml`, the recipe in force with every view
  frozen, plus `--model.learn-slice-budget`; fold 0, seed 1533697213. ×4 0.931051, −0.0018
  against the recipe in force on main and +0.0004 against the 120 mm control; ×1 0.924549
  at epoch 9. Training 1,458 s, 1.29 times the incumbent's (138 s per epoch: 117 s of steps
  and 20 s of validation), final prediction 78 s.
  - Budgets moved the same way every epoch and flattened with the learning rate: axial
    T1/PD 6 to 4.64, axial fluid 6 to 6.70, coronal T1/PD 5 to 4.46, coronal fluid 5 to
    5.71, sagittal T1/PD 5 to 4.67, sagittal fluid 5 to 5.81.
  - Per label, ×4 against the recipe in force: medial OA +0.0023; fracture −0.0056,
    effusion −0.0037, MCL −0.0029, synovitis −0.0028, contusion −0.0026, lateral meniscus
    −0.0022, Baker's cyst −0.0019, PF OA −0.0018.
- **Audit of run 1 (2026-10-02).**
  - The budget shift is not what cost the score, and it did not help either. The same
    checkpoint at ×4 scores 0.931520 with the configured budgets, 0.931199 with the learned
    ones held to each study's total, and 0.931051 as run. One checkpoint, so differences of
    0.0003 to 0.0005.
  - Loophole, fixed. The six budgets sum to 32, but axial T1/PD is absent in about four
    studies in five, so slices moved off it gave most studies 27 slices in validation and 29
    in training where the recipe has 26. The study total is now held.
  - Intensity scale differed from main, fixed. The loader clipped each series at the
    0.5% and 99.5% quantiles of the whole 144 mm canvas, where main takes them over the 84
    or 93 mm window. Against main on the same slices (12 studies, 64 series) the gain was
    0.89 on coronal T1/PD (0.81 to 1.02 across series), 1.12 on coronal fluid, 1.06 on
    sagittal fluid. The model now restretches every window between the 0.5% and 99.5%
    quantiles of its own pixels, per series and view, and the loader clips the canvas at a
    fifth of those tails so the window's limits always lie inside. After the fix the gain
    is 0.98 to 1.00 on every type and the mean difference 2 to 4 grey levels of 255.
  - Still unexplained: with the configured budgets the checkpoint is 0.0014 under the
    recipe in force. Candidates are the intensity difference above and training with
    down-weighted slices that validation never sees; no run separates them yet.
- **Run 2.** `outputs/training/views-budget-learnable-centre` (2026-10-02, launched by the
  agent at the user's request): `configs/learnable-views.yaml` plus
  `--model.learn-slice-budget`, on the fixed code; it learns spacing, both offsets, the
  label bias and the budget together from the 120 mm centre crop. ×4 0.931580, −0.0013
  against the recipe in force on main, +0.0009 against the 120 mm control, +0.0005 against
  run 1 and −0.0004 against the third learnable-views run; ×1 0.924501 at epoch 10.
  Training 1,545 s, 1.36 times the incumbent's (about 148 s per epoch), final prediction
  75 s.
  - Budgets reproduced run 1's direction and went a little further: axial T1/PD 4.12, axial
    fluid 6.72, coronal T1/PD 4.56, coronal fluid 5.94, sagittal T1/PD 4.66, sagittal fluid
    5.99. Two runs agree that the training loss prefers fluid slices over T1/PD ones.
  - Views did not reproduce the third learnable-views run. Widths ended between 103 and
    122 mm (axial T1/PD 109, axial fluid 106, coronal T1/PD 109, coronal fluid 103, sagittal
    T1/PD 111, sagittal fluid 122); sagittal T1/PD, which had settled at 84 mm there, ended
    at 111 mm here. Offsets: coronal fluid 10.0 mm down, as the hand rule; axial T1/PD
    11 mm up, axial fluid 11 mm lateral, coronal T1/PD 6 mm lateral, sagittal T1/PD 5 mm up.
    The two runs differ in the intensity fix, the budget learning and the random draws, so
    the disagreement is not attributed.
  - Per label, ×4 against the recipe in force: Baker's cyst +0.0080, PF OA +0.0016;
    fracture −0.0117, synovitis −0.0042, effusion −0.0030, contusion −0.0030, MCL −0.0012,
    medial meniscus −0.0010. Against the 120 mm control: synovitis +0.0047, Baker's cyst
    +0.0040, ACL +0.0038, PF OA +0.0028; MCL −0.0034, contusion −0.0020, fracture −0.0019.
  - Open after this run: the learned view values change from run to run at `view_lr`
    0.006, and training with partial slices costs about 0.001 even with the configured
    budgets restored at prediction (run 1's checkpoint, 0.931520 against the frozen
    control's 0.932627).

### Second version of the learned sampling and the view jitter (2026-10-02)

Built at the user's request after the frozen control tied main.

- **Full-weight slices, fixed study total.** While the budget is learned, the real shares
  are rounded at random per study and epoch (systematic rounding, so every study keeps
  exactly its total and the expected count of each type equals its share). Every slice
  counts fully. One slice per type is still marked; its log-weight is zero in the forward
  pass and only carries the gradient to the budget. Training bags now match prediction bags.
- **Shares over the present types.** Each study's shares are computed in the model over
  the types it has, so a type absent from a study receives exactly zero gradient from it and
  the gradients of the present types sum to zero. Before this, with six shares of one fixed
  sum and a constant scale from the loader, the "slices are useful" signal of the present
  types pushed the absent type's budget down: three studies without axial T1/PD gave its
  budget a gradient of 0.0027. Axial T1/PD is absent in about four studies in five and was
  the budget that fell most in runs 1 and 2 (6 to 4.64 and 4.12), so that fall is at least
  partly this artefact.
- **No freeze.** A freeze of the budget after five epochs was built and removed the same day
  at the user's request, before any run completed with it; the budget is learned throughout
  (`model.learn_slice_budget`).
- **Jitter on the view placement.** The affine augmentation's shift and scale are no longer
  a separate warp. The shift (`affine_augmentation.shift_mm`, uniform per axis, in
  millimetres, replacing `translation_fraction`) is added to the window centre before
  clipping, and the scale divides the view's spacing; only the rotation stays in the
  resampling's affine. The learned offsets and spacing receive their gradient at the
  jittered window, so a wider jitter lets them feel structure farther away, at the price of
  settling less finely. Frozen configs use 4.0 mm (4.8 and 5.6 mm for the ResNet-50 and
  ReXNet configs) and 0.95 to 1.05, today's sizes; `configs/learnable-views.yaml` uses
  8.0 mm and 0.9 to 1.1, a guess aimed at reaching the hand rule's 7 and 10 mm.
- **Checked on CPU.** Whole-number budgets reproduce the old rule in all 189 cases. With
  the learned budgets of the combined run, random rounding gives 26 slices to every study
  without axial T1/PD and mean counts equal to the shares to three decimals. In training
  mode the frozen recipe jitters 69% of series by up to 3.99 mm and 0.953 to 1.053; the
  learnable config 71%, up to 7.99 mm and 0.909 to 1.105. Evaluation-mode geometry is
  bit-identical to the build the frozen control ran on, and intensities are unchanged. On
  seven studies without axial T1/PD its budget gradient is exactly zero and the six
  gradients sum to zero. A tiny complete run on the GPU went through training, validation
  and final prediction.
- **Run 3.** `outputs/training/views-budget-jitter-centre-v2` (launched 2026-10-02 18:36 by
  the agent at the user's request): `configs/learnable-views.yaml`, budget learned
  throughout with the shares over present types, views learned from the 120 mm centre crop
  with the wider jitter, `view_lr` 0.006, fold 0, seed 1533697213. Control:
  `view-frozen-control`, 0.932627. `views-budget-jitter-centre` is the same configuration
  stopped by the user after three epochs, before the present-types fix and with the freeze;
  it carries no result.

  | | ×4 macro AUC | ×1, last epoch | Training, s | Final prediction, s |
  |---|---:|---:|---:|---:|
  | Run 3 | 0.932704 | 0.9273 | 1,521 | 91 |
  | Recipe in force on main | 0.932880 | 0.9293 | 1,134 | 32 |
  | Frozen control on this build | 0.932627 | 0.9290 | 1,419 | 76 |
  | 120 mm control on main | 0.930666 | 0.9275 | 1,149 | 40 |

  - It ties both runs of the hand recipe at ×4 (−0.0002 and +0.0001) and is 0.0020 above the
    120 mm recipe it started from; at ×1 it is level with the 120 mm control and 0.002 under
    the hand recipe. One run each.
  - Views at the end (start 120 mm, centred): axial T1/PD 108 mm, 6.8 mm medial, 4.5 mm
    down; axial fluid 121 mm, 11.4 mm medial, 2.9 mm up; coronal T1/PD 107 mm, 2.2 mm up;
    coronal fluid 117 mm, 1.7 mm up; sagittal T1/PD 105 mm, 11.0 mm backward, 2.6 mm up;
    sagittal fluid 108 mm, 1.9 mm backward, 3.8 mm up. They moved by several millimetres
    until about epoch 6 and were steady from epoch 8. They are not the hand views (84 and
    93 mm, 10 mm down, 7 mm forward).
  - Budgets stayed within a quarter of a slice of the configured ones: axial T1/PD 5.75,
    axial fluid 5.99, coronal T1/PD 5.15, coronal fluid 5.21, sagittal T1/PD 4.82, sagittal
    fluid 5.12. With the absent-type artefact, the down-weighted slices and the extra slices
    gone, the "fluid up, T1/PD down" movement of runs 1 and 2 did not reappear.
  - Per label, ×4, against the recipe in force on main, the frozen control and the 120 mm
    control: PF OA +0.0081, +0.0093, +0.0092; fracture −0.0077, −0.0055, +0.0020; synovitis
    −0.0028, −0.0032, +0.0062; MCL −0.0041, 0.0000, −0.0063; ACL +0.0016, −0.0007, +0.0051;
    lateral OA +0.0024, +0.0001, +0.0043; Baker's cyst +0.0022, −0.0002, −0.0017; the rest
    within 0.0015 of both hand-recipe runs.
  - Not attributed: the run differs from the frozen control in the learned views, the
    wider jitter (8 mm and 0.9 to 1.1 against 4 mm and 0.95 to 1.05) and the budget
    learning at once, so whether the tie comes from where the views ended or from the
    stronger augmentation is open.
- **Budget augmentation.** While the budget is learned, every study has an amount
  `data.slice_budget_shift` of its continuous shares moved from one randomly chosen present
  type to another before the random rounding (0.5 in `configs/learnable-views.yaml`, 0
  elsewhere); a share never goes below one slice. For a whole 1 this is exactly one slice
  moved; for 0.5 the donor loses a slice half the time. The gradient uses the unshifted
  shares. Simulated over 20,000 draws: with the configured budgets and 0.5, a type's count
  is one slice off in 20% of draws and never two, half the studies keep the exact split,
  totals are always 26 or 32, and the mean count stays within 0.006 of the share. Chosen
  with the user over a multiplicative factor (built first, with a factor up to 1.25: half
  of all counts one slice off and only 9% of studies unchanged, an effect that depended on
  the budget size and the rounding), a Gaussian factor (tails), a Dirichlet draw
  (unintuitive concentration, tails) and multinomial sampling (about two slices of spread).
- **Run 4.** `outputs/training/views-budget-shift-centre` (launched 2026-10-02 19:20 by the
  agent at the user's request): run 3's configuration plus the budget augmentation at 0.5,
  on the code with middle-slice-only loading and per-epoch logging; also the first full-run
  timing of those two.

  | | ×4 macro AUC | ×1, last epoch | Training, s | Final prediction, s |
  |---|---:|---:|---:|---:|
  | Run 4 | 0.932489 | 0.9273 | 1,096 | 44 |
  | Run 3 | 0.932704 | 0.9273 | 1,521 | 91 |
  | Recipe in force on main | 0.932880 | 0.9293 | 1,134 | 32 |
  | Frozen control on this build | 0.932627 | 0.9290 | 1,419 | 76 |
  | 120 mm control on main | 0.930666 | 0.9275 | 1,149 | 40 |

  - A tie again: −0.0004 against main, −0.0001 against the frozen control, +0.0018 over the
    120 mm recipe it started from, and −0.0002 against run 3. At ×1 it is 0.002 under the
    hand recipe, as run 3 was. One run each.
  - Speed: about 95 s of training and 10 s of validation per epoch, faster than main.
  - Views at the end (start 120 mm, centred; settled from epoch 8): axial T1/PD 93 mm,
    1.4 mm medial; axial fluid 135 mm, 1.4 mm lateral; coronal T1/PD 91 mm, 0.9 mm down;
    coronal fluid 120 mm, 2.5 mm lateral, 1.9 mm down; sagittal T1/PD 122 mm, 5.4 mm
    backward, 4.0 mm down; sagittal fluid 112 mm, 2.9 mm backward, 2.3 mm up.
  - Against run 3 (108, 121, 107, 117, 105, 108 mm): in both runs axial and coronal T1/PD
    ended narrower than their fluid views and axial fluid ended widest; the size of the
    move, the sagittal widths and the shifts differ. The two runs also differ in the budget
    augmentation and the middle-slice loading, so this is not a pure repeat.
  - Budgets within a quarter of a slice of the configured ones, as in run 3: axial T1/PD
    5.79, axial fluid 5.95, coronal T1/PD 5.13, coronal fluid 5.13, sagittal T1/PD 4.87,
    sagittal fluid 5.14. The budget augmentation did not make them move.
  - Per label, ×4, against the recipe in force on main, the frozen control and the 120 mm
    control: ACL +0.0041, +0.0018, +0.0075; MCL −0.0006, +0.0036, −0.0028; medial meniscus
    +0.0026, +0.0032, +0.0044; lateral meniscus −0.0051, −0.0029, −0.0041; medial OA
    −0.0004, −0.0006, +0.0018; lateral OA +0.0036, +0.0013, +0.0055; PF OA +0.0021,
    +0.0034, +0.0033; effusion −0.0028, −0.0026, −0.0012; synovitis −0.0050, −0.0055,
    +0.0039; Baker's cyst +0.0013, −0.0012, −0.0027; contusion −0.0018, −0.0016, −0.0008;
    fracture −0.0027, −0.0005, +0.0071. Shared with run 3 against the hand recipe: PF OA
    and lateral OA up, synovitis down.
- **Averages of two runs (fold 0, ×4 validation probabilities, same seed throughout).** A
  learned-view run is as different from the hand recipe as the 120 mm control is and as
  strong as the hand recipe, so the pair gains more over its better member. Not separated
  from plain seed variety: no pair here has the same views and a different seed. Filed as
  a candidate ensemble member in the [endgame backlog](../../../docs/backlog/endgame.md#ensembling).

  | Pair | Average | Over the pair mean | Over the better member | Mean logit correlation |
  |---|---:|---:|---:|---:|
  | Main + frozen control (same views) | 0.933144 | +0.0004 | +0.0003 | 0.994 |
  | Main + 120 mm control | 0.933289 | +0.0015 | +0.0004 | 0.979 |
  | Main + run 3 | 0.934165 | +0.0014 | +0.0013 | 0.976 |
  | Main + run 4 | 0.933996 | +0.0013 | +0.0011 | 0.977 |
  | Frozen control + run 3 | 0.933973 | +0.0013 | +0.0013 | 0.977 |
  | Frozen control + run 4 | 0.933911 | +0.0014 | +0.0013 | 0.978 |
  | Run 3 + run 4 | 0.933473 | +0.0009 | +0.0008 | 0.982 |
- **S94, run 4 on Kaggle: 0.947 public** against 0.950 for S93 (the hand recipe) and
  0.947 for S92 (the 120 mm recipe), at S93's inference settings (×4, τ 0.9, batch 4, no
  mirror TTA; submission 56779307, bundle `e7807c7d`, checkpoint epoch 9 of 10). Fold 0 at
  ×4 had them tied (0.932489 against 0.932880); the public score follows the ×1 order
  instead (0.9273 against 0.9293, with 0.9275 for the 120 mm control). One submission.
  A prediction-path fault is ruled out locally: the
  [parity check](#merge-of-the-view-config-schema-2026-10-02) through the worker in FP16 gives
  0.932472. The notebook's visible run took 24 s of worker time on three studies (47 s for
  S93).
- **×1 against ×4.** Both learned runs score 0.9273 at ×1, 0.002 under the hand recipe
  (0.9293 on main, 0.9290 frozen on this build) and level with the 120 mm control (0.9275),
  and gain +0.0052 and +0.0054 from ×1 to ×4 against +0.0032 to +0.0036 for the three
  fixed-view runs. Untested guesses: the wider jitter (8 mm, ±10%) leaves each slice read
  less sharply, or the learned views (fluid 112 to 135 mm) carry less detail per slice
  than 84/93 mm.
- **Run 5, the learned views frozen.** `outputs/training/frozen-learned-views-run4`
  (launched 2026-10-02 19:49 by the agent at the user's request),
  `configs/frozen-learned-views.yaml`: `configs/training.yaml` with only the six views
  replaced by run 4's values from its selected checkpoint (the S94 model), image-anchored:
  axial T1/PD 93.2 mm, axial fluid 135.3 mm, coronal T1/PD 91.0 mm, coronal fluid
  119.4 mm, sagittal T1/PD 121.6 mm, sagittal fluid 112.5 mm, with its shifts. Jitter
  4 mm and ±5%, no label bias, configured budgets 6/6/5/5/5/5 (the learned ones are
  fractional and within a quarter of a slice). It separates the two guesses: ×1 back near
  0.929 points at the jitter, ×1 still near 0.927 at the views. Controls: the frozen
  control (0.932627, ×1 0.9290) and run 4.

  | | ×4 macro AUC | ×1, last epoch | Training, s | Final prediction, s |
  |---|---:|---:|---:|---:|
  | Run 5, learned views frozen | 0.930609 | 0.9262 | 1,008 | 44 |
  | Run 4, the run they came from | 0.932489 | 0.9273 | 1,096 | 44 |
  | Frozen control, hand views | 0.932627 | 0.9290 | 1,419 | 76 |
  | Recipe in force on main | 0.932880 | 0.9293 | 1,134 | 32 |
  | 120 mm control on main | 0.930666 | 0.9275 | 1,149 | 40 |

  - The final views alone do not carry run 4's score: frozen, they are −0.0019 under it at
    ×4 and −0.0011 at ×1, level with the 120 mm centre crop (−0.0001) and −0.0020 under the
    hand views on the same build. One run.
  - The narrow jitter did not close the ×1 gap (0.9262), so the first guess is not
    supported; as fixed views these are weaker than 84/93 mm at ×1 and at ×4.
  - What run 4 adds over its own final views is therefore in how it trains: the wide
    jitter, the views moving during training, the label bias or the random slice counts.
    Not separated; the campaign's first three seeds each remove one.
  - Per label, ×4, against the recipe in force on main and the frozen control: synovitis
    −0.0078, −0.0083; fracture −0.0037, −0.0014; MCL −0.0035, +0.0006; effusion −0.0030,
    −0.0027; medial OA −0.0029, −0.0030; lateral meniscus −0.0026, −0.0004; medial meniscus
    −0.0020, −0.0015; lateral OA −0.0010, −0.0033; ACL −0.0004, −0.0027; Baker's cyst
    +0.0018, −0.0006. Synovitis falls in every run with these or similar views.
  - Averaged with the recipe in force it gains +0.0001 over the better member (run 4:
    +0.0011), so the ensemble value of a learned run is not in its final views either.
  - Time: 1,008 s of training, the first full frozen run on the current code, against
    1,134 s on main.
- **Still proposed, not built.** Learning the spread of the slices along the stack
  (`data.sampling_sigma_fraction` per type), which needs its own estimator; and using the
  jitter draws as direct probes of the loss in place of the gradient through the resampling.

## Merge of the view-config schema (2026-10-02)

The configuration schema of the view-config worktree is merged into this build (user,
2026-10-02), so a frozen and a learned recipe are written the same way:

- **Per view:** `spacing_mm`, `slices`, and `anchor` and `offset_mm` for each axis, all written
  out. `max_images` per type (in channel appearances) is gone; a type's slices are the sum of
  its views'. `data.study_triplet_total` is `data.study_slice_total`.
- **One switch:** `data.learn_views`, the list of view fields the model learns from their
  configured values (`spacing_mm`, `offset_mm`, `slices`). It replaces the per-view
  `learn_spacing` and `learn_shift` and `model.learn_slice_budget`. Empty is frozen.
- **Each view selects its own slices.** The loader tags every prepared series with its view and
  the model reads it, in place of repeating each slice once per view of its type. The budget
  logits, the random rounding, the budget shift and the marked slice are per view; with one
  view per type nothing changes.
- **Not taken from the other worktree:** its pipeline (this build's canvas replaces it), the
  view names carrying the width (a learned width changes), and root `data.spacing_mm` (here
  `selection_radius_mm`, `selection_resolution_floor_mm` and the canvas fields; the user
  deferred the name).
- **Configs:** `configs/training.yaml` is now the learnable recipe at its starting values,
  because campaign candidates train from that file; the hand recipe in force on main is
  `configs/hand-views.yaml`; `configs/learnable-views.yaml` is gone (earlier entries of this
  document name it). The ResNet-50, ReXNet and frozen-learned-views configs are converted.
- **Identity check.** On six training and six validation studies, for the hand recipe and the
  learnable one (with moved view values and learned-like budgets), the code before and after
  the merge selects the same slices and gives identical canvases, marked slices, features,
  outputs and budget, spacing and shift gradients (difference 0.0), and the same state-dict
  keys and shapes. `views-budget-shift-centre` therefore stays the reference (user: no retrain).
- **Two views on one type, CPU check.** With coronal T1/PD given a 120 mm view of 3 slices and
  an 84 mm view of 2: studies keep 26 or 32 slices, the two views draw their own slices, the
  label bias has one row per view (7 by 12), and budget and spacing gradients are non-zero.
- **Knee anchors on both axes, CPU check** (before the campaign's knee-centre seeds, which are
  the first to anchor the vertical axis and the axial views on the knee centre). On 40
  validation studies with every view knee-anchored on both axes, the window centre on a series'
  middle slice sits on the knee centre within 0.0 to 0.3 mm at the median, on both axes, in
  axial, coronal and sagittal series; 17 of 928 trusted slices were clipped and 136 untrusted
  ones stayed on the image centre. The knee centre lies 3.2 to 4.1 mm from the image centre at
  the median and 9.5 to 12.5 mm at the 95th percentile. The window is placed once per series,
  so in a tilted stack the other slices drift from the knee centre by up to 0.07 mm per mm of
  depth.
- **Kaggle-path parity of run 4** (queued after run 5): the selected checkpoint predicted on
  fold 0 through the prediction worker in FP16 scores 0.932472 against 0.932489 from the run's
  own prediction (probability difference mean 0.0006, 99th percentile 0.004, largest 0.016);
  the S94 bundle at τ 0.9 scores 0.932589. S94's 0.947 is therefore the model, not the
  prediction path.
- A control run on the merged code (`learnable-views-control`) was started by the agent and
  stopped after two minutes at the user's request; its output was removed.

## Blends submitted on 2026-10-02

Recorded in
[2026-10-02 learned-view and blend submissions](../../../docs/inference/submissions/2026-10-02-blend-submissions.md):
S94 (the reference of the learnable-views campaign alone, 0.947), S95 (with the hand-view
ResNet-18, 0.950), S96 (ReXNet + ResNet-50 + hand-view ResNet-18, 0.953) and S97 (28 networks,
0.954).

What they mean for this work: a learned-view ResNet-18 adds nothing to the hand-view one on the
public set, although the pair gains +0.0011 on fold 0, so the
[pair table of the learned runs](#second-version-of-the-learned-sampling-and-the-view-jitter-2026-10-02)
overstates what such a model adds; another architecture is what complements. On fold 0 the
campaign's first three seeds, added to the S96 trio as a fourth member, give −0.0001 to −0.0003.

## What separates S93 from the learned recipe (2026-10-03)

Asked by the user after 33 campaign runs without an adoption. Read from the saved fold-0
predictions and per-epoch logs of the hand recipe (S93 on main, `view-frozen-control` on this
build), the two frozen diagnostics and the 28 learned runs with one view per series type (the
reference and 27 campaign runs); scripts `hand_vs_learned.py` and `runs_table.py` in the session
scratchpad. Every run is one run; the spread of the 28 learned runs (SD 0.0008 at ×4, 0.0010 at
×1) is used as the yardstick and overstates run noise, since the runs differ by mechanism.

- **Size of the gap.** ×4: hand 0.932627 and 0.932880 against a learned mean of 0.931825, about
  one SD, with 4 of 28 learned runs above the main hand run. ×1: 0.9290 against 0.9273, 1.7 SD,
  with 1 of 28 above. Public: 0.950 against 0.947 for the one learned run submitted.
- **The view tables differ and learning never reproduces the hand one.** Over the 28 runs the
  final widths average 111 (axial T1/PD), 120 (axial fluid), 113 (coronal T1/PD), 114 (coronal
  fluid), 109 (sagittal T1/PD) and 114 mm (sagittal fluid), with SD 8 to 15 mm; coronal T1/PD
  never ends under 91 mm and no fluid view under 95 mm, against 84 and 93 mm by hand. Started
  on the hand views (`start-from-hand-recipe`) they widen again (coronal 84 to 105 and 93 to
  111 mm) and the run scores at the learned mean (0.931728, ×1 0.9265).
- **The view table is worth about 0.002 when nothing else differs.** Hand views frozen against
  the learned final views frozen, same build and recipe: +0.0020 at ×4 and +0.0028 at ×1,
  largest on synovitis (+0.0083), lateral OA, medial OA, effusion and ACL (+0.003 each). The
  frozen learned views score like the 120 mm centre crop. On main the same step, 120 mm to
  84/93 mm, was +0.0022 and +0.003 public.
- **The learned recipe's own gain is the label bias, not the views.** Without the bias the
  learned recipe scores 0.930803 (×1 0.9249), the level of the frozen centre crop; the bias is
  worth +0.0017 (80% interval of that run excludes zero). The narrow jitter (−0.0008) and the
  fixed budget (−0.0005) are inside the yardstick. So the two recipes reach 0.932 to 0.933 by
  different routes: the hand one by its views, the learned one by the label bias on views that
  are no better than the centre crop.
- **Per label at ×4, hand against the 28 learned runs** (both hand runs agree in sign): ahead
  on fracture (+0.004 and +0.006, 1 of 28 above), synovitis (+0.004, 1), contusion (+0.002, 2),
  effusion (+0.001, 1) and the lateral meniscus (+0.002 and +0.004, 4); behind on Baker's cyst
  (−0.002 and −0.005, 24 of 28 above) and PF OA (−0.002 and −0.003, 24). Synovitis is the label
  most tied to the view table (+0.004 with the hand views frozen, −0.004 with the learned views
  frozen). Fracture and contusion are also ahead in the two other runs with the 4 mm jitter
  (+0.002 each), so part of that gap may be the wide jitter.
- **A zoomed view does not read better alone.** At ×1 the tokens of each hand view alone score
  the same or lower than the learned ones (coronal T1/PD 0.720 against 0.737, sagittal T1/PD
  0.783 against 0.795; Baker's cyst from coronal T1/PD alone −0.11), and across the learned
  runs a narrower view is not a better one (correlation of a view's width with its own AUC
  between −0.44 and +0.45). The hand advantage appears only in the pooled prediction: zoomed
  coronal and sagittal views beside a wide axial one. This fits a gradient that does not find
  the zoom: narrowing one view costs that view its periphery at once, and the gain arrives only
  with the other views and a backbone adapted to the finer pixels.
- **Not tested: the two routes together.** The hand views frozen with the label bias on. The
  campaign excludes frozen recipes; `start-from-hand-recipe` is not this test, since its views
  drift and it trains with the wide jitter. It is a configuration-only run
  (`configs/hand-views.yaml` with `model.view_label_bias: true`).
  Queued in the campaign on 2026-10-03 10:40 CEST as the user seed
  `hand-views-frozen-with-label-bias`, a user exception to the frozen-recipe exclusion.
  Result (run by hand from the campaign's frozen source after the WSL restart of 11:20, finished
  11:57): ×4 0.933307, ×1 0.9299, against 0.932627 and 0.9290 for the same views without the
  bias (+0.0007, +0.0009), 0.932489 and 0.9273 for the learned reference, and 0.932880 for S93 on
  main. One run; the harness's paired comparison is the keeper session's to record.

## Review of the campaign's runs (2026-10-03)

A read-only review of the 33 trained and scored ideas, the reference and the three controls,
asked by the user: which mechanisms move which label, whether each learned quantity learns, and
what the per-epoch curves show. Tables and scripts: `outputs/analysis/l/learnable-views-review-20261003/`
(`runs_deltas.csv`, `single_change_pairs.csv`, `mechanism_by_label.csv`,
`learned_parameter_summary.csv`, `label_bias_by_view_label.csv`, `per_epoch_curves.csv`,
`curve_summary.csv`). Every idea is one run. Yardstick over the 34 learned runs: ×4 SD 0.00083,
×1 SD 0.00117; single-change pair deltas have SD 0.00088 at ×4.

- **The reference sits above the average learned run.** ×4 0.932489 against a campaign mean of
  0.931904 (+0.0006); every campaign run is below it on the medial meniscus and 32 of 33 on
  ACL. The two frozen runs trained before the schema merge show the same deficit against it, so
  it reads as a favourable draw on those two labels, not as an effect of the merge. Deltas
  against the reference understate a mechanism by about 0.0006. The code is not the cause: the
  merged code gives identical slices, images, outputs and gradients on twelve studies
  ([merge check](#merge-of-the-view-config-schema-2026-10-02)), and the harness's calibration
  at the campaign's start (a CPU replay of the sampling) passed. Neither is a retraining:
  training does not repeat from run to run. The keeper session saw two runs of one recipe,
  `knee-frame-3d-position` and its EMA variant before the averaging starts, match in training
  loss to 0.0001 at epoch 1 and differ by 0.0062 in validation AUC (0.8937 against 0.8999).
- **What each learned quantity does** (runs that learn it):

  | Quantity | Behaviour | Verdict |
  |---|---|---|
  | Width, coronal and sagittal views (28) | 120 mm to a mean of 109–114 mm, SD 8–12; narrow starts widen to the same range; set by epoch 2 to 4 | drifts to about 110 mm from both sides, random within ±10 mm |
  | Width, axial views (28) | SD 10 to 15 mm, as many up as down; axial T1/PD is absent in 731 of 912 studies | random |
  | Shifts (24) | means within ±2.5 mm, SD 3 to 7 | random |
  | Slice budget (27) | same direction in 26 of 27 runs, by a quarter of a slice | reproducible, too small to change the counts |
  | Label bias (27) | 59 of 72 entries keep their sign in at least 24 of 27 runs; largest 0.12; 90% of the final value by epoch 4; rank correlation +0.63 with which view reads the label best alone | learns the same table every run; moves attention by under 0.02 |
  | Centre weights (11) | sagittal fluid up in 10 of 11; the centre ends 0.2 to 1.0 mm from the computed one; 99% of its gradient comes from the window path | moves, no effect |
  | Slice spread (4) | coronal spreads collapse from 0.867 to 0.17–0.32 in 3 of 3 unfloored runs; axial fluid widens to 1.4–1.7 in 4 of 4 | reproducible, and it hurts |

- **Mechanisms with a repeating label pattern.** Learned slice spread: ×4 negative in 3 of 3
  introducing pairs (−0.0024, −0.0016, −0.0006), MCL −0.015 and −0.019, +0.015 back with a
  floor; training loss the lowest of all with no validation gain. A second, narrower
  sagittal-fluid view (5 runs, one parent): lateral meniscus +0.004, synovitis +0.003, PF OA
  +0.0025, ACL −0.002; the ordered pair and its routing child reach the hand recipe's ×1
  (0.9291 and 0.9301) with its curve shape. 3D position in place of 1D and 2D: negative in both
  pairs (−0.0010, −0.0001), medial OA and Baker's cyst up in 6 of 6, synovitis and lateral
  meniscus down in 6 of 6, and the OA detector silent in 5 of 5.
- **Inside one pair's noise:** knee anchors, the learned centre (+0.0011 over its control, the
  family of 11 at +0.0002), the first-order OA detector (+0.0014 and −0.0005), jitter, fixed
  budget, view schedules and optimizer, clipping gradients, EMA, photometric Jacobians, the
  hand start. Clearly negative: views learned on held-out studies (below at every epoch),
  the free slice spread, removing the label bias (below from epoch 3).
- **Curves.** ×1 at epochs 0 to 5 does not predict the final ×4 (correlation about 0; 0.31 to
  0.45 from epoch 6). Runs with narrow views start 0.005 to 0.007 behind at ×1 and finish
  ahead (both hand runs and the ordered sagittal pairs). The gain from ×1 to ×4 averages
  +0.0044 with SD 0.0011, more than the SD of ×4 itself; it is smallest for the free-spread
  runs and the hand recipe. No dip in validation follows the epochs where the widths move.
- **Faults seen in the logs.** The complementary depth contrast of
  `ordered-sagittal-pair-with-complementary-depth-routing` is zero from epoch 5; the
  widen-only spread floor binds on 21 to 88% of steps, so that run tests "coronal fixed, axial
  free"; in `two-views-sagittal-fluid` the 93 mm view ends at 139 mm, wider than its sibling;
  the learned budget shares round back to 6/6/5/5/5/5 at ×1.

## Campaign (started 2026-10-02 21:05 CEST on Codex)

Written as the [learnable-views campaign](INDEX.md):
contract, prior work, definition and fourteen forced seeds; `validate` passes. The user judged the
open questions campaign material on 2026-10-02.

- **Goal:** improve the learnable recipe (`configs/training.yaml` on this branch: views, label
  bias and slices learned from the 120 mm centre crop). Frozen-view recipes are not seeds
  (user): a frozen run is a diagnostic for a learned one, not a candidate.
- **Reference:** run 4, `views-budget-shift-centre` (×4 0.932489, ×1 0.9273, S94 0.947 public),
  kept by the user in place of a retrain on the merged code, with declared exceptions and the
  [identity check](#merge-of-the-view-config-schema-2026-10-02) as their proof.
- **Seeds, in order:** two learned views on sagittal fluid; two on every type (the user's first
  named direction); every view anchored on the knee centre with learned offsets; the knee
  centre fit with a learned weight per series type (the user's second direction, a learnable
  centre); one side-normalized 3D position per token measured from the learned centre, in
  place of the 1D and 2D positions (user: retake it now that the centre is learned; worth about
  zero three times with the computed centre, never with a centre it could teach); the spread
  of each view's slices along the stack learned from 0.867 (user); label
  bias off; normal jitter; slices fixed; start from the hand views; views frozen after five
  epochs.
- **Three seeds on how the views are learned** (agreed with the user): the view values
  updated only from a tenth of the training studies that the weights never see (they are
  trained on the training loss today, and frozen they score like the centre crop); the view
  values stepped on a gradient averaged over about a hundred steps (they wander and end
  elsewhere in each run); and the views kept at their start for three epochs (most of their
  movement happens while the model is untrained).
- **Endless research** (user): `rotate_after: 1000000`, and the researcher never returns an
  empty refill.
- **Diagnostic proposed, not run:** the hand views with the wide jitter (8 mm, ±10%), a frozen
  recipe and therefore outside the campaign.
- **Combinations** (user): the researcher is told that a rejected seed is not a dead mechanism
  and that, after the seeds, its main work is to combine what acted, from one table of every
  decided run.
- **Prior evidence on the spread:** the overall scale of the sampling Gaussian was inert on
  macro in September (0.60 to 1.10, span 0.00674, seed noise) and the centre along the stack
  sat inside a ±30 mm band of flat utility
  ([peripheral coverage](../../../docs/data-preparation/peripheral-coverage.md#what-is-settled-and-what-is-not));
  what is untested is a width per view, with labels whose evidence sits at different depths.
- **Learned centre, as agreed with the user:** the centre differs per study, so the rule that
  computes it is learned, not the centre. The fit keeps its form and gets a weight per series
  type, equal at the start; it has no free offset, so it cannot take over a view's offset and
  stays inside what the study's series say, with no hard-coded bound. Slice selection draws
  around the learned centre and teaches it through a per-slice weight that is zero in the
  forward pass, as the budget is learned. One centre: the meniscus mask, the OA detector
  regions and the half-knee loss measure medial, lateral and anterior from it too, with their
  hard cut-offs (they follow it and give no gradient); soft cut-offs are left to the
  researcher.
- **Hand-off of learned loader-side values (built for it):** the model's `loader_state()` is
  passed to the datasets' `set_loader_state` each epoch and saved in the checkpoint; the final
  prediction and the Kaggle prediction worker pass it on, so a campaign idea can add a learned
  quantity the loader needs without a script change. The identity check against the code
  before the merge still gives a difference of 0.0. Checkpoints saved before this change carry
  no `loader_state` and cannot be predicted by this code.
- **Evidence asked of every run:** ×1 beside ×4, and where the views and budgets ended.
- **Budget:** training time between 75% and 125% of the reference's 1,096 s (822 to 1,370 s; user).

## Left open at the close

- Whether the affine augmentation hides the signal for the learned views. It moves every
  window at random by up to 4% of its width per axis (4.8 mm at 120 mm, 3.4 mm at 84 mm),
  scales it by 0.95 to 1.05 and rotates it by up to 10 degrees, on 70% of series. That
  trains the model to be indifferent to shifts and scale changes of that size, which are the
  quantities the learned offsets and spacing move by. Reasoned, not measured; a check would
  compare the consistency of the view gradients over batches with the augmentation on and
  off, on an existing checkpoint.

- The result of `view-learnable-centre` against the recipe in force.
- A full run's `training_seconds` on the fixed build, expected near 1.26 times main's.
- Preparing only the middle slice of each triplet would cut the loader's work to about a
  third, in training and in the final prediction; it changes the intensity quantiles
  slightly, which needs a parity check first.
- Kaggle prediction time and memory with the canvas, to be measured on T4s.
- How to confirm the final recipe outside fold 0: a second fold, or a public submission as with
  S93. About 30 choices have been made on the same 912 studies.
