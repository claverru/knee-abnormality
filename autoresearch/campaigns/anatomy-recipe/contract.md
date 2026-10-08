# Anatomy-recipe campaign contract

## Objective

Raise the macro AUC of the side-routed model by exploiting the anatomy masks of a pretrained
segmenter, used offline as training targets only. The campaign continues the work of 2026-10-07
recorded in [prior work](prior-work.md), all of it together:

1. **The anatomy map:** what the model is taught about each token (bones, joints), with which
   loss, and how much of that teaching reaches the backbone.
2. **Sections and routing:** how the map and the joint plane fitted to it decide which tokens
   each label reads.
3. **Crops:** how the sagittal and coronal crops are shifted and turned to the knee's own pose,
   read from the map on a first look.
4. **Measurements as targets:** numbers read from the masks (the joint gap per side, the
   kneecap's position) taught beside the labels.

The base and starting incumbent is `rn18-anatomy-crops-sample550`
(`outputs/training/rn18-anatomy-crops-sample550`: fold 1 of `all_5fold_scanner_v2`, seed
1110484336, ResNet-18, ten epochs), the first run with all four parts of the design in one
model; its results are in [prior work](prior-work.md#the-incumbent). The yardstick outside the
campaign is run 24 (`outputs/training/20261006-163650-side-routed-spot-frame-ring-sectors`: ×4
0.925067, ×1 0.917070, 0.953 public as S118). The campaign works locally and makes no Kaggle
submission.

## The rule about the segmenter

The segmenter (MRAnnotator) ran once, offline, on training and validation studies. Its masks
exist only as the prepared targets under `data/labels/prepared/anatomy-mrannotator-sample550-joints/`.
No seed runs a segmenter, adds one to the loader or the model, or needs one at validation or
prediction. Everything a knee is given at prediction comes from its own images through the
model.

## Scope

**Topic 1, the anatomy map.** A seed may change:

- the map's head, its loss terms and their balance, and the classes it predicts as long as they
  are derived from the seven prepared classes;
- how much of the map's gradient reaches the backbone;
- the weight of the whole anatomy loss (`trainer.anatomy_loss_weight`).

**Topic 2, sections and routing.** A seed may change:

- how the joint plane and the knee's centre are read from the map;
- how a token's section is decided from its shares and its height, and which sections each route
  reads;
- the regions around the knee (core, ring, sectors) and the table from regions to targets,
  including a rule for one label only.

**Topic 3, crops.** A seed may change:

- where the joint centre is placed in the sagittal and coronal crops and at which angle they are
  cut, from the pose of the first look;
- the canvas margin (`data.view_shift_limit_mm`), the limits on turn and shift, and the random
  jitter in training;
- how the first look is made: which slices it reads and what it reads from them. There is one
  first look per knee, without gradient, in evaluation mode.

**Topic 4, measurements as targets.** A seed may add targets read from
`measurements.json` of the prepared directory (the joint gap on each side of the patient, the
kneecap's extent relative to the joint), with their heads in the training module, and may read
them in regions.

Fixed in every topic: the backbone and its weights, the width in millimetres and the pixel size
of every view, the number of slices per view and per study, the 120 mm axial views (never moved
or turned), the prepared targets, the augmentations other than the crop jitter, the optimiser,
the schedule and the epochs, the twelve labels and their loss, the weights of the loss terms
other than the anatomy loss, and the inference multiplier and temperature.

## Principles

1. **The geometry is taught, not discovered.** The joint plane, the sections and the pose come
   from the map, and the map learns from its own targets. The label loss never moves a plane, a
   centre or an angle (every earlier frame that learned its geometry from the labels collapsed
   or drifted).
2. **What a token is comes before where it is.** A token the map calls femur, tibia or joint
   belongs to that section; only what the map leaves undecided is placed by height.
3. **Shares are used as shares.** No class is taken by argmax, and nothing is thresholded
   without a measurement.
4. **Shared classifiers stay blind to the compartment.** Geometry alone assigns a shared output
   to a target, and the regions of a mirrored pair are mirror images.
5. **A number set by hand comes from a measurement** the seed cites, taken on the masks or on
   saved runs, never from a sweep on the validation labels.
6. **Everything is computable for an unseen knee** from its own images through the model.
7. **Teaching costs something.** The anatomy loss shapes the backbone; Synovitis fell in every
   run that trained the map ([prior work](prior-work.md#the-cost-to-synovitis)). A seed that
   strengthens the map says what it expects for Synovitis, Effusion and PF OA.
8. **One mechanism per run.** A seed changes one thing and states what it removes.

## How a seed is written

Each seed is one training run and states:

- **Evidence:** the numbers or drawings that motivate it, with their run or file.
- **Mechanism:** what changes, exactly, with the files and the formulas.
- **Replaces:** what it removes. A seed that only adds says why nothing can go.
- **Shown wrong by:** the logged value or per-target result that would contradict it.
- **Label:** evidence-backed or a guess, per change.

A seed never only measures: every seed trains, and what it needs to know it logs. Per epoch, on
the validation knees, every run logs the map's overlap per class, the joint plane's error, and
the mean and spread across knees of every turn and shift it applies.

Out of bounds: running or adding a segmenter; hyperparameter sweeps; changes to the weights of
the label, side or report losses; hand-set distances, angles or regions without a cited
measurement; crop widths, image size, slice counts; finer tokens; more than one first look, or a
first look that reads more slices than a ×1 pass; anything looked up by study identity at
validation or prediction; trained models in the loader; fewer epochs; weight averaging;
reflection-probability changes; the class-balance exponent; gold-study weighting; multi-seed or
confirmation runs; other backbones; choosing anything on the campaign fold's validation labels;
an offset pattern or an average over displaced crops at prediction.

**Lifted by the user on 2026-10-08 for two forced seeds:** the bar on crop widths, for
`coronal-fluid-crops-at-108-mm` and `sagittal-fluid-crops-at-112-mm`, whose field comes
from bone extents measured on the 550 mask volumes; and the bar on the report loss's weight, for
`report-attribute-loss-off`, which sets it to zero.

Closed by the anatomy-masks campaign ([its run notes](../anatomy-masks/run-notes.md)): a tenth of
the map's gradient to the backbone; the map's loss without Dice; a first look on fewer slices,
end or interior; the first look on the whole canvas; both series of a plane a ninth wider; a
fixed forward move of the fluid sagittal crop; the joint gap in millimetres; subpixel crop
shifts; projecting the map's gradient away from the label gradient; soft bone shares in the plane
fit; unsaturated kneecap shares; equal view exposure in the joint centre.

Closed by the runs of 2026-10-07 ([prior work](prior-work.md)): a height predicted per token as
the plane's source; the femur's or tibia's direction read from a line through their tokens; a
map head that reads the tokens without any gradient reaching the backbone; sections chosen by
route with the front regions including the joint row; a core radius learned from the labels;
random crop offsets around an unposed crop.

## Cost

Fixed by the user on 2026-10-08. A run adds no training time its mechanism does not need: no new
diagnostics in the training step (a seed's check is reported once on one batch, or logged on
validation only, at most five new metrics), no operation in the training step that forces the GPU
to synchronise (`.item()`, `nonzero`, `unique`, boolean-mask indexing, Python loops over studies,
series or views), no second evaluation of the pose, the frame, the routing weights or the
tokens, and no extra forward pass. A seed that states the control's cost trains within 1.02 of
its time; the coronal field seed and the kneecap seed within 1.08 and the sagittal field seed
within 1.1, from their larger canvases only; the runtime refuses anything above 1.1. Where the
time of the anatomy-masks runs went is in
[Training runtime and memory](../../../docs/modeling/training-runtime-and-memory.md#posed-crop-recipe-where-a-step-goes-and-what-was-removed-2026-10-08).

## Guardrails

**Before a run trains**, the developer runs
`rtk proxy uv run --locked python -m scripts.tools.check_side_routing --config configs/training.yaml`
in the candidate and reports its JSON. A candidate that fails does not run: the developer
repairs it or returns it blocked. The script is not edited by a seed.

A topic 3 candidate also confirms on one batch that with the turn and shift at zero its crops
equal the central crops of the canvases, largest absolute difference at most 1 grey level of
255, in training and in evaluation mode, and reports the residual angle of the joint plane in
the re-cut crop when the plane is known.

## Adoption

Fixed by the user and applied by the keeper without asking. The runtime computes its shared
verdict; the keeper adopts with
`python -m scripts.tools.autoresearch adopt anatomy-masks --experiment EXPERIMENT_ID`, citing
this section, every valid run that passed the pre-run guardrails and whose macro AUC is **at
least the incumbent's at both multipliers**:

- **×1:** the selected checkpoint's validation macro (`score` in `selected_checkpoint.json`);
- **×4:** the macro of the saved predictions (`prediction_macro_auc`).

No seed of this campaign is adopted by exception.

A run level or better at one multiplier and lower at the other is not adopted. A run that is not
adopted leaves the incumbent unchanged and its results feed the next design. Every run reports
its prediction time against the incumbent's; one above 1.3 times is not adopted, whatever its
score.

## Duration

The campaign is started with `start`, not under the topic loop, and has no end: after the forced
seeds the researcher keeps proposing on the four topics until the user stops it, and never
returns an empty report.

## Evaluation

The [shared evaluation](../../protocol.md#shared-evaluation-and-adoption) applies against the
campaign incumbent. Every result also reports ×1, run 24's two figures, the per-target deltas
against the incumbent with each target's sibling standard deviation
([prior work](prior-work.md#yardstick)), the map's overlap per class, the joint plane's error and
the crop statistics.
