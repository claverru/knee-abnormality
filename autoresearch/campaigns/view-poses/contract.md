# View-poses campaign contract

Written 2026-10-08; started the same day. Sections marked *Open* are not decided; a seed that depends on
one cannot be written until it is.

## Objective

Raise the macro AUC of the anatomy-map recipe by letting each view show what it is read for, at
the resolution that needs:

1. **Which canvases:** the crops are smart in every run, so what the campaign tries in
   combinations is the canvas of each plane-contrast, fixed before training: its pixel size,
   which sets how far that view can zoom, and its extent. Today's assignment was never decided:
   the T1/PD views have the finest canvases and carry the least, and axial fluid, the best
   single view for five labels, has the coarsest ([prior work](prior-work.md#each-view-alone)).
2. **Soft-tissue structures:** the model learns where the menisci and the cartilage are, from
   masks made offline on the 3D DESS series, beside the bone map it already learns.
3. **Canvases:** the loader's canvases are sized so the pose can move as far as knees need and
   no further.
4. **A pose per plane-contrast:** every view, axial included, is shifted, turned and scaled to
   the knee from what the first look predicts.
5. **Detail crops:** a tight crop on the structures a view is read for, beside the wide crop
   that fluid, cysts and marrow need.
6. **Routing by structure:** each soft-tissue label reads the tokens of its own structure.

The reference and first incumbent is `rn18-meniscus-map` (`outputs/training/rn18-meniscus-map`,
trained by the user on 2026-10-08: fold 1 of `all_5fold_scanner_v2`, seed 1110484336, ResNet-18,
224 px, ten epochs). It is `report-attribute-loss-off-2` (×4 0.927291, ×1 0.917342, 1,352 s)
plus an independent three-class meniscus head (neither, medial, lateral) taught at weight 0.1
on the DESS masks (`data/labels/prepared/meniscus-nnunet-knee-dess265`). Its crops are those of
the recipe: sagittal and coronal posed without scale, axial unposed.

The forced seeds, in order. **What the poses are built on:** the meniscus head on the 14 by 14
grid, and the knee's centre taught from the masks. **Room for the pose:** one seed gives every
view its 368 px canvas, the per-knee scale and the axial pose, with the fields unchanged; from
then on every run has smart crops in all six views. **Where each crop looks:** a centre of its
own for each plane-contrast, then two seeds that each move one view's centre. **Canvas
combinations:** four trades of fine and coarse canvases. **Routing:** the meniscus labels read
by meniscus share.

## Settled

- **Every run has smart crops in every view** (user, 2026-10-08): shift, turn and scale per
  knee, in sagittal, coronal and axial, during training and at prediction. No experiment of the
  campaign runs on fixed crops once the canvas seed has given every view its pose.
- **One crop per plane-contrast** (user, 2026-10-08): each of the six views has its own pose,
  centred on what that view is read for. The two contrasts of a plane no longer share a centre.
- **The canvases come before the centres** (user, 2026-10-08, after the reference's maps were
  read): the centre seeds ask for moves of 8 to 13 mm on a knee that already wanders 7 to 8 mm up
  and down in the canvas, and the recipe's 12.75 mm limit would clamp them. The canvas seed keeps
  every field and only gives the crops room.
- **Square canvases of 368 px for every view** (user, 2026-10-08): a margin of 72 px on each
  side, which holds the pose in 99% of images whatever the view's field.
- **The meniscus head is in the reference**, on the 7 by 7 tokens; the first seed moves it to
  the 14 by 14 grid, with the bone map and the label path left on 7 by 7. Cartilage has no head.
- **Crops are centred on the structures a view frames**, read from the predicted map per knee;
  the four hand-measured centre offsets go.
- **224 px.** Centring the token grid (225 and 257 px) and removing the border tokens changed
  nothing ([token grid alignment](../../../docs/validation/results/crops-and-views/2026-10-08-token-grid-alignment.md)).
- **Masks on a subset.** The soft-tissue masks exist for the DESS studies only (about 265, one
  scanner); the model generalises from them as it does from the 550 bone masks.
- **All series of a study are supervised** by its mask, through patient coordinates; a token
  outside the DESS slab has no soft-tissue target.
- **An independent head** for the soft-tissue structures. The seven-class bone map, which fits
  the joint plane and the centre, is not changed by it.
- **The pose comes from predicted anatomy, never from the label loss** and never from a
  hand-measured correction constant.
- **Canvases stay rule-based and bounded:** no whole-slice canvases; sizes from the measured
  need ([prior work](prior-work.md#canvas-sizes)).
- **Every plane gets a pose:** a crop mechanism covers sagittal, coronal and axial together.
- **The canvases are tested in combinations, not assumed** (user, 2026-10-08): they are fixed
  before training, so the campaign runs different sets of six canvases. Each canvas comes from
  a measured extent, not from a sweep.
- **Turning pays for itself in zoom:** a crop that turns with the knee can be up to 27% tighter
  in axial for the same anatomy.

## Open

- **What each view frames** within its field: a first table exists from the literature and
  from inspection of our images ([prior work](prior-work.md#what-each-view-is-read-for)).
  Which views have fine canvases is no longer open: it is what the combination seeds vary.
- **Where the detail crops' slices come from:** taken from the views that add least (axial
  T1/PD, then the sagittal and coronal T1/PD views) or added on top at more time.
- **Cartilage from the sagittal 3D PD series** (about 130 more studies, two more scanners):
  in or out.
- **The first look.** Whether the soft-tissue head can place a box from 12 mm tokens, or the
  finer token grid is needed first.

## Principles

1. **Geometry is taught, not discovered.** Planes, centres, sizes and boxes come from predicted
   anatomy, and the anatomy learns from its own targets.
2. **Structures are used as shares.** No class by argmax, no threshold without a measurement.
3. **A number set by hand comes from a cited measurement**, taken on the masks or on saved
   runs, never from a sweep on the validation labels.
4. **Everything is computable for an unseen knee** from its own images through the model. No
   segmenter at validation or prediction; nothing from a segmenter in the loader.
5. **Zoom is a trade across views.** A view zoomed in gives up field that some labels needed
   there; a combination says which other view keeps or gains that field, and every label keeps
   at least one of its strong views at the field it needs.
6. **Resolution is the lever, not field.** Widening a view for every knee lost twice; a seed
   that enlarges a field says which structure enters and what resolution it costs.
7. **One mechanism per run.** A seed changes one thing and states what it removes.
8. **A seed can be proved active.** Every seed allows one validation-only metric that shows its
   change is applied, or the review blocks it.

## How a seed is written

One training run, stating: **evidence** (numbers with their file or run), **mechanism** (what
changes, with files and formulas), **replaces**, **shown wrong by**, and a **label** per change
(evidence-backed or a guess). A seed that reuses a mechanism already reviewed names that
candidate's source instead of describing it again.

## Out of bounds

Running or adding a segmenter; anything looked up by study identity at validation or prediction;
trained models in the loader; fixed crops in any view; whole-slice canvases; widening a view for
every knee without a trade; a correction constant measured by hand on the centre, the plane or
the pose; letting the label loss move a plane, a centre, an angle, a size or a box; image size
(224 px stays) and slice counts; fewer epochs; hyperparameter sweeps; weight averaging;
reflection-probability changes; gold-study weighting; multi-seed or confirmation runs; other
backbones; choosing anything on the campaign fold's validation labels.

## Cost

A run adds no training time its mechanism does not need: no new diagnostics in the training
step, no operation that forces the GPU to synchronise, no second evaluation of the pose, frame,
routing or tokens, no extra forward pass beyond the single first look. Each seed states its
expected time against the incumbent: the meniscus head within 1.03; the centre seeds, the
canvas combinations and the routing within 1.02; the canvas seed up to 1.25, from its canvases
and resampling.

**The budget is 0.5 to 1.4 of the reference's training time** (the runtime refuses anything
above). The ceiling is not room to spend: it has to hold what gets adopted, since the limit is
fixed against the reference while each adopted seed makes every later run slower. The head and
the canvas seed together bring the incumbent to about 1.25 to 1.3 times the reference; 1.4
leaves a later seed about 10%.

## Guardrails

**Before a run trains**, the developer runs
`rtk proxy uv run --locked python -m scripts.tools.check_side_routing --config configs/training.yaml`
in the candidate and reports its JSON. A candidate that fails does not run. The script is not
edited by a seed.

## Duration

Started with `start`; no end. After the forced seeds the researcher keeps proposing on the
stages of this contract until the user stops it, and never returns an empty report.

## Adoption

Fixed by the user on 2026-10-08. The runtime applies the general rule by itself: a run is
adopted when its macro AUC is at least the incumbent's at ×1 and at ×4, and the next candidate
builds on it. The 80% paired-bootstrap interval of the ×4 change is reported with every run and
does not decide. The two exceptions of the table (seeds 1 and 3) and the cost and defect
conditions below are applied by the keeper without asking, with
`python -m scripts.tools.autoresearch adopt view-poses --experiment EXPERIMENT_ID`, citing this
section, before the next training launches; the keeper requeues any candidate prepared on the
previous incumbent.

| Seeds | Rule |
|---|---|
| 1, `meniscus-head-on-the-14-by-14-grid` | Adopted whatever its macro result. It is judged on its purpose: if both meniscus classes do not overlap the mask better than in the reference, the candidate is defective and is repaired and rerun |
| 3, `canvases-for-the-pose-in-all-six-views` | Adopted unless clearly worse, meaning the central 80% paired-bootstrap interval of its ×4 macro AUC change lies entirely below zero. If it is clearly worse it is not adopted, the keeper reports it and the campaign goes on with the remaining seeds on the incumbent; the keeper does not stop or ask |
| 2 and 4 to 11, and every researcher idea | Adopted when the macro AUC is **at least the incumbent's at both multipliers**: ×1, the selected checkpoint's validation macro (`score` in `selected_checkpoint.json`), and ×4, the macro of the saved predictions (`prediction_macro_auc`) |

- A run level or better at one multiplier and lower at the other is not adopted. A run that is
  not adopted leaves the incumbent unchanged and its results feed the next design.
- **Cost:** a run that trains longer than its seed stated, or predicts in more than 1.3 times
  the incumbent's time, is not adopted until that is explained, whatever its score.
- `joint-centre-taught-from-the-masks` must also lower the centre's error against the masks; if
  it does not, the candidate is defective.
- The ×1 and ×4 scores read different slices of the same knees, so they are two partly separate
  readings of one model. Of the seven runs of the anatomy-recipe campaign this rule adopts two
  (the report loss off, the per-knee scale) and rejects the five that were lower on at least one.

Adoption on the fold decides what the next seed is built on. This line has moved the public
score while the fold stayed level (S119 to S126: fold level, public +0.002), so which models are
submitted stays the user's decision.

## Evaluation

Against the campaign incumbent at ×4 and ×1, per target with each target's run-to-run spread,
with the bone map's overlap per class, the joint plane's error, the soft-tissue head's overlap
per structure on the validation DESS knees, and the pose statistics per view (shift, turn,
scale, share of crops at a limit).
