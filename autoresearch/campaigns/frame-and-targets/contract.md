# Frame-and-targets campaign contract

## Objective

Improve the side-routed model on three topics at once, continuing the
[routing-and-shifts campaign](../routing-and-shifts/INDEX.md) from its incumbent:

1. **The knee frame:** the angles and regions of the frame each knee is given, measured from the
   heads' own attention spots where they are learned from the labels today.
2. **Crops that follow the frame:** each view's crop turned and moved to the knee's own frame
   instead of cut at the scanner's orientation.
3. **Located report targets:** the two new v5 report answers (the bone of a marrow edema, the
   extensor tendon involved) as targets read in regions, and which report questions are worth
   their head.

The base and starting incumbent is `height-cuts-from-the-spots-requeue` of the previous campaign
(`outputs/autoresearch/routing-and-shifts/experiments/height-cuts-from-the-spots-requeue-20261006T232351Z-d2445402/training`:
fold 1 of `all_5fold_scanner_v2`, seed 1110484336, ResNet-18, ten epochs, ×4 0.925389, ×1
0.917411, 0.952 public as S120). It is run 24 of
[the line](../../../docs/ongoing/side-routed-heads.md) with the two height cuts placed per knee
from the meniscus and kneecap spots. The campaign follows the three topics locally and makes no
Kaggle submission.

## Scope

**Topic 1, the knee frame.** A seed may change:

- how the frame's centre, its three angles and the per-knee sizes of its regions are obtained;
- the shape of the regions above the upper cut and below the lower cut, including a band laid
  out around its own axis as long as it stays attached to the joint band at the cut;
- the routing: the cuts, the table from regions to targets, how a cut passes gradient.

**Topic 2, crops that follow the frame.** A seed may change:

- where and at which in-plane angle each 84 and 93 mm view's crop is cut, per knee, from the
  frame the model itself infers for that knee;
- the image the loader hands over, only as far as a movable crop needs: a wider canvas at the
  same pixel spacing (`data.view_shift_limit_mm`), from which the model cuts the crop at the
  view's fixed size in one resampling;
- how the frame for the cut is obtained. At prediction and validation: **one first look**, a pass
  of the same model without gradient on the crops at the hand-set position, from which the frame
  is read; the prediction is then made on the re-cut crops. In training: either the same first
  look, or the knee's own frame as the model computed it the last time it saw that knee in
  training, kept in memory per study. Nothing kept per study is used at validation or prediction;
- the range of the random affine augmentation (`data.affine_augmentation`), in the seeds that
  state it, since aligned crops vary less.

**Topic 3, located report targets.** A seed may change:

- which report answers are targets and in which regions each is read, including the two new
  answers of the v5 extraction, read from the prepared file named in
  [prior work](prior-work.md#the-v5-report-answers) by code in `training/report_attributes.py`;
- which report questions keep a head.

Fixed in every topic: the backbone and its weights, the width in millimetres and the pixel size
of every view, the number of slices per view and per study, the 120 mm axial views (never moved
or turned), the augmentations other than the affine range and where the affine is applied, the
optimiser, the schedule and the epochs, the twelve labels and their loss, the weights of the
existing loss terms, and the inference multiplier and temperature.

## Principles

1. **Shared classifiers are blind to the compartment.** Geometry alone assigns a shared output
   to a target, and the regions of a mirrored pair are mirror images.
2. **A token is one point.** It belongs wholly to the region its centre is in; a cut is a hard
   step with a soft gradient. No token is shared across regions by overlap.
3. **The frame is inferred per knee**, and the report attributes are routed.
4. **Regions are generous.** Leaving evidence out costs and including too much is free (runs
   18, 19 and 30): a seed that narrows a region says which measurement allows it.
5. **A cut or an angle needs a pull from both sides.** A cut with targets on one side only
   drifts (run 26), one with targets on both sides stays (run 30), and an angle only the labels
   of one band act on goes to one side for every knee (the femur-lean run of 2026-10-07: −25°
   for all knees within two epochs).
6. **Measured before learned, learned before hand-set.** Geometry is read from the attention
   spots without gradient where a spot marks it; what no spot marks may be learned; a number set
   by hand must come from a measurement the seed cites.
7. **Everything is computable for an unseen knee.** Whatever places a crop or a frame at
   prediction comes from that knee's images through the model. No trained model runs in the
   loader.
8. **Readers leave the backbone alone** unless the seed says otherwise: whatever reads the
   side, a position or a spot from the tokens reads them detached.
9. **A landmark is used for what it marks.** The meniscus spots mark the joint line and the
   side axis; the kneecap spot marks the kneecap, not the femur's direction.

## How a seed is written

Each seed is one training run and states:

- **Evidence:** the numbers or drawings from earlier runs that motivate it, with their run.
- **Mechanism:** what changes, exactly, with the files and the formulas.
- **Replaces:** what it removes. A seed that only adds says why nothing can go.
- **Shown wrong by:** the logged value or per-target result that would contradict it.
- **Label:** evidence-backed or a guess, per change.

A seed never only measures: every seed trains, and what it needs to know it logs. A seed that
changes an angle, a cut or a crop logs, per epoch and on the validation knees, the mean and the
spread across knees of every quantity it sets.

Out of bounds: hyperparameter sweeps; changes to existing loss weights; hand-set distances,
angles or regions without a cited measurement; crop widths, image size, slice counts; finer
tokens; more than one first look, or a first look at the prediction multiplier; anything looked
up by study identity at validation or prediction; trained models in the loader; fewer epochs;
weight averaging; reflection-probability changes; the class-balance exponent; gold-study
weighting; multi-seed or confirmation runs; other backbones; choosing anything on the campaign
fold's validation labels; an offset pattern or an average over displaced crops at prediction.

Closed by earlier runs ([prior work](prior-work.md)): finer tokens; tight region tables; more
anchors on the frame; ACL through the height of its region; the old recipe's slice-position and
in-image codes on the own-channel tokens; a routed half-knee auxiliary loss; the side head on
undetached tokens; the old tear and OA detectors pooled over the regions; a crop shift learned
from the labels, per series type, per study or per view; random crop offsets in training; a
shared ligament reader; Synovitis on Effusion's attention; Contusion on the edema head; an angle
of the upper band learned from the labels.

## Guardrails

**Before a run trains**, the developer runs
`rtk proxy uv run --locked python -m scripts.tools.check_side_routing --config configs/training.yaml`
in the candidate and reports its JSON. It applies to every seed of every topic. A candidate that
fails does not run: the developer repairs it or returns it blocked. The script is not edited by
a seed.

| Guardrail | Check |
|---|---|
| Shared classifiers | The two meniscus targets read one channel, the three OA targets another |
| Blind to the compartment | Flipping the knee side swaps the medial and lateral logits |
| Position-free shared tokens | Moving every position leaves the shared classifiers' tokens identical |
| Geometry is the only assignment | Without routing and prior, targets of one channel have equal logits |
| Routing | The meniscus pair's regions do not overlap and every weight lies between 0 and 1 |
| Zero start | The shared rotation and the prior start at zero |
| Readers leave the backbone alone | The side loss has no gradient on the projection or the backbone |
| Loss terms | Only the target loss, the side loss, the report-attribute loss, the anchor penalty and `shift_locator_loss` |
| Mirror | A mirrored pass keeps every token's anatomical position |

A topic 2 candidate also confirms on one batch that with the frame's angle and shift at zero its
crops equal the incumbent's prepared images, largest absolute difference at most 1 grey level of
255, in training and in evaluation mode, and that after the re-cut the frame's axis lies along
the crop's rows or columns (the residual angle, in degrees).

## Adoption

Fixed by the user and applied by the keeper without asking. The runtime computes its shared
verdict; the keeper adopts with
`python -m scripts.tools.autoresearch adopt frame-and-targets --experiment EXPERIMENT_ID`,
citing this section, every valid run that passed the pre-run guardrails and whose macro AUC is
**at least the incumbent's at both multipliers**:

- **×1:** the selected checkpoint's validation macro (`score` in `selected_checkpoint.json`);
- **×4:** the macro of the saved predictions (`prediction_macro_auc`).

A run level or better at one multiplier and lower at the other is not adopted. A run that is not
adopted leaves the incumbent unchanged and its results feed the next design. A topic 2 run also
reports its prediction time against the incumbent's; one above 1.3 times is not adopted, whatever
its score. No decision of the campaign waits for the user.

## Duration

The campaign is started with `start`, not under the topic loop, and has no end: after the forced
seeds the researcher keeps proposing on the three topics until the user stops it, and never
returns an empty report.

## Evaluation

The [shared evaluation](../../protocol.md#shared-evaluation-and-adoption) applies against the
campaign incumbent. Every result also reports ×1, the per-target deltas against the incumbent
with each target's sibling standard deviation ([prior work](prior-work.md#yardstick)), and the
geometry the run set or learned.
