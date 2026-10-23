# Routing-and-shifts campaign contract

## Objective

Improve the side-routed model on two topics at once:

1. **Side routing:** the division of the knee into regions and which targets read which, on the
   design of [the plan and runs](../../../docs/modeling/side-routed-heads.md).
2. **Learned shifts:** where each view's crop is cut, learned instead of set by hand, with the
   size of every crop fixed.

The base and starting incumbent is run 24 of that line, `side-routed-spot-frame-ring-sectors`
(`outputs/training/20261006-163650-side-routed-spot-frame-ring-sectors`: fold 1 of
`all_5fold_scanner_v2`, seed 1110484336, ResNet-18, ten epochs, ×4 0.925067, ×1 0.917070, 0.953
public as S118). The campaign follows both topics locally and makes no Kaggle submission.

## Scope

**Topic 1, side routing.** A seed may change:

- the knee frame: its centre, its rotation, the per-knee sizes of its regions;
- the routing: the cuts, the table from regions to targets, how a cut passes gradient;
- which targets, or which report questions, share a classifier or an attention column, and
  the regions the report targets read.

**Topic 2, learned shifts.** A seed may change:

- where each view's crop is cut: per series type or per study, as one point per study or as
  two offsets per view, and what predicts it;
- the image the loader hands over, only as far as a movable crop needs: a wider canvas at the
  same pixel spacing, from which the model cuts the crop at the view's fixed size;
- one added loss term that trains the shift, in the seeds that state it.

Fixed in both topics: the backbone and its weights, the width in millimetres and the pixel size
of every view, the number of slices per view and per study, the augmentations other than where
the affine shift is applied, the optimiser, the schedule and the epochs, the labels, the existing
loss terms and their weights, and the inference settings. The random affine shift stays and is
applied around the learned position.

## Principles

1. **Shared classifiers are blind to the compartment.** Geometry alone assigns a shared output
   to a target, and the regions of a mirrored pair are mirror images.
2. **A token is one point.** It belongs wholly to the region its centre is in; a cut is a hard
   step with a soft gradient. No token is shared across regions by overlap.
3. **The frame is inferred per knee**, and the report attributes are routed.
4. **Regions are generous.** Leaving evidence out costs and including too much is free (runs
   18, 19 and 30): a seed that narrows a region says which measurement allows it.
5. **A cut needs a reader on both sides.** A cut with targets on one side only drifts (run 26);
   one with targets on both sides stays where it starts without help (run 30).
6. **Nothing geometric is set by hand** where it can be learned or measured. A learned shift
   starts at zero, which is today's placement.
7. **A shift is computable for an unseen knee.** Nothing is looked up by study identity, and
   no trained model runs in the loader: whatever predicts a shift is part of the model.
8. **Readers leave the backbone alone** unless the seed says otherwise: whatever reads the
   side, a position or a shift target from the tokens reads them detached.

## How a seed is written

Each seed is one training run and states:

- **Evidence:** the numbers or drawings from earlier runs that motivate it, with their run.
- **Mechanism:** what changes, exactly, with the files and the formulas.
- **Replaces:** what it removes. A seed that only adds says why nothing can go.
- **Shown wrong by:** the logged value or per-target result that would contradict it.
- **Label:** evidence-backed or a guess, per change.

A seed never only measures: every seed trains, and what it needs to know it logs. Each topic 2
seed logs, per epoch and on the validation knees, the mean and the spread across knees of every
learned shift in millimetres.

Out of bounds: hyperparameter sweeps; changes to existing loss weights; hand-set distances,
angles or regions; crop widths, image size, slice counts; finer tokens; a second backbone pass
at full resolution; anything looked up by study identity; trained models in the loader; fewer
epochs; weight averaging; reflection-probability changes; the class-balance exponent; gold-study
weighting; multi-seed or confirmation runs; other backbones; choosing anything on the campaign
fold's validation labels.

Closed by earlier runs ([prior work](prior-work.md)): finer tokens; tight region tables; more
anchors on the frame; ACL through the height of its region; the old recipe's slice-position and
in-image codes on the own-channel tokens; a routed half-knee auxiliary loss; the side head on
undetached tokens; the old tear and OA detectors pooled over the regions.

## Guardrails

**Before a run trains**, the developer runs
`rtk proxy uv run --locked python -m scripts.tools.check_side_routing --config configs/training.yaml`
in the candidate and reports its JSON. It applies to every seed of both topics, because moving a
crop moves the tokens the routing reads. A candidate that fails does not run: the developer
repairs it or returns it blocked. The script is not edited by a seed.

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

A topic 2 candidate also confirms on one batch that with every learned shift at zero its crops
equal the incumbent's pixel for pixel, and reports the largest absolute difference.

## Adoption

Fixed by the user on 2026-10-06 and applied by the keeper without asking. The runtime computes
its shared verdict; the keeper adopts with
`python -m scripts.tools.autoresearch adopt routing-and-shifts --experiment EXPERIMENT_ID`,
citing this section, every valid run that passed the pre-run guardrails and whose macro AUC is
**at least the incumbent's at both multipliers**:

- **×1:** the selected checkpoint's validation macro (`score` in `selected_checkpoint.json`);
- **×4:** the macro of the saved predictions (`prediction_macro_auc`).

A run level or better at one multiplier and lower at the other is not adopted: run 29 equalled
the incumbent at ×4 (0.925098) while 0.0016 below it at ×1, with the largest ×1-to-×4 gain of
the line. A run that is not adopted leaves the incumbent unchanged and its results feed the next
design. A topic 2 run also reports its prediction time against the incumbent's; one above 1.3
times is not adopted, whatever its score. No decision of the campaign waits for the user.

## Duration

The campaign is started with `start`, not under the topic loop, and has no end: after the forced
seeds the researcher keeps proposing on both topics until the user stops it, and never returns
an empty report.

## Evaluation

The [shared evaluation](../../protocol.md#shared-evaluation-and-adoption) applies against the
campaign incumbent. Every result also reports ×1, the per-target deltas against the incumbent
with each target's sibling standard deviation ([prior work](prior-work.md#yardstick)), and the
learned geometry.
