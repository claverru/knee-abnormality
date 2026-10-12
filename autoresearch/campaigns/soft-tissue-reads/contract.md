# Soft-tissue reads campaign contract

Written 2026-10-10 with the user. Started 2026-10-11 00:10 CEST with Codex.

## Objective

Raise the macro AUC of `rn18-soft-tissue-added-read-bone-routes` by exploiting the soft-tissue
masks better, on the reference's own line: a seven-class map of the menisci and cartilage taught
from the masks, and a read of the third-stage cells routed by that map (B) beside the
fourth-stage read (A). The measurements behind every seed are in [prior work](prior-work.md).

The reference and first incumbent is `outputs/training/rn18-soft-tissue-added-read-bone-routes`
(trained by the user on 2026-10-10: fold 1 of `all_5fold_scanner_v2`, seed 1110484336,
ResNet-18, ten epochs; ×4 0.928909, ×1 0.918680, 1,599 s of training).

Five forced seeds, in order, then the researcher without end:

1. `residual-cell-read-on-a-detached-fourth-stage`: A is trained alone on the label; B is
   trained on `A.detach() + B`; the wide regions of run 4 come back. The user's design.
2. `oa-cartilage-and-bone-zone-pooled-apart`: the OA labels read their cartilage cells and their
   bone-zone cells in two pools.
3. `cell-position-inside-the-structure`: each cell knows where it lies relative to its own
   structure's centre.
4. `pf-oa-own-cell-reader`: PF OA stops sharing B's reader with Medial and Lateral OA.
5. `soft-tissue-map-on-the-28-by-28-grid`: the map is predicted and taught on the second-stage
   grid and averaged down to the cells for routing.

## Settled

- **B corrects A** (user, 2026-10-10). In seed 1, A takes the label loss alone on its own logit
  for all twelve labels and never sees B in training; B takes the loss on `A.detach() + B` for
  the five labels. B's loss still reaches backbone stages 1 to 3 through the cell features: not
  detached, by the user's decision. A loss on B's own logit is out (run 3).
- **Prediction is A + B.**
- **The routing is read from detached maps.** The label loss never reaches a map head through a
  share, a centre, a region or a weight.
- **A label reads its own side and its own compartment** in B. A already carries the other
  compartments.
- **Bone routing of Contusion and Fracture stays as it is** and is not a subject of the
  campaign.
- **The seven-class targets** (`soft-tissue-nnunet-knee-dess265`) and the bone targets are the
  only mask sets. No new mask set is built during the campaign.
- **Granularity of the fallbacks:** a label is unrouted per study (total weight below one cell
  in that study); a compartment falls back to the bone zone per study.

## Principles

1. **B is worth what A lacks.** A seed says which fine local detail B gains; it does not hand B
   evidence A already has.
2. **Structures are used as shares.** No class by argmax, no threshold without a measurement.
3. **A number set by hand comes from a cited measurement**, taken on the masks or on saved
   runs, never from a sweep on the validation labels.
4. **Everything is computable for an unseen knee** from its own images through the model. No
   segmenter and no prepared target at validation or prediction.
5. **One mechanism per run.** A seed changes one thing and states what it replaces. A superseded
   behaviour is removed, not kept behind an option.
6. **A seed can be proved active** by validation-only metrics it names, at most six. A seed
   whose mechanism acts only in training (targets, losses or slices that validation does not
   have) may instead name a training metric the recipe already logs, such as `meniscus_tokens`;
   it adds no new diagnostic to the training step (the user's keeper, 2026-10-11).

## Bundles (the user, 2026-10-11 07:53 CEST)

After sixteen runs without an adoption the user lifted principle 5 for combination seeds: "Look
experiments wins and loses per label to propose more seeds, maybe there's a good combination of
experiments. Check each experiment properties. Don't isolate changes and aim for the best
solution with the information in hand."

- A seed or a researcher idea may combine several mechanisms that already ran in this campaign,
  chosen by their per-label results and their diagnostics in the [run notes](run-notes.md). It
  names each run it takes from and starts from those candidates' reviewed sources.
- A bundle is judged as one run under the same adoption rule. It is not followed by runs that
  take its parts out one at a time.
- Everything else in this contract holds, including the out-of-bounds list and the cost section.

## Wider scope (the user, 2026-10-11 09:40 CEST)

After nineteen runs without an adoption and the cross-run analysis in the
[run notes](run-notes.md), the user widened the campaign ("okay let's do them"; "drop the seeds
without potential, and add your seeds and the other agent seeds that have it"). What the
analysis measured: the runs behave as retrainings of one model (×4 SD 0.0004), B adds nothing
measurable on its five labels, the map as a side task shows on the other seven, the map is
almost never taught on sagittal fluid (20 of 265 mask knees have that series), and ×4 over ×1
is +0.010 macro from slice coverage.

The objective is now: raise the macro AUC of the reference by any better use of the masks and
of the training bag. Allowed in addition to the rest of this contract:

- **Training slices.** A run may encode extra slices per study in training without gradient
  and let them enter the pools. Validation and prediction stay as they are.
- **The model's own map as a target across planes.** A detached map of one plane may teach the
  map of another plane of the same study in patient space. No segmenter, no prepared target.
- **B may be replaced.** "Prediction is A + B" is no longer settled: a run may fold the cell
  features into A's bag, or drop B and keep the map as a side task.
- **Changes outside the five labels** when the mechanism is not label-specific.
- **Pseudo-volumes of the map.** The folder `pseudo_volumes` inside the prepared mask set (the
  reference's own map on every fold-1 training study, fused across views in patient
  coordinates, built by the keeper on 2026-10-11 at the user's request) may teach the map.
- **One detail view.** The seed `sagittal-fluid-detail-view-on-the-same-slices` may add a view
  row for sagittal fluid that shares the first view's slices (the user, 2026-10-11).

Still out: other views, crops, canvases, fields, image size, the backbone, fewer epochs, new
prepared series, new mask sets from a segmenter, weight averaging, test-time changes, and the
rest of the out-of-bounds list.

**How a result is read from now on.** One run resolves about 0.0004 macro at ×4 and 0.003 to
0.007 on a label with under 100 positives. A per-label effect is read on the ×1 AUC averaged
over epochs 6 to 10 against the incumbent at the same epochs, together with ×4, and against the
level and spread of the earlier runs in the run notes. Adoption is unchanged.

## How a seed is written

One training run, stating: **evidence** (numbers with their run or section of prior work),
**mechanism** (what changes, with files and formulas), **replaces**, **shown wrong by**, and a
**label** per claim (evidence-backed or a guess). A seed that reuses a mechanism already
reviewed names that candidate's source instead of describing it again.

## Out of bounds

Building or adding a mask set, running a segmenter, masks predicted by a model as targets;
anything looked up by study identity at validation or prediction; trained models in the loader;
crops, views, canvases, fields, image size (224 px) and slice counts; the backbone; fewer
epochs; a loss on B's own logit; letting the label loss move a map, a centre or a region; a
correction constant measured by hand; changes to the bone routing or to the labels outside the
five (a seed may extend B to another label only with evidence in prior work); hyperparameter
sweeps; weight averaging; reflection-probability changes; gold-study weighting; multi-seed,
confirmation or control runs; choosing anything on the campaign fold's validation labels, except
the parts of a bundle (see Bundles).

## Cost

A run adds no training time its mechanism does not need: no new diagnostics in the training
step, no operation that forces the GPU to synchronise, no second evaluation of the maps, the
routing or the tokens, no extra forward pass. Each seed states its expected training time
against the incumbent. Seeds 1 to 4 stay within 1.03 of the incumbent; seed 5 within 1.15.

**The budget is 0.75 to 1.25 of the reference's training time** (1,199 to 1,998 s). Since
2026-10-11 the runtime's own ceiling is 2,558 s of training and 3,000 s per experiment, raised on
the user's instruction because another session shares the machine and slows runs; this is not a
larger budget. The keeper reads a run above 1,998 s by its epoch times and counts it over budget
only when its own cost put it there. A reviewer does not hold the difference between the two
figures against a candidate. The ceiling is fixed against the reference while each adopted seed
makes every later run slower, so it is not room to spend.

## Duration

Started with `start`; no end. After the forced seeds the researcher keeps proposing inside this
contract until the user stops it, and never returns an empty report.

## Adoption

By the runtime alone, under the
[shared rule](../../protocol.md#adoption-policy): a run is adopted when its macro AUC is
strictly above the incumbent's at both ×1 and ×4. There are no exceptions per seed and no
manual adoption, so the GPU never waits for a decision. The keeper repairs and requeues; it
does not adopt.

A run that trains outside the budget is invalid whatever its score.

## Evaluation

Every run is read against the incumbent at ×4 and ×1, per label from largest gain to largest
loss, with: A alone, B alone and A + B for the five labels where the run logs them; the map's
overlap per class; B's effective open cells, routed share and mean absolute logit per label; and
the training and prediction seconds. The mask scanner's 52 validation studies are read apart
from the other 822.
