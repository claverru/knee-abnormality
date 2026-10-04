# Learnable-views campaign

**Closed: stopped by the user on 2026-10-03 23:40 CEST after 47 scored runs, none adopted by
rule.** Outcome and every run:
[2026-10-03 learnable-views campaign](../../../docs/validation/results/2026-10-03-learnable-views-campaign.md).
It started on 2026-10-02 21:27 CEST on Codex with the goal of improving the learnable recipe
(user, 2026-10-02): the RadImageNet ResNet-18 that learns each series type's view width and
shift, a per-view label bias and the slice budget by gradient, starting from the 120 mm centre
crop. Fourteen forced seeds each change one thing of the recipe, starting with the user's named
directions (two views on one series type, a learned knee centre, a 3D position measured from
it, a learned spread of the slices along the stack) and three changes to how the views are
learned; after them the researcher combines what acted and is free inside three
limits (the time budget, the study's slice total, no input above 224 px). Recipes with
hand-set, frozen views were not candidates, until the user made one the reference
([close](#close)). It follows the
[smart-crops campaign](../smart-crops/INDEX.md) and the
[view-optimisation build](build-notes.md).

| Read when | Document |
|---|---|
| Checking the objective, the limits, what the learnable recipe does, the seeds, budget, fixed and excluded mechanisms and evidence requirements | [Contract](contract.md) |
| Checking where the learnable recipe stands against the hand recipe at ×1, ×4 and on the public set, where the learned views end, every run of the build, the flaws already fixed and what is not built | [Prior work](prior-work.md) |
| Checking the sealed recipe, reference, budget, scope and role instructions | [Campaign definition](campaign.yaml) |
| Reading seeds 1 to 5 (two learned views on sagittal fluid, two on every type, knee-anchored views with learned offsets, the learned knee centre fit, the 3D position from the learned centre) | [Seeds](seed-ideas.yaml) |
| Reading seeds 6 to 10 (the learned slice spread, views learned on held-out studies, a longer gradient average for the views, views starting late, label bias off) | [Spread and view learning](seed-ideas-2.yaml) |
| Reading seeds 11 to 14 (normal jitter, fixed budget, start from the hand views, views frozen after five epochs) | [Single changes](seed-ideas-3.yaml) |
| Reading the user seed that keeps the 1D and 2D position embeddings and adds the 3D position from the learned centre beside them, forced after seeds 4 and 5 on 2026-10-02 | [Both encodings](position-both-ideas.yaml) |
| Reading the user seed that adds EMA (existing option, BatchNorm statistics averaged, no refresh, averaged loader state) to seed 5's recipe, lifting the weight-averaging exclusion for it only, 2026-10-02 | [EMA seed](ema-ideas.yaml) |
| Reading the user seed that puts the 3D position from the computed centre on the incumbent in place of 1D and 2D, with no learned centre and no knee anchors, 2026-10-02 | [Computed centre](computed-centre-3d-ideas.yaml) |
| Reading the follow-up to the audit of seed 5: its recipe with the centre logits in their own optimizer group at beta1 0.99, and the centre diagnostics that motivate it, 2026-10-02 | [Centre long average](centre-long-average-ideas.yaml) |
| Reading the user seed that displaces the knee centre at random in training (4 mm) as the origin of the 3D positions and the medial and anterior axes, on seed 5's recipe, 2026-10-02 | [Centre noise](centre-noise-ideas.yaml) |
| Reading why the OA detector stays dead when the 3D position replaces 1D and 2D, and the user seed that starts it first-order (zero readout, positive gain) on seed 5's recipe, 2026-10-03 | [OA detector](oa-detector-ideas.yaml) |
| Reading the keeper's seed (user's standing permission) that floors the learned slice spread at its starting value, after two spread runs lost the x1-to-x4 gain with coronal spreads narrowed to a quarter, 2026-10-03 | [Spread floor](spread-floor-ideas.yaml) |
| Reading the user seed that freezes the hand views of S93 and keeps the label bias, a user exception to the frozen-recipe exclusion, after the comparison of S93 with the 28 learned runs, 2026-10-03 | [Hand views with label bias](hand-views-label-bias-ideas.yaml) |
| Reading the user seed that starts the learned views from the hand recipe and holds them for three epochs (diagnostic), relayed from the view-building session, 2026-10-03 | [Hand start, held early](hand-start-late-ideas.yaml) |
| Reading the two view-size menu seeds: two slots per coronal and sagittal type drawing sizes from a fixed menu, uniformly (measurement) and following the label bias (candidate), with the keeper's choices for the menu, evaluation rule and hand-over, 2026-10-03 | [View-size menu](view-menu-ideas.yaml) |
| Reading the user seed that learns only the view shifts with the sizes fixed at the hand recipe's, 2026-10-03 | [Learned shifts](learned-shifts-ideas.yaml) |
| Reading the last user seeds on the hand-views reference: a wide second sagittal-fluid view with the slice budget learned, and the learned budget alone, 2026-10-03 | [Hand views, learned budget](hand-views-budget-ideas.yaml) |
| Reading the user seed that trains the label bias at ten times the head rate on the hand-views reference, and the bias trajectory behind it, 2026-10-03 | [Label-bias rate](label-bias-rate-ideas.yaml) |
| Reading the keeper's three bets on the hand-views reference, queued at the user's request: the learned recipe's training randomness with nothing learned, a zoomed patellofemoral axial-fluid view, and a label bias per view and slice depth, 2026-10-03 | [Keeper bets](keeper-bets-ideas.yaml) |
| Checking each decision as the runtime recorded it | [Findings](findings.md) |
| Reading how the learnable setup was built and checked, the runs before the campaign, the learned slice budget, the schema merge, what separates the hand recipe from the learned one and the review of the first 33 runs (moved from `docs/ongoing/view-optimisation.md`, which the seed files still name) | [Build notes](build-notes.md) |

## Setup

- **Branch and worktree:** `view-optimisation` in `../knee-abnormality-view-optimisation`, off
  `8ef12dc`. The build was uncommitted while the campaign ran (the runtime works from
  uncommitted source) and is committed on the branch as `b290da6`.
- **Recipe:** `configs/training.yaml`, which on this branch holds the learnable recipe at its
  starting values (the 120 mm centre crop, `data.learn_views: [spacing_mm, offset_mm, slices]`),
  because candidates train from that file. The hand recipe in force on main is kept as
  `configs/hand-views.yaml`; `configs/frozen-learned-views.yaml` is the diagnostic run of the
  reference's final views.
- **Reference:** `views-budget-shift-centre` (`outputs/training/views-budget-shift-centre`, ×4
  0.932489, ×1 0.9273, 0.947 public as S94), trained on 2026-10-02 from this worktree before the
  view-config schema was merged into the build. The user kept it as the reference in place of a
  retrain (2026-10-02). The merge renamed configuration and left the computation unchanged, so
  the reference is declared with:
  - *Config exceptions:* `data.series_types` (per-type `max_images` and per-view `learn_spacing`
    and `learn_shift` became per-view `slices`), `data.study_triplet_total` and
    `model.learn_slice_budget`.
  - *Equivalences:* `data.study_slice_total: 26` and
    `data.learn_views: [spacing_mm, offset_mm, slices]`.
  - *Identity exception:* `preprocessing_identity` (the hashed sources changed).
  - *Proof:* on six training and six validation studies, for the hand recipe and the learnable
    one, the code before and after the merge selects the same slices and gives identical images,
    features, outputs and budget, spacing and shift gradients (difference 0.0;
    [notes](build-notes.md#merge-of-the-view-config-schema-2026-10-02)).
- **Budget:** range 0.75–1.25 of the reference's 1,096 s training time (822–1,370 s; user,
  2026-10-02). Epochs, batch size and the study's slice total are not editable.
- **Seed order:** forced: the two-view seeds, the two knee-centre seeds, the 3D position, the
  slice spread, the three view-learning seeds, then the single-change seeds.
- **Endless research** (user, 2026-10-02): `rotate_after: 1000000`, and the researcher is told
  never to return an empty refill and to open a new family when one closes. Only the user stops
  the campaign. The runtime leaves a campaign idle after an empty refill, so the keeper watches
  for it.
- **Links:** `outputs` and `data` in the worktree are links to the main checkout's.

## Settled before the start

- **Kaggle path:** the reference predicted through the Kaggle worker path in FP16 scores
  0.932472 against 0.932489 locally, so S94's 0.947 is the model and not the prediction path.
- **Config merge:** the view-config worktree's schema is merged into the build (per-view
  `slices`, required axis fields, `data.study_slice_total`, each view selecting its own slices),
  with `data.learn_views` as the one switch between a frozen and a learned recipe. The name that
  replaces main's root `data.spacing_mm` stays as in this build (`selection_radius_mm`,
  `selection_resolution_floor_mm`, `canvas_*`); the user deferred it.
- **×1:** the shared adoption rule reads ×4 only; the contract asks for ×1 beside it as
  evidence.

## Close

- **Reference changed by the user (2026-10-03).** `hand-views-frozen-with-label-bias`, a user
  seed and an exception to the frozen-recipe exclusion, scored ×4 0.933307 against 0.932489
  (+0.0008, 80% [−0.0007, +0.0023]): a reject by rule, adopted as the reference by the user's
  decision. It was trained by hand from the campaign's frozen source after a WSL restart and
  registered in the state (`manual-run/` in its experiment folder). The ten runs after it are
  judged against it.
- **Adoption rule changed during the campaign (user, 2026-10-02):** the 80% interval alone, the
  0.001 gain condition dropped ([protocol](../../protocol.md#shared-evaluation-and-adoption)).
  No decision would differ under the older rule.
- **Harness repairs made for this campaign**, on the branch only because main's model has no
  learned loader state: the validation-determinism gate compares selections between passes with
  the same `loader_state` (a learned budget may change the selection between checkpoints), and
  the exporter keeps `loader_state` in the checkpoint.
- **State at the stop:** 46 decided, one scored and undecided
  (`ordered-sagittal-routing-with-shared-affine-exploration`, −0.0025), one built and not run
  (`hand-views-routed-shift-learning`), one queued (`view-size-menu-with-deployment-replay`).
  The runtime state is `outputs/autoresearch/learnable-views/state.json`; resuming needs the
  branch's worktree.
- **Checkpoints:** the selected checkpoint of ten runs is kept, every other one deleted on
  2026-10-04
  ([pruning record](../../../docs/architecture/configs-and-experiment-artifacts.md#checkpoint-pruning)).
