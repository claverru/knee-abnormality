# Side-routing campaign contract

## Objective

Improve the side-routed design ([plan and runs](../../../docs/ongoing/side-routed-heads.md)):
one shared classifier for the two meniscus targets and one for the three OA targets, with the
knee's geometry deciding which target a token's evidence counts for. Two submissions of this
design scored 0.952 (S112), 0.953 (S113) and 0.952 (S114) public against 0.950 for the recipe it replaces on
the same fold, while scoring level with it locally. The campaign follows this line locally and
makes no Kaggle submission.

The base and starting incumbent is `side-routed-run5`
(`outputs/training/20261005-205251-side-routed-run5`: fold 1 of `all_5fold_scanner_v2`, seed
1110484336, ResNet-18, ten epochs, ×4 0.924476, ×1 0.915823), the
[fifth build](../../../docs/ongoing/side-routed-heads.md#fifth-build-2026-10-05). What the
campaign settles is carried to the ResNet-50 afterwards by the user.

## Scope

A seed may change only:

- the knee frame: its rotation and how the knee side is read;
- the routing: the cuts, the table from regions to targets, the softness of a cut;
- the position prior on the attention and the position the own-channel classifiers read;
- which targets, or which report questions, share a classifier.

Fixed: the backbone and its weights, the views and slices, every augmentation, the optimiser,
the schedule and the epochs, the labels, the loss terms and their weights, the recipe's
configuration and the inference settings.

## Principles

1. **Shared classifiers are blind to the compartment.** They read position-free tokens.
2. **Geometry is the only thing that assigns a shared output to a target.**
3. **The routing weights of a shared classifier's targets sum to one at every token.** A region
   one target gains is a region another loses, which is what holds a cut in place. A boundary
   with a target on one side and nothing on the other can drift until it means nothing.
4. **Nothing geometric is set by hand.** A learned value starts at zero; a measured one comes
   from the headers. The frame's origin is the fitted centre, with no offset.
5. **A reader does not train the backbone.** Whatever reads the side or a position from the
   tokens reads them detached.
6. **No auxiliary loss is added.**
7. **Geometry is looked at before it is built on.** A seed that moves a cut or a region cites a
   census or a drawing on studies.

## How a seed is written

Each seed is the best full design step from everything known, not one isolated tweak. It
states:

- **Evidence:** the numbers or images from earlier runs that motivate it, with their run.
- **Mechanism:** what changes in the model, exactly, with the files and the formulas.
- **Replaces:** what it removes. A seed that only adds says why nothing can go.
- **Shown wrong by:** the measurement in the run report that would contradict it.
- **Label:** evidence-backed or a guess, per change.

Out of bounds: hyperparameter sweeps; loss weights; auxiliary losses; hand-set distances,
angles or regions; learned geometry per study or per series without a measured spread above
half a token; anything that sends a reader's gradient into the backbone; an offset of the
frame's origin; fewer epochs; weight averaging; reflection-probability changes; the
class-balance exponent; gold-study weighting; trained models in the loader; multi-seed or
confirmation runs; other backbones.

Closed by the five runs and the measurements of 2026-10-05 ([prior work](prior-work.md)):

| Idea | Result |
|---|---|
| A per-study midline landmark | Learned one constant, about 2 mm |
| A per-study front boundary | PF OA improved while it moved from 27 to 12 mm |
| A learned lateral or front offset of the origin | 2 to 3 mm, a quarter of a token; removed with the fifth build |
| A position code on the shared classifiers' tokens | Ended at a tenth of the token's norm |
| No position on the own-channel classifiers' tokens | Fracture and Baker's below all 12 sibling runs |
| A routed half-knee auxiliary loss | MCL −0.0128 |
| The side head on undetached tokens | Trained the backbone; ACL and Synovitis fell |
| A linear side head on pooled detached tokens | 96.5%, against 99.8% for the per-token network |
| Per-series motion correction | 99% of same-plane pairs move under 2.7 mm |
| A per-study vertical crop shift | The joint line varies by about ±5 mm, under half a token |

## Guardrails

**Before a run trains**, the developer runs
`rtk proxy uv run --locked python -m scripts.tools.check_side_routing --config configs/training.yaml`
in the candidate and reports its JSON. A candidate that fails does not run: the developer
repairs it or returns it blocked. The script is not edited by a seed.

| Guardrail | Check |
|---|---|
| Shared classifiers | The two meniscus targets read one channel, the three OA targets another, the other seven one each |
| Blind to the compartment | With a random prior, flipping the knee side swaps the medial and lateral logits exactly |
| Position-free shared tokens | Moving every position leaves the shared classifiers' tokens identical |
| Geometry is the only assignment | Without routing and prior, targets of one channel have equal logits |
| Routing partitions every token | The routing weights of each shared channel's targets sum to one |
| Zero start | The frame starts unrotated and the prior at zero |
| Readers leave the backbone alone | The side loss has no gradient on the projection or the backbone |
| Loss terms | Only the target loss, the side loss, the report-attribute loss and the anchor penalty |
| Mirror | A mirrored pass keeps every token's anatomical position |

**After a run**, `python -m scripts.tools.report_side_routing` writes the table the next design
is written from, the flags below and `joins_lineage`:

| Flag | Raised when |
|---|---|
| Macro | The ×4 macro AUC is below the incumbent's, by any amount |
| Routed sum | The five routed targets, summed, are below the best of the lineage by more than twice the sibling runs' standard deviation of that sum (0.009) |
| Localisation | A routed target separates its own compartment from the others worse than the best of the lineage by more than twice the sibling runs' standard deviation |
| Own targets | An own-channel target is below every sibling run and the incumbent's is not |
| Side head | Validation accuracy below 99% |
| Geometry | A width within 2% of a limit, or an angle within 2% of its bound |

**The best of the lineage is a set of marks that only rise.** `--lineage` lists the fourth run
(`outputs/training/20261005-202001-side-routed-run4`, the best public score, 0.953), the fifth
run and every run adopted since; the mark for the routed sum and for each localisation value
is the highest any of them reached. A run is compared with those marks, not with the run
before it, so permitted losses cannot add up over several steps.

## Adoption

Fixed by the user on 2026-10-05 and applied by the keeper without asking. The campaign is a
lineage. The runtime computes its shared verdict, and the keeper adopts with
`python -m scripts.tools.autoresearch adopt side-routing --experiment EXPERIMENT_ID`, citing this
section, every valid run that passed the pre-run guardrails and whose report says
`joins_lineage: true`: no flag is raised. That is:

- **the macro never falls:** each new incumbent scores at least as high at ×4 as the one before;
- **the shared classifiers never fall below their best** by more than run-to-run noise, counted
  from the highest value any run of the lineage reached;
- **nothing else breaks:** no new own-channel target below every sibling run, a working side
  head, learned geometry inside its range.

A run that raises a flag stays out of the lineage and the incumbent does not change; its report
still feeds the next design. The fifth run, the starting incumbent, is itself below the fourth
run's mark on one value (Medial Meniscus localisation, −0.0034 against a tolerance of 0.0026),
so the first adopted run has to recover it. No decision of the campaign waits for the user.

## Duration

The campaign has no end and does not rotate to another topic: it is started with `start`, not
under the topic loop, and the researcher keeps proposing until the user stops it. The researcher
never returns an empty report; when the listed questions are answered it works on the weakest
number of the latest run reports.

## Evaluation

The [shared evaluation](../../protocol.md#shared-evaluation-and-adoption) applies against the
campaign incumbent. Every result also reports ×1, the per-target deltas and the run report.
