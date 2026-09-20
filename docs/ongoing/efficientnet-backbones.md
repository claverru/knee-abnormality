# EfficientNet backbones

**Open, and pursued for the efficiency prize.** EfficientNet-B0 trains in about 1010 s at 9.4 GiB
against ReXNet-200's 1541 s at 19.6 GiB, on a quarter of the parameters (4.01M against 13.81M) and
a quarter of the multiply-adds. The reason to keep tuning it is that cost, not its accuracy: b0 is
at 0.885263 against 0.888476 for the matched ReXNet-200 control, a −0.0032 gap already at the edge
of the ~0.003 noise floor. Close that gap and the cheap backbone becomes usable; the efficiency
leaderboard combines score and execution time
([requirements](../inference/submission-requirements-and-output-contract.md)).

**Every number on this page predates class-balanced BCE.** The b0/ReXNet comparison and both
hyperparameter queues ran on 2026-09-18 under ordinary BCE; the loss adopted on 2026-09-19 moves
fold 0 by about 0.008, more than twice the gap this line is trying to close
([state](state.md)). The same rule the overnight queue applies to its own history applies here:
none of these deltas is a prior for a run on the current recipe, and the −0.0032 gap has to be
re-measured before the efficiency case can be priced.

Measured evidence lives in two records, not here:

- [EfficientNet backbone results](../validation/results/2026-09-18-efficientnet-backbone-results.md)
  — the B3/B0 comparison, the underfitting diagnosis, the per-label anatomy pattern and the
  +0.0043 three-model ensemble gain.
- [b0 hyperparameter queues](../validation/results/2026-09-18-b0-hyperparameter-queues.md) — the
  batch-size-8 and batch-size-16 queues with seeds, wall times and per-run deltas.

The mapping code and three registry entries are in `modeling/spatial_tokens.py`, so returning to an
EfficientNet is a one-line config change
([swapping a backbone](../modeling/backbone-swaps.md#code-requirements)).

## Where it stands

- **Best b0 config:** `efficientnet_b0.ra_in1k`, `backbone_lr: 4e-4`, `max_epochs: 15`, batch
  size 8. More optimizer steps is the only axis that has moved it, consistent with the family
  underfitting rather than overfitting here.
- **Batch size is settled as useless.** Epoch time is 99–101 s at both 8 and 16, so `batch_size` is
  a pure optimizer knob with no throughput benefit; at matched steps the two tie.
- **The efficiency case is real but not yet bankable.** The training saving is measured, but
  inference is what the prize scores, and the [loading profile](loading-speedup.md) shows the
  loader and not the two T4s currently sets inference throughput. A cheaper backbone buys little
  until the loading work lands, so the ordering is loading first, then re-price b0. Note also that
  multiply-add savings did not convert to wall time for B3, which was *slower* than ReXNet-200
  despite 37% fewer MACs — only b0's much larger cut showed up as real time.

## Open

- **`head_lr` has never been raised.** It has been 6e-4 in every experiment in this project, and
  the only prior test moved it *down* (−0.0017 on ReXNet). The batch-size-16 gains scaled it
  alongside `backbone_lr` and cannot separate the two, so raising `head_lr` alone at batch size 8
  is the direct next run.
- **The learning-rate response has not turned over.** The largest rate tried was also the best
  (both groups ×2, +0.0034); where it peaks is unknown.
- **EMA is untested**, not rejected: the one attempt used decay 0.9999, which leaves 64.5% of the
  untrained initialisation after 4,380 steps and is void. Use 0.999.
- **ReXNet-200 has never run at 15 epochs on this recipe**, so the −0.0032 gap compares a 15-epoch
  b0 against a 10-epoch ReXNet. On the older recipe 15 epochs gave ReXNet +0.0023, which would
  restore most of the gap.
- **Ensembling is the strongest use found so far** (+0.0043 for ReXNet + B3 + B0), but that waits
  for the [final phase](../backlog/endgame.md).
