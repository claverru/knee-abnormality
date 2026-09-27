# EfficientNet backbones

**Open: re-measure B0 against ReXNet-200 under class-balanced BCE.** B0 is already deployed:
the five-B0 bundle scored 0.940 publicly and is the efficiency entry, while five B0 plus five
ReXNet models with mirror TTA scored 0.943. The pending question is how that tradeoff changes
under the current training loss.

## Established evidence

All comparisons below used ordinary BCE. They motivate the next controlled comparison but do not
measure the gap under balanced BCE.

- [Backbone comparison](../validation/results/2026-09-18-efficientnet-backbone-results.md): B0
  used about 1,010 s and 9.4 GiB against ReXNet's 1,541 s and 19.6 GiB at matched epochs, with
  4.01M versus 13.81M parameters. B3's lower MAC count did not give lower wall time.
- [B0 queues](../validation/results/2026-09-18-b0-hyperparameter-queues.md): the retained B0
  configuration is `efficientnet_b0.ra_in1k`, backbone LR 4e-4, head LR 6e-4, fifteen epochs and
  batch eight. Batch eight and sixteen took 99–101 s per epoch; matched-step results tied.
- [Fold queue](../validation/results/2026-09-19-fold-ensembles-and-sigma-sweep.md): fifteen-epoch
  B0 took 25.7 minutes against 26.2 for ten-epoch ReXNet, so the retained recipes have similar
  training cost. Their mean per-fold AUCs were 0.87702 and 0.87722; the ensemble gained 0.0061
  pooled OOF over ReXNet. Combining fifteen B0 epochs with both LRs doubled lost 0.0033 against
  fifteen epochs alone. ReXNet also ran fifteen epochs, gaining 0.0025 in one inconclusive run.
- [Submissions](../inference/submissions/2026-09-19-ensemble-submissions.md): single-fold B0
  scored 0.934 versus ReXNet's 0.936. Measured B0 forward time was 0.77× ReXNet per image;
  loading and member count limit the total saving. Submitted-to-scored times include queueing
  and are not official hidden-test execution times.

Backbone mapping is already implemented; see [swapping a backbone](../modeling/backbone-swaps.md#code-requirements).

## Open

- **EfficientNet-B3 on the current recipe, 2026-09-27:** the first queue put B3 at −0.0008 ×4
  against ReXNet (`backbone_lr` 4.5e-4), and the pair at +0.0029 locally; the pair scored 0.946 public (bundle `5bf80349`, titled S70) against 0.947 for ReXNet alone (S68)
  ([result](../validation/results/2026-09-27-b3-overnight-queue.md)). The second round of seven
  learning-rate and schedule runs found one gain, a two-epoch warmup: ×4 0.905638, +0.0012 over the
  B3 control and +0.0004 over ReXNet (95% [−0.0029, +0.0040])
  ([second round](../validation/results/2026-09-27-b3-overnight-queue.md#second-round-learning-rates-and-schedule)).
  The recipe has since moved to the scanner confusion head on fold 4, so a B3 follow-up needs a
  control there.
- Compare B0 and ReXNet with the current balanced-BCE recipe, matched controls and recorded
  budgets. Re-price the score/runtime tradeoff before adopting further B0 settings.
- An isolated B0 head-LR increase remains untested. ReXNet's balanced-BCE 0.0012/0.0024 arms
  both trailed their controls
  ([results](../validation/results/2026-09-20-overnight-hyperparameter-results.md)). Do not assume
  the older joint-LR gain transfers to fifteen-epoch B0 or to balanced BCE.
- B0 EMA at decay 0.999 remains untested. Its 0.9999 attempt was void because it retained too
  much initialization. ReXNet's EMA/SWA comparisons are already measured; recipe-wide adoption
  and follow-ups stay in [endgame](../backlog/endgame.md#weight-averaging).
- A second fold/seed ensemble using balanced-BCE members remains in
  [endgame](../backlog/endgame.md#ensembling). The ordinary-BCE ensemble round is complete.
