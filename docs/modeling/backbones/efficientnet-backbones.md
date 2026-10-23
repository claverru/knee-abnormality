# EfficientNet backbones

**EfficientNet-B0 and B3 were tried against ReXNet-200 in September 2026 and dropped with the
ReXNet era.** The five-B0 bundle scored 0.940 public and was the efficiency entry of that time,
while five B0 plus five ReXNet models with mirror TTA scored 0.943. No EfficientNet was run on the
class-balanced BCE recipe or later, and the EfficientNet mappings were removed from the code on
2026-10-14. The final families are in the [final solution](../../final-solution.md).

## Evidence

All comparisons below used ordinary BCE.

- [Backbone comparison](../../validation/results/backbones/2026-09-18-efficientnet-backbone-results.md): B0
  used about 1,010 s and 9.4 GiB against ReXNet's 1,541 s and 19.6 GiB at matched epochs, with
  4.01M versus 13.81M parameters. B3's lower MAC count did not give lower wall time.
- [B0 queues](../../validation/results/campaigns-and-queues/2026-09-18-b0-hyperparameter-queues.md): the retained B0
  configuration was `efficientnet_b0.ra_in1k`, backbone LR 4e-4, head LR 6e-4, fifteen epochs and
  batch eight. Batch eight and sixteen took 99–101 s per epoch; matched-step results tied.
- [Fold queue](../../validation/results/validation-and-ensembles/2026-09-19-fold-ensembles-and-sigma-sweep.md): fifteen-epoch
  B0 took 25.7 minutes against 26.2 for ten-epoch ReXNet, so the two recipes had similar
  training cost. Their mean per-fold AUCs were 0.87702 and 0.87722; the ensemble gained 0.0061
  pooled OOF over ReXNet. Combining fifteen B0 epochs with both LRs doubled lost 0.0033 against
  fifteen epochs alone. ReXNet also ran fifteen epochs, gaining 0.0025 in one inconclusive run.
- [Submissions](../../validation/results/validation-and-ensembles/2026-09-19-ensemble-submissions.md): single-fold B0
  scored 0.934 versus ReXNet's 0.936. Measured B0 forward time was 0.77× ReXNet per image;
  loading and member count limited the total saving. Submitted-to-scored times include queueing
  and are not official hidden-test execution times.
- **EfficientNet-B3, 2026-09-27:** the first queue put B3 at −0.0008 ×4 against ReXNet
  (`backbone_lr` 4.5e-4), and the pair at +0.0029 locally; the pair scored 0.946 public (bundle
  `5bf80349`, titled S70) against 0.947 for ReXNet alone (S68)
  ([result](../../validation/results/campaigns-and-queues/2026-09-27-b3-overnight-queue.md)). The
  second round of seven learning-rate and schedule runs found one gain, a two-epoch warmup: ×4
  0.905638, +0.0012 over the B3 control and +0.0004 over ReXNet (95% [−0.0029, +0.0040])
  ([second round](../../validation/results/campaigns-and-queues/2026-09-27-b3-overnight-queue.md#second-round-learning-rates-and-schedule)).

The mapping code for `efficientnet_b3.ra2_in1k`, `efficientnet_b0.ra_in1k` and
`efficientnet_b0.ra4_e3600_r224_in1k` is described in
[swapping a backbone](backbone-swaps.md#code-requirements) as it stood before its removal.

## Never tested

- B0 against ReXNet under class-balanced BCE with matched controls and recorded budgets.
- An isolated B0 head-LR increase. ReXNet's balanced-BCE 0.0012/0.0024 arms both trailed their
  controls ([results](../../validation/results/campaigns-and-queues/2026-09-20-overnight-hyperparameter-results.md)),
  so the older joint-LR gain was not assumed to transfer to fifteen-epoch B0 or to balanced BCE.
- B0 EMA at decay 0.999. Its 0.9999 attempt was void because it retained too much
  initialization; ReXNet's EMA/SWA comparisons are in
  [the 2026-09-17 results](../../validation/results/campaigns-and-queues/2026-09-17-overnight-results.md#weight-averaging).
