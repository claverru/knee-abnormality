# Revisits campaign contract

## Objective

Re-test, on the RadImageNet ResNet-18 recipe, ideas whose earlier verdict no longer holds, then
tune the hyperparameters ResNet-18 inherited from ReXNet-200. The nine forced seeds come from
re-scoring every past comparison on `rebuild_state_v2` and from untried neighbours of those ideas
([prior work](prior-work.md)): plane specialization on the shared backbone (dense per-plane
readouts, plane/contrast readouts, plane/contrast feature adapters, plane-conditioned
BatchNorm), removal of adoptions that lost their support (DSU, the anatomical attention bias,
bilateral smoothing, resolution degradation) and stochastic depth. After the seeds, the researcher
searches hyperparameters until the user stops the campaign (`rotate_after: 1000`, user,
2026-09-30: low-potential runs are better than an idle GPU).

The reference and starting incumbent is `rn18-l2sp-no-confusion`
(`outputs/training/rn18-l2sp-no-confusion`, seed 1533697213, fold 0 of `all_5fold_scanner_v2`,
×4 0.928062, 0.943 public as S86), the `configs/training.yaml` recipe since 2026-09-30.

## Evaluation

The [shared evaluation and adoption rule](../../protocol.md#shared-evaluation-and-adoption)
applies unchanged against the campaign incumbent.

## Seeds

Forced, in this order (evidence first, user, 2026-09-30): dense per-plane readouts, plane/contrast
readouts, no DSU, no anatomical bias, no bilateral smoothing, no resolution degradation,
plane/contrast feature adapters, plane-conditioned BatchNorm, stochastic depth. Each applies to
the incumbent at its turn, so an accepted seed carries into the later ones. The one exception:
plane/contrast readouts replace dense per-plane readouts if those were accepted, since both
rebuild the same readout.

## Budget

Wall-clock range: each candidate's training time must lie between 50% and 165% of the
reference's 1,222 s, that is 611–2,016 s. The upper bound was 150% (1,833 s) until the user raised
it on 2026-10-01 so the ResNet-50 could train ten epochs (decision record under
`outputs/autoresearch/revisits/decisions/`). Epochs, batch size and gradient accumulation may change
within it; the studies seen per epoch, triplets per study and images per series type may not.

## Fixed

Split, fold, evaluation labels (`rebuild_state_v2`), class-balanced BCE and its exponent 0.5,
image size 224, crop and sampling geometry, triplets per study (26), the evaluation and
prediction multipliers.

## Permitted

- **Seeds:** exactly the nine mechanisms above, each under a new `model` option where it needs
  code.
- **Hyperparameters after the seeds:** backbone and head learning rates, weight decay, the L2-SP
  strengths, warmup, schedule floor, epochs, batch size with scaled rates, gradient clipping, the
  half-knee and report-attribute loss weights, augmentation probabilities and strengths, and the
  options the seeds add. One knob and one value per run, direction from the curves.

## User seeds added during the run

EMA at decay 0.999 and 0.9995 (`optimizer.weight_averaging`, [EMA seeds](ema-ideas.yaml)), forced
by the user on 2026-09-30. They lift the weight-averaging exclusion below for these two ideas only;
the researcher still proposes no weight averaging. Both seeds ran over the time budget on their
BatchNorm refresh; their retakes ([EMA subset refresh](ema-subset-refresh-ideas.yaml), 2026-10-01)
refresh on a fixed quarter of the training studies and carry the same exception. The refresh was a
mistake in the seed text: the existing EMA option already averages BatchNorm statistics with the
weights. On the user's instruction (2026-10-01) EMA moved to that path, configuration only
([EMA buffer average](ema-timm-ideas.yaml)); the 0.9995 refresh retake was withdrawn before launch.

ResNet-10t on the incumbent recipe, at backbone LR 5e-5 and at the weight-RMS-scaled 6e-5
([ResNet-10t seeds](resnet10t-ideas.yaml)), and the Refined-RadImageNet ResNet-50 at 5e-5 and at
the weight-RMS-scaled 5.5e-5, ten epochs each ([Refined ResNet-50 seed](refined-resnet50-ideas.yaml),
[scaled LR](refined-resnet50-scaled-ideas.yaml)), forced by the user on 2026-10-01:
the only backbone changes in this campaign. They are measurements only: the user does not adopt
a ResNet-50, whose training time is too high (2026-10-01). If one passes the adoption rule, the
campaign keeper restores the ResNet-18 incumbent by manual adoption. The original-RadImageNet ResNet-50 seeds
([ResNet-50 seeds](resnet50-ideas.yaml)) were withdrawn before launch in its favour.

One backbone per plane on the incumbent recipe, ResNet-18 and ResNet-10t, each with the joint pool
and plane-contrast readouts (A) and with dense per-plane readouts and learned fusion (B)
([Plane backbones](plane-backbone-ideas.yaml)), forced by the user on 2026-10-01. Like the ResNet-50
runs, they are measurements only: if one passes, the campaign keeper restores the ResNet-18
incumbent by manual adoption.

## Excluded

Weight averaging (SWA, EMA; except the two user seeds above), image sizes above 224, more triplets or images per study,
ensembling and test-time augmentation (endgame); the class-balance exponent; sampling-geometry
sweeps; label-source changes; gold-study weighting; multi-seed or confirmation runs; ideas
already measured on ResNet-18 at the same value ([prior work](prior-work.md#measured-on-resnet-18)).

## Evidence requirements

- Every candidate logs its own mechanism scalar per epoch next to the per-stage drift and
  gradient columns the Lightning module already writes.
- The researcher starts each refill from the per-epoch curves (×1 validation, training loss,
  per-label AUC, per-stage drift and gradients) of every decided run and the reference, and names
  the curve behaviour each proposal targets.
