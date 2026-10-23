# 2026-09-18 EfficientNet backbone results

**Three EfficientNet runs lost to ReXNet-200 on fold 0: B3 at 0.879449 (−0.0090), B0 at 0.880331
(−0.0081) and B0 with a doubled backbone learning rate at 0.883206 (−0.0053). The macro loss is
confirmed by a paired bootstrap and is not a seed artifact. They nevertheless complement the
champion: averaging ReXNet-200 with B3 and B0-4e-4 scores 0.892798, +0.0043 with a 95% interval
excluding zero.** Runs `outputs/training/effnetb3`, `effnetb0` and `effnetb0higherlr`, fold 0 of
`all_5fold_v3`, seed 42, validation multiplier ×1, launched by the user on the uncommitted tree at
`62ad67d` plus the mapping change below. Control is `04-total-topup-26` from the
[2026-09-18 overnight queue](../campaigns-and-queues/2026-09-18-overnight-results.md), the only prior run matching the
recipe of that date: its effective config differs from these only in `model.backbone`, the removed
`data.robust_center_outlier_mm` (`None` there, so the same code path) and the seed.

## Runs

| Run | Backbone | Channels | Params | `backbone_lr` | Best macro AUC | Epoch | BCE | Train | Peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `04-total-topup-26` | `rexnet_200.nav_in1k` | 2560 | 13.81M | 2e-4 | **0.888476** | 7 | 0.45403 | 1541.3 s | 19.58 GiB |
| `effnetb3` | `efficientnet_b3.ra2_in1k` | 1536 | 10.70M | 2e-4 | 0.879449 | 7 | 0.45970 | 1651.7 s | 17.84 GiB |
| `effnetb0` | `efficientnet_b0.ra_in1k` | 1280 | 4.01M | 2e-4 | 0.880331 | 9 | 0.45802 | 1023.1 s | 9.38 GiB |
| `effnetb0higherlr` | `efficientnet_b0.ra_in1k` | 1280 | 4.01M | **4e-4** | 0.883206 | 8 | 0.45571 | 1027.6 s | 9.38 GiB |

All four ran 4,380 optimizer steps. Paired bootstrap against the control, 2,000 resamples of the
904 fold-0 studies with checkpoints fixed: B3 −0.0090 [−0.0139, −0.0042], B0-4e-4 −0.0053
[−0.0093, −0.0010]. Both exclude zero.

## Code required

`efficientnet_b3.ra2_in1k`, `efficientnet_b0.ra_in1k` and `efficientnet_b0.ra4_e3600_r224_in1k`
were added to `BACKBONE_MAPPINGS` with a new `_efficientnet_mapping`, and `backbone_mapping` became
a name-to-function lookup. See
[swapping a backbone](../../../modeling/backbones/backbone-swaps.md#code-requirements). No change to
`modeling/model.py` was needed: it already derives the projection width from `num_features`.

- **Non-`tf_` EfficientNets share ReXNet-200's anchors exactly**, origin `[0, 0]` and stride
  `[32, 32]`, verified for B0–B3 and `efficientnetv2_rw_t` before registration.
- **`tf_` ports remain unusable.** `tf_efficientnet_b0.ns_jft_in1k` walks to origin **51.0**
  because `Conv2dSame` reports zero padding and pads inside `forward`. This blocks every
  AdvProp, AutoAugment and noisy-student/JFT EfficientNet checkpoint, since all of them are `tf_`
  ports. `efficientnet_b3` has exactly one non-`tf_` tag.
- **Normalization was matched to each checkpoint.** B3 `ra2_in1k` and B0 `ra_in1k` both declare the
  configured ImageNet mean/std, so `data.mean`/`data.std` were unchanged and the backbone stayed
  the only variable. B0 `ra4_e3600_r224_in1k` declares `[0.5, 0.5, 0.5]` and was rejected for that
  reason; it stays registered but unrun.

## EfficientNet underfits; it does not overfit

Training loss per epoch is worse for EfficientNet at nearly every epoch, while the generalization
gap is *smaller*:

| Epoch | Train loss ctrl / B3 / B0 | Val BCE − train loss, ctrl / B3 / B0 |
|---:|---|---|
| 0 | 0.5770 / 0.5819 / 0.5806 | −0.0179 / −0.0287 / −0.0504 |
| 4 | 0.4794 / 0.4841 / 0.4919 | −0.0004 / −0.0147 / −0.0057 |
| 9 | **0.4242** / 0.4351 / 0.4467 | **+0.0285** / +0.0237 / **+0.0113** |

Final training loss orders control < B3 < B0, the reverse of parameter count, while B0 has by far
the smallest gap. Gradient norms are indistinguishable across runs (median 0.520 / 0.497 / 0.530,
maximum 0.894) and **`trainer.gradient_clip_val: 1.0` never fires in any run**.

This contradicts the hypotheses in [EfficientNet training](../../../research/backbones-and-pretraining/efficientnet-training.md),
which were written to diagnose a convergence failure. These runs converge normally; they converge
to a worse point. Its proposed runs B (`backbone_lr` 5e-5) and C (frozen BN statistics) both
*reduce* adaptation and were not run. Supporting evidence from the same overnight queue on
ReXNet-200: `12-backbone-lr-1e-4` −0.0030 and `16-backbone-eval` −0.0088.

## Doubling the backbone learning rate

`backbone_lr` 2e-4 → 4e-4 on B0 gained **+0.0029** (0.880331 → 0.883206) and moved the peak from
epoch 9 to 8. Bootstrap [−0.0010, +0.0067], P(better) 93.8%: the interval includes zero and the
gain sits on the ~0.003 single-run threshold, so this is **promising, not established**.

**The mechanism was not the one predicted.** Final training loss barely moved (0.4467 → 0.4450) and
was equal or worse through epochs 1–7, so the higher rate did not make the backbone fit faster. If
the gain is real it is a regularization effect, not an adaptation one. The prediction that
motivated the run — that EfficientNet underfits because its backbone adapts too slowly for the
configured rate — is not supported. Under AdamW the update is already normalized per parameter,
which is a reason not to expect a learning-rate change to act through fitting speed.

At 2e-4 B0's validation curve was still rising at the final epoch, which would have motivated a
longer budget; at 4e-4 it turns over inside ten epochs, so that case is weaker.

## Per-label

Fold-0 per-target AUC, paired bootstrap, 2,000 resamples. No `*` marks a significant per-label
difference except B3's MCL.

| Target | ReXNet | B3 | B0 | B0-4e-4 | B3 − ReXNet | B0-4e-4 − ReXNet | Pos |
|---|---:|---:|---:|---:|---:|---:|---:|
| MCL | 0.8445 | 0.8044 | 0.8332 | 0.8314 | **−0.0400** \* | −0.0131 | 134 |
| Lateral Meniscus | 0.8807 | 0.8665 | 0.8443 | 0.8683 | −0.0142 | −0.0124 | 138 |
| Fracture | 0.9274 | 0.9139 | 0.9251 | 0.9175 | −0.0135 | −0.0099 | 63 |
| ACL | 0.9120 | 0.8989 | 0.9100 | 0.9133 | −0.0131 | +0.0013 | 184 |
| PF OA | 0.8719 | 0.8630 | 0.8593 | 0.8620 | −0.0089 | −0.0099 | 406 |
| Medial Meniscus | 0.9187 | 0.9104 | 0.9105 | 0.9095 | −0.0083 | −0.0092 | 356 |
| Synovitis | 0.8947 | 0.8883 | 0.8971 | 0.8992 | −0.0064 | +0.0046 | 115 |
| Baker's | 0.9057 | 0.9004 | 0.8944 | 0.9074 | −0.0053 | +0.0017 | 216 |
| Lateral OA | 0.8604 | 0.8570 | 0.8491 | 0.8497 | −0.0034 | −0.0107 | 241 |
| Medial OA | 0.9009 | 0.8994 | 0.8972 | 0.8929 | −0.0015 | −0.0080 | 329 |
| Effusion | 0.8626 | 0.8631 | 0.8644 | 0.8624 | +0.0005 | −0.0002 | 525 |
| Contusion | 0.8823 | 0.8883 | 0.8796 | 0.8849 | +0.0059 | +0.0026 | 153 |

- **B3's loss is one label.** MCL at −0.0400 is the only per-label interval excluding zero
  ([−0.0715, −0.0112], P 0.4%). Excluding it, the other eleven average −0.0062.
- **B0-4e-4's loss is broad and shallow** — seven labels between −0.008 and −0.013, no outlier,
  four mildly positive. Consistent with the regularization reading above.
- **The durable pattern is anatomy, not architecture size.** Every EfficientNet run trails on thin,
  localized structures (MCL, lateral meniscus, fracture, medial meniscus) and matches or beats
  ReXNet-200 on diffuse findings (effusion, contusion, synovitis). Individually none of these is
  significant, so treat the pattern as suggestive.
- **The two EfficientNets fail on different labels** (B3 worst on MCL, B0 worst on lateral
  meniscus) while landing within 0.001 of each other on macro, which argues against a single
  shared architectural cause.

## Complementarity

Probability averaging of saved fold-0 predictions, no retraining, equal weights, bootstrap against
ReXNet-200 alone. Probability averaging beat rank averaging everywhere (+0.0035 against +0.0019 for
the three-way).

| Combination | Macro AUC | Δ | 95% interval | P |
|---|---:|---:|---|---:|
| ReXNet-200 alone | 0.888476 | — | | |
| B3 + B0 | 0.886663 | −0.0018 | | |
| ReXNet + B0 | 0.890743 | +0.0023 | [+0.0000, +0.0045] | 97.7% |
| ReXNet + B3 | 0.891443 | +0.0030 | [+0.0007, +0.0053] | 99.2% |
| ReXNet + B0-4e-4 | 0.891759 | +0.0033 | [+0.0013, +0.0055] | 99.9% |
| **ReXNet + B3 + B0-4e-4** | **0.892798** | **+0.0043** | [+0.0018, +0.0069] | 100.0% |

- **The weaker model is the better partner.** B3 loses by 0.0090 alone but adds more than B0, and
  it is the most decorrelated from ReXNet-200 (mean Spearman 0.9062 against 0.9200 for B0). The two
  EfficientNets are less alike to each other (0.9109) than ReXNet is to B0.
- **Two EfficientNets together still lose to ReXNet-200 alone.** They are a complement, not a
  substitute.
- **The ensemble gains where the EfficientNets were competitive and loses where they were weak:**
  effusion +0.0081 and contusion +0.0074, against lateral meniscus −0.0080 and MCL −0.0037.
- **Selection optimism is uncorrected.** All checkpoints were chosen by best epoch on this same
  fold-0 set; the bootstrap resamples studies with checkpoints fixed and does not correct for that.
  The true gain is smaller. Ensembling was left to the endgame ([untried ideas](../../../backlog/untried-ideas.md#ensembles)).

## Cost: multiply-adds did not predict runtime

| Backbone | GMAC per triplet | Train | Peak |
|---|---:|---:|---:|
| ReXNet-200 | 1.561 | 1541.3 s | 19.58 GiB |
| EfficientNet-B3 | ~0.98 (−37%) | **1651.7 s (+7.2%)** | 17.84 GiB |
| EfficientNet-B0 | ~0.40 (−74%) | 1023.1 s (−33.6%) | 9.38 GiB |

B3 has 37% fewer multiply-adds and ran **slower**; B0 has 74% fewer and ran 34% faster, well short
of proportional. Depthwise convolutions and squeeze-excite blocks are bandwidth-bound and do not
convert multiply-add savings into wall time on the local 3090. This confirms the existing rule in
[swapping a backbone](../../../modeling/backbones/backbone-swaps.md) against substituting model-card figures for
measured runtime. These are local training measurements; inference cost must be measured on Kaggle
T4s, and the [loading speedup](../../../inference/profiling/loading-cost-attribution.md) profile shows the loader, not the
GPUs, set inference throughput then, so a cheaper backbone could not be priced until that was
fixed.

## Reading

EfficientNet is rejected as a replacement backbone. This is the second EfficientNet evidence on this
dataset after EfficientNetV2-S ranked 38th of 40 in the
[fast backbone search](../../../modeling/backbones/backbone-search.md), and it agrees with it. The hyperparameter
axis is close to exhausted: the two directions proposed in
[EfficientNet training](../../../research/backbones-and-pretraining/efficientnet-training.md) both point the wrong way for a model
that underfits, and doubling the learning rate produced a sub-threshold gain through a mechanism that
was not the predicted one.

The durable positive finding is complementarity: a family that loses by 0.005–0.009 alone adds
+0.0043 in a three-model average with intervals excluding zero, and B0 does so at a third of the
parameters and half the training memory. That belongs to the final-phase ensembling work, not to
backbone selection.
