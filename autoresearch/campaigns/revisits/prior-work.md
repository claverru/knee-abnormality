# Revisits prior work

Where the seeds come from and what ResNet-18 has already measured. Every row is one run and weak
evidence. Re-scored Δ is ×4 macro AUC on `rebuild_state_v2` with the 80% paired report-group
interval ([campaign re-scoring](../../../docs/validation/results/2026-09-28-campaign-rescoring.md)).

## Seed precedents

| Seed | Earlier run | Recipe | Recorded Δ | Re-scored Δ [80%] |
|---|---|---|---:|---|
| Dense per-plane readouts | `dense-plane-readouts-20260927T222330Z-a9747fc9` | three EfficientNet-B0s, one per plane | +0.0047 | +0.0032 [+0.0010, +0.0053] |
| Learned plane fusion on top | `dense-readouts-learned-fusion-20260927T231847Z-33417f97` | same | +0.0009 | +0.0008 [−0.0005, +0.0022] |
| Plane/contrast readouts | `plane-contrast-readouts-20260925T020628Z-2efb517d` | ReXNet-200, near-misses | −0.0006 | +0.0018 [+0.0001, +0.0035] |
| DSU (adoption) | augmentation campaign, on smoothing | ReXNet-200 | +0.0022 | +0.0011 [−0.0014, +0.0035] |
| DSU (replication) | `dsu-without-smoothing-vs-report-attributes-control` | ReXNet-200 | +0.0012 | −0.0003 [−0.0020, +0.0015] |
| DSU at probability 0.25 | `rn18-lighter-dsu` | ResNet-18 | +0.00005 | (already v2) |
| Anatomical bias, learned strengths | `anatomical-strength-vs-control-repeat` | ReXNet-200 | +0.0023 | +0.0001 [−0.0019, +0.0023] |
| | `anatomical-learned-vs-control-b` | ReXNet-200 | +0.0009 | +0.0016 [−0.0008, +0.0039] |
| Bilateral smoothing | augmentation campaign | ReXNet-200 | +0.0020 | +0.0008 [−0.0009, +0.0027] |
| No contrast mixing | `rn18-native-contrast-preservation` | ResNet-18 | +0.0006 | (already v2) |
| GIN replaced by a 13× weaker augmentation | `site-acquisition-mixture-contrast-final` | ResNet-18, sites-out split | −0.0015 | (already v2) |
| Stochastic depth 0.1 | `rexnet-stochastic-depth-20260926T045526Z-1de7a03e` | ReXNet-200 | +0.0011 | +0.0006 [−0.0013, +0.0025] |
| Plane/contrast feature adapters | none | — | — | [backlog](../../../docs/backlog/small-changes.md#planecontrast-specific-feature-adapters) |
| Plane-conditioned BatchNorm | none | — | — | queued in [radimagenet-rn18](../radimagenet-rn18/ideas.yaml) |

Frozen sources for the ported seeds are under `outputs/autoresearch/<campaign>/experiments/<run>/source/`.

## Measured on ResNet-18

Do not propose these again at the same value
([radimagenet-rn18 findings](../radimagenet-rn18/findings.md),
[site-generalization findings](../site-generalization/findings.md)):

- **Rates and schedule:** backbone LR 1e-4 (−0.0017), layer-wise LR decay (−0.0006), two warmup
  epochs (−0.0007), warmup-stable-decay (−0.0003), cosine floor 0.1 (−0.0002), 14 epochs (+0.0010
  for 40% more time), 40 triplets in nine epochs (−0.0015), separate backbone/head clipping
  (−0.0024).
- **Backbone handling:** linear probe then fine-tune (−0.0012; +0.0005 on L2-SP), frozen stem and
  layer1 (−0.0029), L2-SP (adopted), BN-adapted head start
  (−0.0004), frozen BatchNorm (−0.0086), early-stage anchor (−0.0004).
- **Regularization and input:** series dropout 0 (adopted, +0.0023), DSU 0.25 (+0.00005), 36
  triplets (+0.0018, deferred to the endgame), mean-initialized adjacent slices (−0.0039), gated
  adjacent residuals (−0.00005), stride-16 tokens (−0.0023), full-width projection (−0.0012).
- **Head and loss:** asymmetric focal loss (−0.0024), 101-bin head (−0.0139).
- **Attention temperature:** τ 0.9–1.0 beats 0.75 at inference
  ([result](../../../docs/validation/results/2026-09-29-rn18-attention-temperature.md)); the campaign
  scores at τ 1.

## Dropped from the list

Early-learning regularization (+0.0017 and +0.0008 re-scored in two runs; the model never reaches
the memorization phase it guards against), the token-dropout variants (inert mechanism), the
OA-geometry residual and any-OA auxiliary (zero net of the reference shift), ACL-gated oblique
tokens (oblique line retired) and LP-FT (already on L2-SP).
