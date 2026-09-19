# 2026-09-18 per-plane sampler sigma on the token cache

**Coronal is the only axis whose sampler width matters; axial never does.** On frozen checkpoints
at the ×1 count, narrowing the coronal draw to 30 mm is worth +0.00135 macro AUC [+0.0004, +0.0025]
and widening it to 120 mm costs -0.0015, with the sign holding at every count multiplier from ×1 to
×4. Axial sigma does nothing anywhere between 20 and 120 mm, at matched prediction movement. A
sagittal gain at ×1 (+0.0011) does not survive higher counts and is withdrawn.

This reverses the arm the [spatial importance map](../../ongoing/peripheral-coverage.md#spatial-importance-map)
proposed, which ranked axial as about seven times as sigma-sensitive as sagittal and led to the
`axial 30, coronal and sagittal 40` first arm in
[per-plane sampler sigma](../../backlog/small-changes.md#per-plane-sampler-sigma-superseded). The map's axis
ranking does not survive a direct test.

Measured at inference only. Training-time sigma is a different quantity and still needs its run.

## Setup

- **Checkpoints and cache:** `ranked-filtered-seed{42,43,44}` on the fold-0 token cache
  (`outputs/analysis/token-cache-20260917/`), 904 studies. Selections are exact: sigma is passed
  straight to the snapshot's own `select_triplets`, and the sigma-40 baseline reproduces
  `selections.npz` triplet for triplet. No slot has recorded failures, so the preparation retry is
  a no-op.
- **Counts:** training counts, 6/6/5/5/5/5 in slot order, which is the ×1 validation the current
  recipe uses; realised mean 25.36 triplets per study, identical at every sigma. Sigma changes
  which triplets are drawn, never how many.
- **Baseline:** sigma 40 on all three planes, macro AUC 0.887271 (search 0.889653, confirm
  0.885384).
- **Statistics:** means over the three checkpoints; paired bootstrap over studies, 500 resamples,
  seed 20260918. The bootstrap SEs come in at 0.7–0.9× the
  [detection-floor rule](../validation-strategy.md#what-a-comparison-can-resolve)
  `SE ≈ 0.067·p^1.05`, as that rule predicts for a three-checkpoint mean.
- **Selection-jitter floor:** sigma 39.9 and 40.1 move macro AUC by at most 0.00009, so the
  discrete reshuffling of which triplets get picked is not what the effects below are made of.
- **Scripts and outputs:** `outputs/analysis/per-plane-sigma-20260918/`
  (`01_score_sigma.py`, `02_coronal_detail.py`, `03_combined.py`, `04_bootstrap.py`).

## One plane at a time

Δ macro AUC against the sigma-40 baseline, the other two planes held at 40.

| Sigma | Axial | Coronal | Sagittal |
|---:|---:|---:|---:|
| 20 | +0.00001 | +0.00085 | +0.00018 |
| 25 | +0.00020 | +0.00094 | +0.00110 |
| 30 | -0.00009 | +0.00135 | +0.00027 |
| 35 | -0.00058 | +0.00074 | +0.00008 |
| 45 | +0.00039 | -0.00032 | -0.00006 |
| 50 | +0.00035 | -0.00035 | +0.00005 |
| 60 | +0.00003 | -0.00060 | -0.00021 |
| 80 | +0.00020 | -0.00140 | +0.00004 |
| 120 | -0.00013 | -0.00150 | +0.00014 |

- **Coronal is monotone and signed**: every value from 10 to 35 mm gains, every value from 45 to
  120 mm loses, sixteen settings with no sign exception. The gain is a plateau, not a peak — 12 mm
  (+0.00165) and 30 mm (+0.00135) are within each other's interval — so the readable statement is
  "narrower than 40", not a located optimum.
- **Sagittal does not hold up**: 25 mm is the best point at ×1 (+0.0011, interval crossing zero)
  but flips to -0.0005 at ×2 and -0.0003 at ×3. Treat it as a low-count artifact, not a finding.
- **Axial is flat.** Not a leverage problem: axial sigma moves realised mean |depth| from 14.2 mm
  at sigma 20 to 27.1 mm at sigma 120, the widest range of the three planes (coronal 13.5–23.5,
  sagittal 13.7–22.3). Sigma 120 moves predictions as much as coronal 30 does (mean absolute
  probability change 0.0142 against 0.0137) and still scores -0.00013 [-0.0012, +0.0009]. Matched
  movement, no effect.

## Bootstrap and confirmation

| Setting | p | Δ all | 95% CI | Δ search | Δ confirm |
|---|---:|---:|---|---:|---:|
| Coronal 30 | 0.0137 | **+0.00135** | [+0.00039, +0.00249] | +0.00086 | +0.00198 |
| Sagittal 25 | 0.0236 | +0.00110 | [-0.00046, +0.00260] | +0.00096 | +0.00127 |
| Coronal 30 + sagittal 25 | 0.0285 | **+0.00229** | [+0.00056, +0.00401] | +0.00179 | +0.00297 |
| Axial 25 | 0.0156 | +0.00020 | [-0.00114, +0.00133] | +0.00071 | -0.00025 |
| Axial 45 | 0.0057 | +0.00039 | [-0.00012, +0.00092] | +0.00082 | -0.00017 |
| Axial 120 | 0.0142 | -0.00013 | [-0.00121, +0.00089] | -0.00000 | -0.00027 |
| Global 25 | 0.0352 | +0.00213 | [-0.00009, +0.00405] | +0.00257 | +0.00190 |

- **The two planes add up.** +0.00135 and +0.00110 separately, +0.00229 together against +0.00245
  predicted by addition.
- **Within a plane, both slots contribute**, mostly the fluid one: at sigma 30, coronal T1/PD
  +0.00042 and coronal fluid +0.00109 alone, +0.00135 together.
- **Adding axial 45 on top of coronal 30 + sagittal 25** buys +0.00036 on the full fold and
  +0.00001 on confirm. Nothing.
- **Global narrowing is the coronal and sagittal effect and nothing else.** Global 25 scores
  +0.00213, indistinguishable from coronal 30 + sagittal 25 at +0.00229.

## Count multipliers

Δ against the sigma-40 baseline at the same multiplier, ×1 being the recipe's training and
validation count (25.4 triplets per study) and ×4 what recent submissions use (99.3).

| Setting | ×1 | ×2 | ×3 | ×4 |
|---|---:|---:|---:|---:|
| Coronal 30 | +0.00135 | +0.00070 | +0.00048 | +0.00033 |
| Coronal 120 | -0.00150 | -0.00080 | -0.00073 | -0.00057 |
| Sagittal 25 | +0.00110 | -0.00051 | -0.00025 | +0.00013 |
| Axial 30 | -0.00009 | +0.00014 | -0.00022 | -0.00008 |
| Axial 120 | -0.00013 | -0.00009 | +0.00019 | -0.00000 |

- **The coronal effect decays with count and keeps its sign at all four.** More triplets cover the
  depth range by brute force, so width buys less; the gain is a coverage effect, largest exactly at
  the ×1 count that training draws.
- **Nothing here is worth an inference-side change.** At ×4 coronal 30 is +0.00033, inside noise.
  Unlike [attention temperature](2026-09-17-attention-temperature.md), this does not transfer to a
  submission on its own.
- **The axis contrast is what survives everywhere**, which is the part that bears on the map.

## Why this is trusted against the map

- **The cache reproduces the global sigma history it was not fitted to.** Training runs settled 30
  tying 40 (+0.000039) and 60, 80 and 120 all losing
  ([overnight 2026-09-15](2026-09-15-overnight-results.md)). The cache gives global 30 +0.00146 and
  global 60 -0.00071: both signs match, with the narrow side larger at inference than in training,
  which is what a train/test mismatch penalty on the training side would do.
- **Both quantities are inference-time.** The map's swings (0.677 axial, 0.310 coronal, 0.092
  sagittal) come from leave-one-out window ablation on the same ensemble, so this is not a
  training-versus-inference disagreement. The map's expected-utility-per-slice model ranks the
  axes; scoring the actual selections reverses that ranking.
- **The likely reason is the map's own stated bias.** It is observational under the sigma-40
  sampler, so it understates utility where the sampler rarely goes, and the axial axis is where the
  candidate pool extends furthest. Its per-axis absolute profiles were already flagged as not
  surviving that bias; this measurement shows the between-axis ratio does not survive it either.

## Per-target, coronal 30 + sagittal 25

Gains: MCL +0.0066, lateral OA +0.0064, lateral meniscus +0.0059, Baker's cyst +0.0050, medial OA
+0.0021, effusion +0.0017, fracture +0.0011. Losses: synovitis -0.0011, medial meniscus -0.0010,
PF OA -0.0002.

The gainers are the compartment and collateral labels, which is where narrowing concentrates the
coronal and sagittal draws. Synovitis and effusion sit where the
[depth profiles](../../ongoing/peripheral-coverage.md) put their evidence, 30–50 mm off centre in
axial fluid, and the axial draw is untouched here.

## Limits

- **Inference only.** The checkpoints were trained at sigma 40, so these numbers measure sampling
  a fixed model better, not training under a different sampler. The narrow settings are a
  train/test mismatch that still wins, which bounds the training gain from below only if training
  at 30 does not change what the model learns — it will.
- **One fold, three checkpoints of one recipe.** The three seeds share the recipe, so they bound
  selection noise, not recipe sensitivity.
- **The coronal plateau is not resolved.** Anything from 12 to 35 mm scores the same here; 30 is
  chosen as the conservative end of the plateau, nearest the settled global value.
- **The combined and per-target numbers above are ×1 only.** The +0.00229 pair and its per-target
  split include the withdrawn sagittal arm; the durable part is coronal.
