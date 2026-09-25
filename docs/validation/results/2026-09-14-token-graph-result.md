# 2026-09-14 token-graph result

**A fold-0 tie, and the trained model does not use the graph.** The `token-graph` run
selected epoch 9 (index 8) at **0.892376**, +0.000016 over reflection/sigma 40's 0.892360.
The paired 95% interval for the macro difference is [−0.00349, +0.00360]. There is no
public score. [Graph interventions](#graph-interventions-the-graph-is-unused) show that
removing the graph changes predictions negligibly, which meets the plan's retirement rule.

| Selected model | Macro AUC | Meniscus mean | OA pair mean | Other eight | Soft-target BCE | Gold (11) |
|---|---:|---:|---:|---:|---:|---:|
| Reflection/sigma 40 | 0.892360 | 0.901105 | **0.882491** | **0.892641** | 0.451855 | **0.922156** |
| Token graph | **0.892376** | **0.905196** | 0.882151 | 0.891727 | **0.448040** | 0.920734 |

- **Per target:** medial meniscus +0.0089, MCL +0.0069, Synovitis +0.0043, PF OA
  +0.0021, Fracture +0.0012 and medial OA +0.0009 improved. Baker's −0.0114,
  Effusion −0.0051, ACL −0.0041, lateral OA −0.0016, Contusion −0.0011 and lateral
  meniscus −0.0008 declined. Report-derived AUC moved from 0.891725 to 0.891814.
- **Family intervals** (report-group bootstrap, 2,000 draws, 852 groups) all include zero:
  meniscus +0.0040 [−0.0027, +0.0111], OA [−0.0074, +0.0069] and other eight
  [−0.0055, +0.0037].
- **Compartments:** the exclusive-positive meniscus margin rose from 0.9690 to 0.9729,
  interval [−0.0046, +0.0120]; the OA margin fell from 0.8791 to 0.8670, interval
  [−0.0440, +0.0179]. Lateral cross-talk was unchanged: 31 of 45 meniscus false positives
  are medial-positive in both runs, and 51 of 80 OA against 53 of 83.
- **Learning curve:** the graph trailed the champion at epochs 2 and 4–8 (for example
  0.8680 against 0.8794 at epoch 4) and caught up only at epochs 9–10, finishing at 0.892240
  against 0.891347. Its best checkpoint is two epochs later than the champion's, so the
  lower BCE partly reflects a later epoch.
- **Prediction change:** mean absolute probability difference was 0.054, and per-target
  Spearman correlations were 0.90–0.97, lowest for Baker's.
- **Cost:** training took 1,737 s against 1,700 s, validation prediction 34.7 s against
  31.7 s, and peak CUDA allocation was 20.32 GB against 20.46 GB, with 4,380 optimizer
  steps each.

Configurations differ only in name and the two graph fields. Label, split, membership
and series identities match; preprocessing identity and sources differ by the graph,
the removed 3D embedding, and the earlier CoAtNet refactor and knee-side retirement.
The run changes the graph and 3D removal together, and intervals exclude training-seed and
checkpoint-selection variability. The
[comparison artifact](../../../outputs/analysis/token-graph-vs-reflection-sigma40/comparison.json)
was produced by the [fold-run comparison](../fold-run-comparison.md); its run-file
fingerprints were rechecked against the current run.

## Graph interventions: the graph is unused

The [encoding diagnostic](../manual-encoding-usage-diagnosis.md) on the selected checkpoint
reproduced saved probabilities (maximum difference 3e−8) with no preparation conditions.
All 904 studies contain all three planes, and no token was isolated.

| Variant | Macro AUC change | Mean absolute probability change | Paired 95% interval |
|---|---:|---:|---|
| Without graph | +0.000007 | 0.00055 | [−0.00005, +0.00007] |
| Without grid / same-series / cross-plane edges | −0.00002 / +0.000004 / −0.00001 | ≤0.00009 | all include zero |
| Without all geometric edges | −0.00001 | 0.00021 | [−0.00007, +0.00004] |
| Without 1D | −0.00121 | 0.0506 | [−0.00225, −0.00009] |
| Without 2D | −0.00007 | 0.0364 | [−0.00048, +0.00037] |
| Without plane / contrast | +0.000008 / +0.00002 | 0.0019 / 0.0025 | include zero |
| Without all position (1D, 2D and graph) | −0.00203 | 0.0854 | [−0.00317, −0.00088] |

- **No target moved** by more than 0.00014 AUC without the graph. Meniscus and OA margins
  and cross-talk were unchanged (0.9729/0.8671; 68.9%/63.7%).
- **The graph barely trained.** Query, key and message weights have RMS 0.036–0.038, equal to
  their default initialization scale. The zero-initialized output projections grew only to
  RMS 0.0038. Each token still received about 18 valid edges, including 4.7–6.5 cross-plane
  edges per token by receiving plane, so neighbours were available. The optimizer found no
  useful direction for message passing.
- **Plane-only pooling:**

  | Plane only | Macro AUC | Meniscus margin | OA margin |
  |---|---:|---:|---:|
  | Sagittal | 0.8626 | 0.9582 | 0.8244 |
  | Coronal | 0.8537 | 0.9650 | 0.8763 |
  | Axial | 0.8255 | 0.6459 | 0.5971 |

  All three compare with 0.8924 for every plane. Because the graph is unused, these
  approximate the model's per-plane content, not cross-plane benefit. Sagittal and coronal
  tokens separate meniscus compartments almost as well as the full model. This contradicts
  the research hypothesis that sagittal tokens lack compartment identity, and plane-only
  inputs were rare in training.

Following the plan's reading rule (`f076a09:docs/history/token-graph-sources/token-graph-plan.md.txt`) (section 6),
an unused graph is retired whatever its macro AUC. The result rejects this optimization
setup (zero-initialized residual update before LayerNorm, head learning rate and weight
decay, ten epochs), not every possible form of token interaction. Artifacts are in
`outputs/analysis/encoding-usage-token-graph/`.

## 2026-09-15 forced graph training: still unused

`configs/graph-forced.yaml` retrained the graph to force message passing to learn. Its
configuration differs from `token-graph` in three ways: default Linear initialization for
the output projections (`graph_output_gain: 1.0`), the update applied after `mil_norm`
(`graph_placement: after_layernorm`), and a separate graph optimizer group with learning
rate 0.0018 and no weight decay. Seed, fold and recipe are unchanged.

The run `outputs/training/graph-forced` selected epoch 10 at **0.892679** macro AUC, with
soft-target BCE 0.44639. That compares with 0.892763 for the graph-off
`00-baseline` of the 2026-09-14 overnight queue, 0.892376 for `token-graph` and 0.892360
for the champion. Training took 1,687 s, with peak allocation 20.33 GB. There is no public
score.

The graph weights barely moved from initialization: output projection RMS stayed at
0.051, the default scale, and query RMS rose only from about 0.036 to 0.039. The
[encoding diagnostic](../manual-encoding-usage-diagnosis.md) reproduced saved probabilities
(maximum difference 3e−8) and showed the model does not rely on the graph:

| Variant | Macro AUC change | Mean absolute probability change | Paired 95% interval |
|---|---:|---:|---|
| Without graph | +0.000039 | 0.0038 | [−0.00015, +0.00023] |
| Without cross-plane edges | −0.000005 | 0.00018 | [−0.00005, +0.00004] |
| Without all geometric edges | −0.000003 | 0.00039 | [−0.00006, +0.00006] |
| Without grid edges | −0.000062 | 0.00044 | [−0.00015, +0.00001] |
| Without 1D | −0.003009 | 0.0525 | [−0.00427, −0.00184] |
| Without 2D | −0.000671 | 0.0331 | [−0.00107, −0.00027] |
| Without all position (1D, 2D and graph) | −0.003318 | 0.0950 | [−0.00463, −0.00204] |

No target moved by more than 0.0007 without the graph. Meniscus and OA margins were
unchanged (0.9690/0.8783 intact). Starting the graph with active messages, a higher
learning rate and post-norm placement therefore did not make message passing useful.
The model adapted to near-initial messages instead of learning from them. 1D and 2D
position mattered more than in earlier checkpoints. Artifacts are in
`outputs/analysis/encoding-usage-graph-forced/`.
