# 2026-09-22 stride-16/stride-32 fusion: three designs tie with `k_05`

**Sum fusion ties `k_05` on fold 0 at no extra cost** (+0.0011 at ×4, paired 95% −0.0023 to
+0.0044; −0.0002 at ×1). It recovers truncation's ACL and PF OA losses. A branch ablation on the
same checkpoint shows the predictions rest almost entirely on the deep term: zeroing the fine term
costs −0.0032 macro and leaves the menisci unchanged, although the fine term is the larger one. No
submission was made. Part of the [stride-16 investigation](../../history/stride16-feature-fusion.md),
closed without adoption: [deep-branch dropout](#deep-branch-dropout-rejected) traded labels
(macro −0.0038 against `fusion-sum`) and [per-scale heads](#per-scale-heads) tied (+0.0006
against `k_05`).

## Run

`fusion-sum` sets `model.stride16_fusion: true`: tokens are `P16(F16) + resample(P32(F32))` on the
14×14 stride-16 grid, 196 per triplet, with the full ReXNet-200 kept. Its effective configuration
equals `k_05`'s except `model.stride16_fusion` and the byte-identical label-manifest path: fold 0 of
`all_5fold_scanner_v1`, [combined labels](../../public-labels/combined-labels.md), class-balanced
BCE with k=0.5, seed 2263738815 (training is not deterministic), ten epochs. It was trained from
the uncommitted working tree on `ab49830`, captured in `source_snapshot.zip`, and selected epoch 10
at ×1; the ×1 curve was still rising (0.8976 at epoch 9, 0.8988 at epoch 10), while `k_05` and
`stride16` peaked at epoch 8.

| Run | Selection score (×1) | Macro AUC (×4) | Training | Peak allocated | Parameters |
|---|---:|---:|---:|---:|---:|
| `k_05` | 0.899042 | 0.902469 | 1,584 s | 21.0 GB | 14.60M |
| [`stride16`](2026-09-22-stride16-truncation-result.md) | 0.894765 | 0.898229 | 1,379 s | 18.5 GB | 4.38M |
| `fusion-sum` | 0.898837 | 0.903516 | 1,584 s | 21.1 GB | 14.67M |

## Validation

Paired report-group bootstrap, 2,000 draws, from [`compare_fold_runs`](../fold-run-comparison.md);
`outputs/analysis/f/fusion-sum-vs-k05/` and `outputs/analysis/f/fusion-sum-vs-stride16/`.

| Group | vs `k_05` | vs `stride16` |
|---|---|---|
| Macro | +0.0011 [−0.0023, +0.0044] | +0.0053 [+0.0014, +0.0091] |
| Meniscus | +0.0027 [−0.0039, +0.0092] | −0.0036 [−0.0104, +0.0028] |
| OA | −0.0014 [−0.0077, +0.0044] | +0.0011 [−0.0054, +0.0080] |
| Other eight | +0.0013 [−0.0029, +0.0058] | +0.0085 [+0.0037, +0.0135] |
| OA margin | +0.0229 [+0.0021, +0.0458] | +0.0106 [−0.0115, +0.0338] |

| Label | `k_05` | `stride16` Δ | `fusion-sum` Δ |
|---|---:|---:|---:|
| ACL | 0.9463 | −0.0096 | +0.0094 |
| MCL | 0.9249 | −0.0052 | +0.0041 |
| Medial Meniscus | 0.9295 | +0.0026 | +0.0048 |
| Lateral Meniscus | 0.8972 | +0.0101 | +0.0008 |
| Medial OA | 0.9008 | +0.0002 | +0.0008 |
| Lateral OA | 0.8756 | −0.0054 | −0.0037 |
| PF OA | 0.8647 | −0.0319 | −0.0091 |
| Effusion | 0.8660 | −0.0035 | −0.0024 |
| Synovitis | 0.8877 | −0.0063 | +0.0007 |
| Baker's | 0.9147 | +0.0085 | +0.0009 |
| Contusion | 0.9113 | −0.0077 | −0.0071 |
| Fracture | 0.9109 | −0.0027 | +0.0133 |

Truncation's meniscus gain was itself inside noise (+0.0063 [−0.0009, +0.0143]), so fusion's
failure to keep it is not established by the between-run comparison alone. Gold macro AUC
(13 studies) is 0.9025 against 0.9075 and cannot be weighed.

## Branch ablation

[`diagnose_fusion_branches`](../../../scripts/adhoc/diagnose_fusion_branches.py) re-predicted the selected
checkpoint at ×4 three times (`outputs/analysis/f/fusion-branches-fusion-sum/summary.json`). The
unmodified pass reproduced the recorded 0.903516 within 2.5e-7. Zeroing a term also shifts the
LayerNorm input, so the deep-zeroed drop overstates dependence; the fine-zeroed pass is the
informative one. The fusion model code is not in the tree, so the script runs only from the
sources in `outputs/training/fusion-scaleheads/source_snapshot.zip`, on GPU:
`python -m scripts.adhoc.diagnose_fusion_branches --run outputs/training/fusion-sum --output-dir <fresh dir>`.

| Label | Full | Fine zeroed Δ | Deep zeroed Δ |
|---|---:|---:|---:|
| Macro | 0.9035 | −0.0032 | −0.1367 |
| ACL | 0.9556 | −0.0005 | −0.2159 |
| MCL | 0.9290 | +0.0064 | −0.1475 |
| Medial Meniscus | 0.9343 | −0.0007 | −0.1468 |
| Lateral Meniscus | 0.8979 | +0.0005 | −0.2089 |
| Medial OA | 0.9016 | −0.0087 | −0.0378 |
| Lateral OA | 0.8719 | −0.0167 | −0.0255 |
| PF OA | 0.8556 | −0.0028 | −0.0657 |
| Effusion | 0.8636 | −0.0052 | −0.0486 |
| Synovitis | 0.8885 | −0.0027 | −0.0999 |
| Baker's | 0.9157 | −0.0081 | −0.3322 |
| Contusion | 0.9042 | −0.0010 | −0.1035 |
| Fracture | 0.9242 | +0.0006 | −0.2085 |

Over 18.4M visually valid tokens, the fine term's median norm is 61.6 against 30.0 for the
resampled deep term (mean 66.0 against 52.5), 62% of each token's combined norm on average, and
the two are nearly orthogonal (mean cosine 0.072). Magnitude is therefore not what sidelines the
fine term, and the projections already occupy separate subspaces.

**Interpretation (hypothesis, moderate confidence).** Optimization favors the pretrained deep
features, which lower the loss faster, and the readout largely ignores the fine subspace except
for OA, Baker's and effusion. This matches the greedy-learning pattern reported for multi-modal
networks (Wu et al., ICML 2022); its application to two branches of one backbone is an inference.
Deep-branch dropout and per-scale heads tested it (below).

## Deep-branch dropout (rejected)

`fusion-deepdrop25` added `model.deep_branch_dropout: 0.25` to `fusion-sum`: during training
only, each study lost the resampled deep term in all its triplets with probability 0.25, without
rescaling. Its configuration otherwise equals `fusion-sum`'s; it was trained from the uncommitted
tree on `ab49830` with the dropout code added, selected epoch 10 at ×1, and took 1,591 s and
21.1 GB. It scored 0.899699 at ×4 (0.895556 at ×1). The option was removed after this run.

| Group | vs `fusion-sum` [95%] | vs `k_05` [95%] |
|---|---|---|
| Macro | −0.0038 [−0.0069, −0.0007] | −0.0027 [−0.0060, +0.0006] |
| Meniscus | +0.0070 [+0.0013, +0.0133] | +0.0097 [+0.0025, +0.0174] |
| OA | +0.0014 [−0.0039, +0.0068] | −0.0001 [−0.0061, +0.0062] |
| Other eight | −0.0077 [−0.0119, −0.0038] | −0.0065 [−0.0107, −0.0021] |

Lateral Meniscus rose +0.0157 against `fusion-sum`; PF OA fell −0.0189, Synovitis −0.0127, MCL
−0.0102 and Effusion −0.0063 (`outputs/analysis/fusion-deepdrop25-vs-{fusion-sum,k05}/`). The
split reproduces truncation's (meniscus +0.0063, other eight −0.0073): forcing the fine features
into use trades labels rather than adding signal. Dropping the dominant branch at random is also
the weakest balancing method in the multimodal literature
([research report](../../research/multiscale-features-knee-mri.md)).

## Pair ensembles

Averaging two runs' fold-0 ×4 probabilities (point estimates, no intervals):

| Pair | Scales | Pair AUC | Over the pair mean |
|---|---|---:|---:|
| `k_05` + `oblique-x4` | 32 + 32 | 0.9054 | +0.0030 |
| `k_05` + `fusion-sum` | 32 + fused | 0.9059 | +0.0029 |
| `k_05` + `stride16` | 32 + 16 | 0.9044 | +0.0040 |
| `oblique-x4` + `stride16` | 32 + 16 | 0.9042 | +0.0039 |
| `fusion-sum` + `stride16` | fused + 16 | 0.9044 | +0.0036 |

Scale-diverse pairs gain about 0.001 more than same-scale pairs, so scale complementarity is
small next to run-to-run variance; a single model combining scales cannot collect the
variance-averaging part.

## Per-scale heads

`fusion-scaleheads` set `model.fusion_readout: scale_heads` on the same fused backbone: the deep
map kept its native 7×7 stride-32 tokens (49 per triplet) and the existing attention-MIL head, the
fine map's 196 stride-16 tokens got a separate head with its own embeddings, LayerNorm, attention,
anatomical strength and readout, and the logit was the sum of both. Training added BCE on each
head's own logits to BCE on the sum, weight 1 each; inference used the sum, with τ applied to both
heads. It was trained from the uncommitted tree on `ab49830` (`source_snapshot.zip`), selected
epoch 10 at ×1 (0.899284; epochs 9–10: 0.8991, 0.8993), and took 1,642 s and 20.5 GB with 14.72M
parameters. It scored 0.903023 at ×4.

| Group | vs `k_05` [95%] | vs `fusion-sum` [95%] |
|---|---|---|
| Macro | +0.0006 [−0.0031, +0.0040] | −0.0005 [−0.0041, +0.0030] |
| Meniscus | +0.0070 [+0.0004, +0.0142] | +0.0042 [−0.0011, +0.0101] |
| OA | +0.0012 [−0.0044, +0.0070] | +0.0027 [−0.0030, +0.0083] |
| Other eight | −0.0012 [−0.0063, +0.0035] | −0.0025 [−0.0076, +0.0022] |

Against `k_05`: Lateral Meniscus +0.0116, Synovitis +0.0097, Fracture +0.0086, ACL +0.0044;
Contusion −0.0090, PF OA −0.0086, MCL −0.0078, Baker's −0.0064
(`outputs/analysis/fusion-scaleheads-vs-{k05,fusion-sum}/`). A separately supervised fine head
kept the meniscus gain without deep-branch dropout's losses elsewhere, but macro stayed inside the
ensemble-implied ceiling of about +0.001. With a combined-loss BCE sum, the gradient norm ran at
about 1.0–1.4 against `gradient_clip_val: 1.0`, so clipping was frequent.
