# Stride-16 feature fusion (closed 2026-09-22)

**Closed without adoption.** Four fold-0 runs tested stride-16 ReXNet-200 features against the
stride-32 recipe `k_05`. Truncation lost; three fusion designs tied within about ±0.001 macro AUC
at ×4, below the +0.002 bar. The stride-16 features trade labels (menisci up, peripheral and
whole-joint labels down) rather than add signal, and the pair-ensemble ceiling for scale
complementarity is about +0.001. No code was kept; each run's `source_snapshot.zip` holds its
implementation. The [research report](../research/multiscale-features-knee-mri.md) reviews the
literature and the project's own precedents.

The investigation owned both late-stage ReXNet truncation and stride-16/stride-32 fusion, merged by
user decision on 2026-09-21 and started with the user on 2026-09-22.

## Runs

All runs: fold 0 of `all_5fold_scanner_v1`, combined labels, class-balanced BCE k=0.5, seed
2263738815, ten epochs, 224 px at 0.625 mm, ×1 checkpoint selection and ×4 τ 1 scoring. Training
is not deterministic, and the reference was the existing `k_05` run rather than a fresh control.
Deltas are against `k_05` with paired report-group 95% intervals.

| Run | Design | ×4 macro | Δ macro [95%] | Meniscus Δ | Other eight Δ | Training, memory |
|---|---|---:|---|---:|---:|---|
| `k_05` | stride 32, 49 tokens per triplet | 0.902469 | – | – | – | 1,584 s, 21.0 GB |
| `stride16` | truncate after `features[10]`, 196 tokens | 0.898229 | −0.0042 [−0.0083, −0.0002] | +0.0063 | −0.0073 | 1,379 s, 18.5 GB |
| `fusion-sum` | tokens `P16(F16)` + bilinear `P32(F32)` at stride-16 anchors | 0.903516 | +0.0011 [−0.0023, +0.0044] | +0.0027 | +0.0013 | 1,584 s, 21.1 GB |
| `fusion-deepdrop25` | `fusion-sum` + per-study deep-term dropout 0.25 in training | 0.899699 | −0.0027 [−0.0060, +0.0006] | +0.0097 | −0.0065 | 1,591 s, 21.1 GB |
| `fusion-scaleheads` | separate attention-MIL head per scale, summed logits, three equal BCE terms | 0.903023 | +0.0006 [−0.0031, +0.0040] | +0.0070 | −0.0012 | 1,642 s, 20.5 GB |

Details: [truncation result](../validation/results/2026-09-22-stride16-truncation-result.md),
[fusion, dropout, scale-heads and pair-ensemble results](../validation/results/2026-09-22-stride16-fusion-result.md).
Both fusion runs and the scale-heads run selected their last epoch.

## Findings

- **The fine branch is ignored under a token sum.** On `fusion-sum`, zeroing the fine term costs
  −0.0032 macro and leaves the menisci unchanged, although it has the larger norm and is nearly
  orthogonal to the deep term (cosine 0.07).
- **Forcing it into use trades labels.** Deep-branch dropout reproduced truncation's split
  (meniscus +0.0070 vs `fusion-sum`, other eight −0.0077). Per-scale heads kept the meniscus gain
  without that collapse but still tied on macro.
- **Scale complementarity is small.** Averaging two runs' predictions gains about +0.003 over the
  pair mean for same-scale pairs and about +0.004 for scale-diverse pairs, so a single model can
  collect only about +0.001 from combining scales.
- **An earlier precedent existed.** The medial-lateral campaign had already read `features.10`
  as local evidence and was rejected at −0.0022 with the path shrinking during training. It was
  the fifth secondary pathway the deep ReXNet path has sidelined; forcing one into use has never
  produced a macro gain.
- **Not checked:** all arms were scored at τ 1, not the deployed τ 0.75; the 196-token models put
  four times as many tokens into each attention softmax. Full inference time was not measured.

## Implementation notes

- **Cut point:** the last stride-16 ReXNet-200 block is `backbone.features[10]` (14×14×257 at
  224 px); `features[11:]` holds the stride-32 blocks and the 2560-channel expansion (9.58M
  parameters). Both mappings have origin `[0, 0]`: stride-16 anchors at 0, 16, …, 208 px and
  stride-32 at 0, 32, …, 192 px, verified by walking the blocks.
- **Resampling:** `grid_sample(align_corners=True, padding_mode="border")` at the stride-16
  anchors must run in float32 with autocast disabled; bf16 coordinates move the anchors by up to
  about 0.2 px.
- **Inference chunking:** a two-map backbone needs `use_fixed_backbone_chunks` to wrap a method
  returning both maps and concatenate each separately.
- **Recovering the code:** `outputs/training/stride16/source_snapshot.zip` (truncation) and
  `outputs/training/fusion-scaleheads/source_snapshot.zip` (fusion with `stride16_fusion` and
  `fusion_readout`, plus `scripts/adhoc/diagnose_fusion_branches.py`). The fusion checkpoints
  load only with that code; see [ensembling](../backlog/endgame.md#ensembling).

## Lessons

Price a new branch beside the dominant ReXNet path before training it: compute the ensemble-implied
ceiling from existing predictions, and design a per-scale supervised readout from the start. Here
the ceiling would have sized the line at about +0.001 before the first run.
