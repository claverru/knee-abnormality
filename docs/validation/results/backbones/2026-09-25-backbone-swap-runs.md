# 2026-09-24/25 backbone swap runs: ConvNeXt-Atto and ReXNet-R

**Both backbones lost to ReXNet-200 on the recipe of their day. ConvNeXt-Atto lost clearly
(−0.0192) and is closed; ReXNet-R lost −0.0053 with a measured over-large learning rate and has a
planned follow-up at `backbone_lr` 8e-5.** Both ran from the `convnext-atto` worktree, whose
backbone mappings, `model.backbone_batch_multiple` padding and configs are uncommitted there and
not in `main`. The candidate list and rationale are in
[same-cost backbone candidates](../../../modeling/backbones/backbone-swaps.md#same-cost-backbone-candidates);
code requirements for a swap are in [swapping a backbone](../../../modeling/backbones/backbone-swaps.md).

## ConvNeXt-Atto

`outputs/training/convnext-atto-pad32`, 2026-09-24: `convnext_atto.d2_in1k` (3.4M parameters,
registered at origin 15.5, stride 32) against `knee-geometry-merged-2`, fold 0 of
`all_5fold_scanner_v1`, seed 638120755, `backbone_lr` 2e-4 unchanged, batch 8, 10 epochs,
`model.backbone_batch_multiple: 32`.

| | ×4 macro | Δ | 95% paired interval | Training s | Peak GiB |
|---|---:|---:|---|---:|---:|
| ReXNet-200 (`knee-geometry-merged-2`) | 0.900955 | — | — | 1,606 | 19.6 |
| ConvNeXt-Atto | 0.881816 | **−0.0192** | [−0.0254, −0.0134] | 801 | 4.3 |

- **Every family lost:** meniscus −0.0129, OA −0.0262 (PF OA −0.0454, Lateral OA −0.0348), the
  other eight −0.0191 (ACL −0.0300, Synovitis −0.0291, MCL −0.0261); only Baker's gained
  (+0.0061). Selected epoch 9 (`outputs/analysis/c/convnext-atto-vs-knee-geometry-merged-2/comparison.json`).
- **Half the time, a quarter of the memory,** but the AUC gap is too large for a cost trade.
- **Runtime finding:** its first contiguous-NCHW BF16 run left the RTX 3090 at 1–33% utilization
  with the autograd thread CPU-bound, because every distinct flattened triplet count paid a new
  cuDNN setup. Zero-padding each backbone batch to a multiple of 32
  (`model.backbone_batch_multiple`, dropping the padded outputs; incompatible with BatchNorm, whose
  statistics the padding would enter) bounded the number of shapes and produced this run.
- **Closed.** The padding option and the ConvNeXt mapping stay in the worktree; ConvNeXt-nano, a
  larger member of the family, remains a candidate that would reuse the mapping.

## ReXNet-R

`rexnetr-200`, 2026-09-25, config `configs/rexnetr-200.yaml` (removed) in the worktree:
`rexnetr_200.sw_in12k_ft_in1k` against `oa-detector-control`, fold 1, seed 1485198267, all rates
unchanged. ×4 macro 0.8952 against 0.9006, **−0.0053**, paired report-group bootstrap 95%
[−0.0098, −0.0010] (`outputs/analysis/r/rexnetr-200-vs-oa-detector-control/comparison.json`).
Training took 1607 s against 1639 s, with the same peak memory. The worktree's older OA-detector
port changes the preprocessing source hash; the comparison allowed that difference.

- **Per label:** the meniscus pair gained +0.0036 (lateral +0.0054, medial +0.0018), with MCL
  +0.0070 and Baker's +0.0042. OA lost 0.0059 (PF OA −0.0181), and synovitis −0.0224,
  effusion −0.0129 and fracture −0.0148 lost most.
- **Why tuning should help (measured mechanism):** the checkpoint's convolution weights are 2.5×
  smaller (RMS 0.0204 against 0.0508), so the same AdamW rate is a 2.5× larger relative step.
  Gradient norms ran 3–4× the control's (median about 2.3 against 0.6), so clipping at 1.0 fired on
  nearly every step. Convolution weights drifted 28% from pretraining against 16%. Validation
  trailed by 0.076 at epoch 0 and closed to −0.004 by epoch 8 as the cosine rate fell, still rising
  at the last epoch, while training loss stayed about 0.05 higher throughout. This is the evidence
  behind the [checkpoint weight-scale rule](../../../modeling/backbones/backbone-swaps.md#checkpoint-weight-scale).
- **Follow-up (not run):** `configs/rexnetr-200-lr8e-5.yaml` (removed), `backbone_lr` 8e-5
  (2e-4 × 0.0204/0.0508), nothing else changed; it now needs the current recipe's control
  (`scanner-confusion-control`, fold 4).
