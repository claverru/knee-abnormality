# 2026-10-04 Refined-RadImageNet campaign

**The checkpoint's own input normalization is the one adoption: `data.mean` 0.3162 and `data.std`
0.3213 in place of the ImageNet values, +0.0006 on the ResNet-18.** The `refined-radimagenet`
autoresearch campaign tested how to get more out of the Refined-RadImageNet pretraining on the
ResNet-18 recipe: the checkpoint's input statistics, the authors' mask-trained weights and their
policy-gradient mask in our fine-tuning, the pretrained 165-class head with its 18 knee classes,
and adjacent slices through Cross-D and ACS convolutions. It scored 18 runs in the campaign and
two by hand; none but the normalization met the adoption rule. The user stopped it on 2026-10-04
at 20:43 CEST. The normalization is in `configs/training.yaml` and `configs/refined-resnet50.yaml`
since 2026-10-05 ([adoption](../../modeling/current-model.md#adoption-history)); the ResNet-50
config takes it without a run of its own (user decision).

Fold 0 of `all_5fold_scanner_v2` (912 studies), `rebuild_state_v2` labels, seed 1533697213, ×4
predictions, paired report-group bootstrap with 2,000 draws. Adoption rule: the lower end of the
central 80% interval above zero
([protocol](../../../autoresearch/protocol.md#shared-evaluation-and-adoption)). Every result is
one run at one seed. Yardstick: two runs of one recipe on this fold differ by about 0.0003 at ×4
(`acs-gated-stem`, whose mechanism stayed inert, landed +0.00003 from the incumbent; the
three slice-teacher durations lie within 0.0005 of each other).

- **Campaign record** (contract, prior work, seeds, decision summary):
  [refined-radimagenet campaign](../../../autoresearch/campaigns/refined-radimagenet/INDEX.md).
- **Per-run details** (per-label deltas, curves, mechanism readings, keeper repairs):
  [run notes](../../../autoresearch/campaigns/refined-radimagenet/run-notes.md).
- **Code:** the campaign's build (weight table for the mask-trained and Cross-D checkpoints, the
  classifier loader, `modeling/crossd.py`) is in the `refined-radimagenet` worktree and is not on
  main; each run's frozen source is under `outputs/autoresearch/refined-radimagenet/experiments/<run>/source/`.
- **Artifacts:** `outputs/autoresearch/refined-radimagenet/` (state, and per run the frozen
  source, `changes.diff`, config, logs, predictions and comparison). Checkpoints were removed on
  2026-10-04 and 2026-10-05 except the incumbent's
  ([cleanup log](../../architecture/configs-and-experiment-artifacts.md)).

## Runs

Reference for run 1: `rn18-view-label-bias` (×4 0.932477, ×1 0.929026, 0.951 public as S101).
From run 2 on the reference is run 1 (×4 0.933058, ×1 0.929948). Runs are in the order they were
scored.

| # | Run | What changes | Δ ×4 | 80% interval | Verdict |
|---|---|---|---:|---|---|
| 1 | `radimagenet-input-statistics` | Mean 0.3162, std 0.3213 | **+0.000581** | [+0.000182, +0.001016] | **adopted** |
| 2 | `rn10t-policy-weights` | Heated ResNet-10t checkpoint (a ResNet-10t against the ResNet-18 incumbent; a measurement) | −0.004281 | [−0.006272, −0.002393] | reject |
| 3 | `rn10t-policy-ft-weights` | Fine-tuned mask-trained ResNet-10t checkpoint (same) | −0.004387 | [−0.006357, −0.002488] | reject |
| 4 | `backbone-eval-pretrained-statistics` | Backbone in evaluation mode, BatchNorm on the pretrained running statistics | −0.003119 | [−0.004573, −0.001732] | reject |
| 5 | `policy-mask-rn10t` | Policy-gradient input mask, all epochs | −0.002292 | [−0.003594, −0.001087] | reject |
| 6 | `policy-mask-two-stage` | The mask in two stages | −0.001733 | [−0.002623, −0.000827] | reject |
| 7 | `policy-mask-pretrained-policy` | The mask from a pretrained policy network | −0.002590 | [−0.003837, −0.001417] | reject |
| 8 | `random-smooth-mask` | Random smooth mask | −0.002325 | [−0.003573, −0.001116] | reject |
| 9 | `knee-logit-token-features` | The 18 knee-class logits as token features | −0.001496 | [−0.002371, −0.000728] | reject |
| 10 | `knee-classifier-readout` | Knee-class readout | +0.000048 | [−0.000351, +0.000442] | reject |
| 11 | `pretrained-slice-teacher` | Frozen pretrained network as slice teacher, all epochs | +0.000333 | [−0.000588, +0.001210] | reject |
| 12 | `knee-logit-attention-prior` | Knee logits as an attention prior | +0.000297 | [−0.000114, +0.000679] | reject |
| 13 | `crossd-resnet18-drop-in` | Cross-D ResNet-18 per slice | — | — | invalid (time) |
| 14 | `crossd-3d-stem` | Cross-D 3D stem over adjacent slices | — | — | invalid (faulty seed) |
| 15 | `crossd-static-kernels` | Cross-D static shifted kernels as a plain ResNet-18 | −0.003903 | [−0.005739, −0.002093] | reject |
| 16 | `crossd-gated-depth-stem` | Gated three-slice stem | −0.004200 | [−0.006117, −0.002313] | reject |
| 17 | `acs-stem-layer1`, `acs-gated-stem-block1` | ACS in the stem and layer1; gated form in the stem and first block | — | — | invalid (time) |
| 18 | `acs-gated-stem` | Gated ACS in the stem only | +0.000027 | [−0.000328, +0.000374] | reject (gate stayed at zero) |
| 19 | `knee-normal-contrast-attention` | Researcher: knee-normal contrast in attention | +0.000095 | [−0.000229, +0.000396] | reject |
| 20 | `acs-gated-stem-signed-gate` | Gated ACS stem with an unclamped gate | tie | — | not recorded (campaign stopped) |
| 21 | Slice teacher released after epoch 6 (manual) | Teacher for epochs 1–6 only | +0.000378 | 95% [−0.0009, +0.0016] | not adopted |
| 22 | Slice teacher released after epoch 1 (manual) | Teacher for epoch 1 only | −0.000083 | 95% [−0.0013, +0.0010] | not adopted |

## Findings

- **Normalization (run 1).** Small and spread over the labels: ten of twelve up, ACL +0.0028,
  Medial OA +0.0017, MCL +0.0016, Fracture −0.0024; level with the reference for four epochs
  and ahead from epoch 7. The frozen network reads our slices very differently under the two
  normalizations (0.088 of the pretrained head's probability on knee classes under the ImageNet
  values, 0.978 under its own), but BatchNorm in training mode absorbs most of an input shift,
  so the fine-tuned gain is small.
- **Mask-trained weights and masks (runs 2–8).** On the ResNet-10t the two mask-trained
  checkpoints trained identically; against the ResNet-10t reference `rn10t-view-label-bias`
  (×4 0.927543) they scored 0.928777 and 0.928671, +0.0012 and +0.0011, with ligaments and
  menisci up and the OA labels down. Every mask in our own fine-tuning lost
  0.0017 to 0.0026; the policy network never learned (Beta parameters at 1.000), so the policy
  runs were random masks.
- **Pretrained head (runs 9–12, 19).** The knee classes as features lost 0.0015; as a readout,
  an attention prior or a slice teacher they are ties, between +0.00005 and +0.0003.
- **Slice teacher (runs 11, 21, 22).** A head start of about +0.005 at ×1 after the first epoch
  that fades whether or not the teacher stays; ×4 +0.0003 to +0.0004 with the teacher for six or
  ten epochs, inside the run-to-run spread. Lateral OA is the one label that moves consistently
  (+0.0031, +0.0065, +0.0067 for one, six and ten epochs).
- **Adjacent slices (runs 13–18, 20).** Cross-D kernels and the gated depth stem lost about
  0.004; ACS convolutions were over the time bound, and the gated stem that fitted the bound
  was inert with a clamped gate and a tie with an unclamped one.

## What stayed open

- **Heated ResNet-50 weights.** Two ResNet-50 runs on the mask-trained checkpoint were started
  and stopped by the user before completion; no result.
- **The follow-up campaign** on the input preparation (`pretraining-input`, on a ResNet-10t
  reference with this normalization) ran on 2026-10-04 and 2026-10-05 in the same worktree;
  its record is not on main yet ([ongoing](../../ongoing/next.md)).
