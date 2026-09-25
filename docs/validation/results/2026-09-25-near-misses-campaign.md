# 2026-09-25 near-misses campaign

The campaign found one adoption, random attention-token dropout, but it did not carry to the
public board, and its dropout was too weak to matter. A near-miss did carry: the
three-compartment OA detector. The user adopted it on main.

## Setup

- **Campaign:** [near-misses](../../../autoresearch/campaigns/near-misses/INDEX.md), on Codex.
  It ran 15 user-ordered seeds, then four researcher ideas. The user stopped it on 2026-09-25.
- **Reference:** `knee-geometry-merged-2`, seed 638120755, ×4 macro 0.900955.
- **Rule:** the [shared adoption rule](../../../autoresearch/protocol.md#shared-evaluation-and-adoption).
  Soft AUC is a diagnostic.

## Results

Hard and soft Δ are macro AUC at ×4 against the incumbent at the time.

| # | Idea | Hard Δ | Soft Δ | Note |
|---|---|---:|---:|---|
| 1 | Shared + residual meniscus head | −0.0010 | −0.0005 | |
| 2 | Folded detector, isolated | −0.0002 | +0.0003 | |
| 3 | #1 + #2 | +0.0005 | +0.0008 | |
| 4 | Side-split meniscus readout | +0.0009 | +0.0003 | Medial Meniscus +0.0022, Lateral −0.0041 |
| 5 | Early-learning regularization | +0.0002 | +0.0004 | |
| 6 | Side-normalized 3D position | +0.0008 | −0.0006 | PF OA +0.0061 |
| 7 | PF OA front-back bias | −0.0007 | −0.0021 | PF OA −0.0053 |
| 8 | **Three-compartment OA detector** | **+0.0013** | +0.0014 | PF OA **+0.0102** (95% above zero); medial and lateral OA flat |
| 9 | **Token dropout, random** | **+0.0025** | +0.0019 | Adopted, 80% [+0.0005, +0.0044]; fresh-seed control 0.905246 |
| 10 | Token dropout, top-attention | −0.0007 | +0.0003 | |
| 11 | Plane/contrast readouts | −0.0006 | −0.0004 | |
| 12 | Any-OA auxiliary | −0.0021 | −0.0011 | PF OA +0.0118, Lateral OA −0.0081 |
| 13 | 3D position + slice orientation | −0.0017 | +0.0011 | PF OA +0.0156 |
| 14 | Plane-negative half-knee MIL | −0.0017 | −0.0015 | Lateral Meniscus −0.0060 |
| 15 | Attention-coupled meniscus MIL | −0.0016 | 0.0000 | |

Seeds 1–9 were measured against 0.900955 and later runs against the fresh-seed control, 0.905246.
The researcher's four ideas were all rejected: an OA geometry attention residual (−0.0003, PF OA
+0.0102), OA detector plus union supervision (−0.0029), spatial-cluster token dropout (−0.0002)
and OA acquisition-mass routing (−0.0014, PF OA +0.0115).

## Findings

- **PF OA:** five different mechanisms that give OA knee-wide context raised PF OA by 0.010 to
  0.016, each with a 95% interval above zero. All but the three-compartment detector cost other
  labels, so none raised the macro AUC.
- **Public check** (both at the reference's seed, paired with S55 at 0.940):
  - S58, the three-compartment detector: **0.944**.
  - S59, the adopted token dropout: **0.939**.
  - The public board ranked them in the opposite order to fold 0.
- **Token dropout was too weak:** rate 0.05 with a cap of 4 tokens meant exactly 4 masked tokens
  per study-label out of about 1,312 (0.3%). Its fold-0 gain is most likely noise. The seed spec
  asked for a "conservative" rate and cap without stating a target fraction, and the review did
  not check the realized fraction.
- **OA detector usage:** by epoch 9 the learned scales were medial 0.040, lateral 0.050 and PF
  0.017, with mean output shifts of 0.085, 0.082 and 0.035 logits. The tibiofemoral labels shape
  the detector. The PF OA gain is larger than its direct shift suggests, so it probably comes
  through the shared token features. The three OA labels' class-balance weights are close
  (w+ 1.17–1.39), so the weights do not drive the imbalance.
- **Meniscus sharing:** no gain on the re-seeded control. The side-split readout helped the
  frequent medial side and hurt the rare lateral side.

## Follow-up on main

The OA-detector control trained on fold 1 at fresh seed 1485198267 (×4 τ 1 0.900568, ×1 0.895202;
public 0.938, S60). Token dropout at 5% of tokens, uncapped (realized 4.95%), at the same fold
and seed was **rejected**:

- ×4 τ 1: 0.899964, −0.0006 (80% [−0.0025, +0.0015]); ×1: +0.0012; it peaked at epoch 7, the
  control at epoch 9.
- It lost at every inference temperature, least at τ 1:

| τ (×4) | Control | Dropout | Δ | 80% interval | Soft Δ |
|---:|---:|---:|---:|---|---:|
| 0.50 | 0.899640 | 0.896243 | −0.0034 | [−0.0060, −0.0007] | −0.0021 |
| 0.60 | 0.901832 | 0.899232 | −0.0026 | [−0.0050, −0.0000] | −0.0018 |
| 0.70 | 0.902633 | 0.900664 | −0.0020 | [−0.0041, +0.0004] | −0.0014 |
| 0.75 | 0.902624 | 0.900956 | −0.0017 | [−0.0037, +0.0007] | −0.0012 |
| 0.80 | 0.902429 | 0.900976 | −0.0015 | [−0.0034, +0.0008] | −0.0011 |
| 0.90 | 0.901820 | 0.900718 | −0.0011 | [−0.0030, +0.0011] | −0.0008 |
| 1.00 | 0.900568 | 0.899964 | −0.0006 | [−0.0025, +0.0015] | −0.0005 |

Each model's best τ: control 0.9026 (τ 0.70–0.75), dropout 0.9010 (τ 0.80), −0.0017. The ×1 gain
and the small ×4 τ 1 loss both reflect sharper attention being worth less after dropout; τ < 1
recovers more for the control. τ 0.75 remains near-optimal for the control.
