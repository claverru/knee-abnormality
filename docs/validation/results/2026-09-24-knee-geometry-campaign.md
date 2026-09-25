# 2026-09-24 knee-geometry campaign

**One adoption in 18 decisions: `meniscus-half-knee-mil`.** A fitted medial-lateral midline, a
meniscus-only own-side attention mask and a training-only half-knee MIL loss gained +0.003074 at
×4 on fold 0 and scored 0.944 public (S54) against 0.942 for the reference (S48). Nothing after
it cleared the rule; most later candidates lost to it.

## Setup

- **Campaign:** [knee-geometry](../../../autoresearch/campaigns/knee-geometry/INDEX.md), on Codex,
  2026-09-23 to 2026-09-24; stopped by the user.
- **Reference:** `k05-clean` (S48), seed 2263738815, ×4 hard 0.902760, soft 0.798458; fold 0 of
  `all_5fold_scanner_v1`, `combined_v1` labels, 905 validation studies in 851 report groups; run
  ceiling 1,846 s.
- **Rule:** the [shared adoption rule](../../../autoresearch/protocol.md#shared-evaluation-and-adoption).
  Soft AUC is a diagnostic.
- **Restart:** the first launch was stopped after calibration and kept as
  `outputs/autoresearch/knee-geometry-aborted-20260924`, so that `inference/engine.py` could be
  made editable: the final ×4 prediction moves only `MODEL_BATCH_KEYS` to the GPU.

## Results

Hard and soft Δ are ×4 macro AUC against the incumbent at the time: `k05-clean` (0.902760) for
rows 1–3, then `meniscus-half-knee-mil` (0.905834).

| # | Idea | Hard Δ | 80% interval | Soft Δ | Note |
|---|---|---:|---|---:|---|
| 1 | Geometry bundle: robust centre, bilateral split, midline at the fitted centre | +0.0016 | [−0.0005, +0.0037] | +0.0012 | Meniscus +0.0037, OA −0.0031 |
| 2 | Sagittal-normal knee frame (on the bundle) | +0.0006 | [−0.0013, +0.0027] | +0.0013 | |
| 3 | **Half-knee MIL** | **+0.0031** | [+0.0015, +0.0048] | +0.0019 | Adopted |
| 4 | Meniscus-only side mask on the fitted midline | −0.0018 | [−0.0034, −0.0002] | −0.0008 | |
| 5 | Shared plus residual meniscus head | −0.0006 | [−0.0021, +0.0007] | −0.0002 | |
| 6 | Coronal condylar frame | −0.0022 | [−0.0039, −0.0006] | −0.0005 | |
| 7 | Study-anchored through-plane position | −0.0013 | [−0.0029, +0.0003] | +0.0002 | |
| 8 | Position-free meniscus auxiliary | +0.0000 | [−0.0017, +0.0018] | +0.0001 | |
| 9 | Detector-ranked meniscus pooling | −0.0018 | [−0.0033, −0.0002] | −0.0013 | |
| 10 | OA own-half readout supervision | −0.0000 | [−0.0016, +0.0014] | −0.0000 | |
| 11 | Axis-consensus meniscus membership | −0.0030 | [−0.0049, −0.0013] | −0.0011 | |
| 12 | Meniscus family pooling normalization | −0.0023 | [−0.0039, −0.0008] | −0.0019 | |
| 13 | Acquisition-local anatomical position | −0.0026 | [−0.0045, −0.0007] | −0.0009 | |
| 14 | Folded medial position for the meniscus detector | −0.0011 | [−0.0027, +0.0003] | −0.0002 | |
| 15 | OA own-half feature dispersion | −0.0013 | [−0.0029, +0.0001] | −0.0012 | |
| 16 | Fractional-footprint meniscus mask | −0.0034 | [−0.0052, −0.0017] | −0.0021 | |
| 17 | Folded auxiliary with gradient isolation | −0.0004 | [−0.0020, +0.0011] | −0.0005 | Meniscus +0.0025, OA −0.0020 |

The 18th decision, attention-coupled meniscus MIL, was invalid: training crashed after 12.5 s.
Six ideas ended blocked without a trained run: the meniscus-only mask on the old midline, the OA
own-side plus knee-channel readout, the curve-density sampler, the side-normalized 3D position,
fitted-midline-only and within-acquisition geometry pooling. The last run, `meniscus-core-band-contrast-readout`, trained but was
not compared when the user stopped the campaign.

## Adopted model

- **Change:** the midline at the fitted study centre, a meniscus-only own-side attention mask (OA
  stays whole-knee) and a half-knee MIL loss on a shared tear detector during training only.
- **Families:** meniscus 0.915157 → 0.919094, OA 0.886309 → 0.888926, other eight
  0.903774 → 0.906746. Medial and Lateral Meniscus +0.0038 and +0.0040; the largest per-label
  gains are Fracture +0.0107 (the only 95% interval above zero, [+0.0009, +0.0226]), Baker's
  +0.0064 and Contusion +0.0060. Its lead over the later candidates sits mostly in those three
  labels, which a re-seed also moves.
- **Public:** S54 0.944 against S48 0.942.

## Findings

- **Geometry bundle strata:** 28 rescued-centre studies moved macro AUC about −0.00031, the two
  bilateral studies are too few to judge, and the 121 studies whose midline moved more than 5 mm
  gained +0.00820. That drove the midline into the adopted model.
- **Nothing beat the half-knee model:** the best later candidate tied it (+0.000001), and seven
  had the whole 80% interval below zero. Against `k05-clean`,
  16 of the 17 compared candidates score above it (only the fractional-footprint mask falls
  below), so most losses measure the gap to the incumbent rather than harm to the base recipe.
- **Seed:** the adopted code at a fresh seed, 638120755, scored 0.899547, −0.0063 from the
  campaign's 0.905834 ([seed noise](../validation-strategy.md#controlled-comparisons)).

## After the campaign

The user adopted the model on main outside the rule, added the robust centre and per-knee geometry
for bilateral studies, and retrained the control at seed 638120755 as `knee-geometry-merged-2`:
×4 0.900955, public 0.940 (S55). The same model without the robust centre and bilateral rule
scored 0.899547 (S56, 0.940) ([current model](../../modeling/current-model.md)). Several
untried or near-miss ideas (side-normalized 3D position, plane-negative and attention-coupled
half-knee MIL) became seeds of the [near-misses campaign](2026-09-25-near-misses-campaign.md).
