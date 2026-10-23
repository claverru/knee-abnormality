# fastMRI joint training campaign: prior work

Fold 3 of `all_5fold_scanner_v2`, seed 1110484336, one run per design. Setup and the full record
of the first run: [fastMRI joint training](../../../docs/validation/results/teacher-student/INDEX.md).

## The reference

Fold 3 of `outputs/training/rn50-joint-fastmri-fold-3`: main's `configs/refined-resnet50.yaml`
at `c1d973b` with `extra_training` set. Every Kaggle batch is followed by a fastMRI batch
labelled by teacher 3; one optimizer step covers the pair at equal weight; L2-SP is applied on
the Kaggle batch only; class weights come from the Kaggle rows only. **×1 0.926211 (epoch 10 of
10), ×4 0.930745, 4,530 s of training** (about 450 s per epoch) on the rented RTX 5090, peak GPU
memory 16.6 GB. **Public: S148 0.962 at ×4 τ 0.9, the best public score on 2026-10-12** (S140, five teachers, 0.961;
S143, a single ResNet-50 on fold 1, 0.960). The plain fold 3 control was not submitted.

The plain control is fold 3 of `outputs/training/rn50-main-0bd598f-folds-3-4`, same machine,
recipe and seed without fastMRI: ×1 0.925655, ×4 0.931023, 2,247 s.

| | Joint | Control | Δ |
|---|---:|---:|---:|
| ×1 macro AUC | 0.926211 | 0.925655 | +0.000556 |
| ×4 macro AUC | 0.930745 | 0.931023 | −0.000278 |
| ×4 soft macro AUC | 0.787786 | 0.787299 | +0.000487 |
| ×4 soft BCE | 0.4021 | 0.4006 | +0.0015 |

×1 macro AUC by epoch, joint minus control: +0.0082, +0.0063, +0.0018, +0.0017, +0.0033,
+0.0018, +0.0006, +0.0008, +0.0002, +0.0006.

Per label at ×4, joint against control, largest gain first:

| Label | Joint | Control | Δ |
|---|---:|---:|---:|
| Lateral Meniscus | 0.9147 | 0.9114 | +0.0032 |
| Baker's | 0.9466 | 0.9435 | +0.0031 |
| Synovitis | 0.8932 | 0.8905 | +0.0028 |
| Contusion | 0.9527 | 0.9506 | +0.0021 |
| Lateral OA | 0.9172 | 0.9157 | +0.0014 |
| Medial OA | 0.9377 | 0.9369 | +0.0008 |
| ACL | 0.9526 | 0.9542 | −0.0016 |
| Medial Meniscus | 0.9446 | 0.9462 | −0.0016 |
| Fracture | 0.9315 | 0.9339 | −0.0024 |
| Effusion | 0.9284 | 0.9308 | −0.0024 |
| MCL | 0.9481 | 0.9507 | −0.0027 |
| PF OA | 0.9018 | 0.9078 | −0.0060 |

One point of AUC on one label is 0.0008 macro.

## The pseudo-labels

Five ResNet-50 fold models of main's recipe, ×4, τ 0.9, mirror TTA, over 10,009 fastMRI studies;
9,828 are trained on. Mean probability and mean spread between the five teachers, per label:

| Label | Mean probability | Mean spread |
|---|---:|---:|
| PF OA | 0.481 | 0.029 |
| Effusion | 0.408 | 0.028 |
| Medial Meniscus | 0.369 | 0.023 |
| Synovitis | 0.352 | 0.027 |
| Medial OA | 0.348 | 0.024 |
| Lateral OA | 0.262 | 0.026 |
| Lateral Meniscus | 0.255 | 0.025 |
| Baker's | 0.226 | 0.019 |
| Contusion | 0.195 | 0.024 |
| Fracture | 0.190 | 0.025 |
| ACL | 0.156 | 0.020 |
| MCL | 0.120 | 0.018 |

PF OA has the mean closest to 0.5 and the largest spread, and is the largest loss of the first
run. The link is a reading, not a tested cause.

fastMRI differs from Kaggle in population: a clinical five-series protocol without T1-weighted
series, fields of 140 to 150 mm, and a second part that is 93% 1.5 T without side tags.

## L2-SP

Adopted on 2026-09-30 on the ResNet-18 for unseen sites
([result](../../../docs/validation/results/validation-and-ensembles/2026-09-30-rn18-l2sp-no-confusion.md),
[campaign](../site-generalization/findings.md)): held-out sites +0.0009 to +0.0023 at ×4 over
four measurements, seen sites and fold validation between −0.0010 and +0.0003; public 0.943
against 0.941 on the sites-out split and 0.943 against 0.943 on full data. Halving either
strength was level on fold 0. **It was never tested on the ResNet-50**, which inherited
`anchor_alpha_early: 0.01` and `anchor_alpha_late: 0.001`. In the joint run the penalty is
added once per optimizer step while the step sums the gradients of two batches, so its pull
relative to the task gradient is lower than in the plain recipe (a derivation from the code, not
a measurement).

## Precedents

| Mechanism | Setting | Macro Δ | Source |
|---|---|---:|---|
| Soft-prevalence class weights | ResNet-50 fold 2, plain recipe | ×1 +0.0013, ×4 +0.0005 | targets-and-inputs campaign, run notes (worktree `knee-abnormality-targets-and-inputs`) |
| Class-weight exponent 0.5 | fold 0 | lost locally, won publicly | targets-and-inputs campaign, prior work (same worktree) |
| OOF distillation, 0.25 teacher share | ResNet-50 fold 1 | ×4 0.935809, S143 0.960 | [student distillation](../student-distillation/INDEX.md) |

## Dropped before the campaign

- The mean of the other four teachers as the fastMRI label: those teachers trained on the fold's
  validation studies.
- A per-label exclusion of PF OA from the fastMRI loss: it would be chosen on the fold's
  validation result.
