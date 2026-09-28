# 2026-09-27 single-plane inference on the control

**Given one plane's series only, `scanner-confusion-control` loses 0.039 (sagittal), 0.043
(coronal) and 0.061 (axial) ×4 macro AUC on its own fold, against 0.9097 with every series.** No
single plane is enough, and each label leans on a different plane: ACL on sagittal, MCL on coronal,
Effusion and Baker's on axial. The menisci lose at least 0.04 from any single plane.
No retraining: the model saw single-plane studies only through series dropout.

## Setup

- **Model:** `outputs/training/scanner-confusion-control`, selected checkpoint (epoch 9), fold 4 of
  `all_5fold_scanner_v1`, combined labels; ×4, attention temperature 1, which reproduces the saved
  0.909733.
- **Intervention:** every slot of the other two planes emptied before loading, so geometry, study
  centre and budgets come from the kept plane alone, as for a study that only has that plane. All
  869 validation studies hold a series in each plane.
- **Budget:** the missing types' budget moves to the kept plane's slots, but the kept series'
  length caps it: the first three studies drew 16, 36 and 48 axial triplets against about 110 with
  every series. Each run predicted in 22–26 s.
- **Reference:** the run's saved ×4 validation predictions on the same studies. Intervals are a
  paired study bootstrap (2,000 draws, seed 20260914), not the report-group bootstrap.
- **Script and outputs:** [`score_axial_only.py`](../../../scripts/adhoc/score_axial_only.py)
  (`--plane`) at commit `338431d` plus the script;
  `outputs/analysis/{axial,coronal,sagittal}-only-scanner-confusion-control-20260927/`
  (`results.json`, `axial_only_predictions.csv`, which holds the kept plane's predictions).

## Macro

| Kept plane | Macro AUC | Δ | 95% interval | Soft macro AUC |
|---|---:|---:|---|---:|
| All series | 0.9097 | — | — | 0.7959 |
| Sagittal | 0.8709 | −0.0388 | [−0.0467, −0.0311] | 0.7682 |
| Coronal | 0.8672 | −0.0425 | [−0.0484, −0.0371] | 0.7606 |
| Axial | 0.8487 | −0.0611 | [−0.0693, −0.0533] | 0.7531 |

## Per label

Δ against all series; the best single plane per label is in bold.

| Label | All series | Axial | Coronal | Sagittal | Positives |
|---|---:|---:|---:|---:|---:|
| ACL | 0.9591 | −0.0842 | −0.0484 | **+0.0027** | 133 |
| MCL | 0.9180 | −0.0510 | **−0.0079** | −0.1349 | 46 |
| Medial Meniscus | 0.9188 | −0.1530 | **−0.0401** | −0.0470 | 360 |
| Lateral Meniscus | 0.9079 | −0.2150 | −0.0905 | **−0.0860** | 139 |
| Medial OA | 0.9065 | −0.0515 | **−0.0208** | −0.0280 | 327 |
| Lateral OA | 0.8741 | −0.0684 | −0.0459 | **−0.0178** | 240 |
| PF OA | 0.8795 | −0.0378 | −0.0862 | **−0.0267** | 279 |
| Effusion | 0.8806 | **−0.0055** | −0.0375 | −0.0223 | 521 |
| Synovitis | 0.9058 | −0.0167 | −0.0346 | **−0.0169** | 114 |
| Baker's | 0.9369 | **−0.0013** | −0.0808 | −0.0319 | 217 |
| Contusion | 0.9075 | −0.0340 | **−0.0133** | −0.0399 | 153 |
| Fracture | 0.9221 | −0.0142 | **−0.0045** | −0.0168 | 57 |

PF OA is best from sagittal (−0.0267), narrowly ahead of axial (−0.0378); Synovitis ties between
axial and sagittal.

## Reading

- Each loss mixes two things: removing the other planes' evidence, and fewer triplets per study.
  These runs do not separate them.
- The per-label pattern matches radiological reading: ACL on sagittal, MCL on coronal, effusion and
  Baker's cysts on axial.
- The menisci need more than one plane: their best single plane still loses 0.040 (medial,
  coronal) and 0.086 (lateral, sagittal).
