# Study 5: do the errors sit where the crop cut the finding

Asked by the user: the labels are caught well, so how do we know the cut regions cost anything?
871 fold-1 validation studies, the incumbent's saved ×4 predictions, geometry from one ×1 pass.
For each label, AUC in the third of studies with the least field around its region against the
third with the most; rank of positives against field (ρ+). 70 label-covariate pairs, not
corrected: 7 AUC intervals exclude zero against 3.5 expected by chance.

**Answer: the crops' in-plane field does not explain the errors.** No label shows a clear
dependence on how much field the crop leaves around its region. Two effects appear, and both are
about which slices are read, not how wide the crop is.

| Label (positives) | In-plane field | What does show |
|---|---|---|
| Effusion (244) | cannot tell; vanishes once axial reach is controlled | **axial upper reach:** AUC 0.872 / 0.933 / 0.948 by third, −0.076 (95% −0.128 to −0.023), ρ+ 0.22 |
| Synovitis (114) | cannot tell | **axial upper reach:** 0.848 / 0.883 / 0.914, −0.067 (−0.145 to −0.001), ρ+ 0.29 |
| MCL (47) | cannot tell | **sagittal medial reach:** 0.860 / 0.970 / 0.976, −0.116 (−0.190 to −0.042), ρ+ 0.51; missed 13 of 18 in the least third. Fragile at 47 positives |
| Baker's (114) | no (largest −0.015) | – |
| Medial Meniscus (339) | no; more coronal field goes with lower positive scores (ρ+ −0.24) | – |
| Lateral Meniscus (135) | no (+0.059, opposite sign) | sagittal lateral reach −0.053, interval includes zero |
| Medial OA, Lateral OA | no (−0.025, 0.000) | – |
| PF OA (290) | no; whole kneecap inside or not: −0.045 (−0.097 to +0.024) | – |
| ACL (131) | no in AUC | small knees lose sensitivity: 30 of 43 missed positives in the smallest third (OR 3.5, p 0.002) |
| Contusion, Fracture | no | – |

- **Axial upper reach:** the highest axial slice read sits a median 1.5 mm above the kneecap top;
  in the least third it is below −2.7 mm, so the pouch is not in any axial slice read. The effect
  survives controlling for knee size and in-plane field, and is about half the size of lacking
  the sagittal fluid series (−0.134 for Effusion).
- **In-plane field is knee size in disguise:** it correlates −0.91 to −0.94 with the size
  estimate and spans only ±4 mm, so this sample cannot separate a field effect from a size
  effect, and neither shows clearly.
- **The weak labels:** Synovitis is partly explained by axial reach. Nothing measured explains
  the errors of PF OA, Lateral Meniscus or Lateral OA.
- **The shift clamp** (177 of 871 studies) changes no label's AUC.

**Limits.** Landmarks are constants times the size estimate, not per-study measurements, so only
the ordering of studies is used. Slice reach is measured at ×1, scores at ×4 (the ×1 scores
agree: Effusion −0.070, MCL −0.126). Few positives for MCL, Fracture, Contusion and Lateral OA.
