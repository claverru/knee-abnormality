# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/student-distillation/findings.md` (not tracked).

Incumbent: `teacher-share-025-rerun-20261010T072609Z-a605253d`

- `teacher-share-050`: provisional_accept, macro AUC delta unavailable, adoption 80% CI unavailable
- `teacher-share-025`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `teacher-share-075`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `teacher-share-100`: reject, macro AUC delta -0.002813, adoption 80% CI [-0.003564, -0.002112], soft AUC delta -0.002615
- `teacher-share-025-rerun`: provisional_accept, macro AUC delta +0.001638, adoption 80% CI [+0.000900, +0.002395], soft AUC delta +0.000965
- `teacher-share-075-rerun`: reject, macro AUC delta -0.003247, adoption 80% CI [-0.004170, -0.002336], soft AUC delta -0.002221
- `teacher-share-per-label`: reject, macro AUC delta -0.000365, adoption 80% CI [-0.000931, +0.000210], soft AUC delta -0.000107
- `reranked-fold-teachers`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `recalibrated-fold-teachers`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `recalibrated-fold-teachers-rerun`: reject, macro AUC delta +0.000174, adoption 80% CI [-0.000344, +0.000721], soft AUC delta -0.000166
- `share-from-plane-agreement`: reject, macro AUC delta -0.000174, adoption 80% CI [-0.000712, +0.000339], soft AUC delta -0.000330
- `reranked-fold-teachers-rerun`: reject, macro AUC delta -0.000394, adoption 80% CI [-0.000878, +0.000066], soft AUC delta -0.000312
- `target-per-plane`: reject, macro AUC delta -0.000625, adoption 80% CI [-0.001315, +0.000092], soft AUC delta -0.000249
- `teacher-beyond-repeat-noise`: reject, macro AUC delta -0.000170, adoption 80% CI [-0.000611, +0.000277], soft AUC delta +0.000279
