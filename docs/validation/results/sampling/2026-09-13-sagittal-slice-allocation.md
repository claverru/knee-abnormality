# 2026-09-13 Sagittal slice allocation (`slice-budget`)

**Raising the sagittal slice allocation at the expense of axial inputs gave +0.000644 full-fold AUC
with worse BCE and gold AUC, and the recipe kept 18/15/15.** The allocation was tried once; no
repeatable benefit was established, and the runtime conditions of the two runs were not isolated.

The completed `slice-budget` run changed the per-contrast training limits from axial/coronal/sagittal
**18/15/15** to **9/15/24**, preserving the total budget. Its configuration otherwise matches
`crop140mm`, and their saved input identities match. Both use seed 42, Steven v4, `all_5fold_v3` fold 0,
224 pixels, 140 mm and 2× validation density; both selected epoch 8.

| Metric | crop140mm | slice-budget |
|---|---:|---:|
| Full validation AUC (904) | 0.886122 | 0.886766 |
| Gold AUC (11) | 0.926984 | 0.897586 |
| Soft-target BCE | 0.452961 | 0.456173 |

No public submission for this run appears in the recorded submission history.

Evidence: [saved configuration](../../../../outputs/training/slice-budget/effective_config.json),
[input identity](../../../../outputs/training/slice-budget/fold_0/input_identity.json),
[selected metrics](../../../../outputs/training/slice-budget/fold_0/epochs/metrics_epoch_007.json),
[reference metrics](../../../../outputs/training/crop140mm/fold_0/epochs/metrics_epoch_007.json) and
[timing](../../../../outputs/training/slice-budget/fold_0/timing.json). Checkpoints, predictions,
provenance and source snapshots remain in the original run directories. The artifacts were inspected
during the backlog audit on 2026-09-13; no run was repeated.
