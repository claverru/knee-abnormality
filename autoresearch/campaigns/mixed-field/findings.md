# Campaign findings

Decision summary. The full orchestrator log is `outputs/autoresearch/mixed-field/findings.md` (not tracked).

Incumbent: `contrast-split-112-140-20261001T153730Z-e26452f3`

- `field-160-single-logged`: reject, macro AUC delta -0.003445, adoption 80% CI [-0.005055, -0.001888], soft AUC delta -0.003079
- `field-120-single`: reject, macro AUC delta +0.001135, adoption 80% CI [-0.000364, +0.002623], soft AUC delta +0.000878
- `field-jitter`: reject, macro AUC delta +0.000511, adoption 80% CI [-0.000432, +0.001412], soft AUC delta -0.000194
- `mixed-field-140-160`: reject, macro AUC delta -0.001250, adoption 80% CI [-0.002318, -0.000271], soft AUC delta -0.000844
- `mixed-field-120-160`: reject, macro AUC delta +0.000584, adoption 80% CI [-0.000715, +0.001799], soft AUC delta +0.000265
- `mixed-field-readout-weights`: reject, macro AUC delta -0.000112, adoption 80% CI [-0.001570, +0.001272], soft AUC delta +0.000003
- `plane-weighted-fields`: reject, macro AUC delta +0.001079, adoption 80% CI [-0.000274, +0.002393], soft AUC delta +0.000521
- `extent-ruled-field`: reject, macro AUC delta -0.001671, adoption 80% CI [-0.002716, -0.000692], soft AUC delta -0.000774
- `field-backbones-routed`: reject, macro AUC delta -0.001159, adoption 80% CI [-0.002743, +0.000305], soft AUC delta -0.000892
- `contrast-split-112-140`: provisional_accept, macro AUC delta +0.001508, adoption 80% CI [+0.000423, +0.002566], soft AUC delta +0.001025
- `two-stream-paired`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
- `two-stream-paired`: invalid, macro AUC delta unavailable, adoption 80% CI unavailable
