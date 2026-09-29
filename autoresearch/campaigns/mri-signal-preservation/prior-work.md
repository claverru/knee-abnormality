# Source findings and measured precedents

`data/preprocessing/device.py` obtains bounds from selected unique-frame pixels inside valid rectangles, then rounds to uint8. `data/preprocessing/intensity.py` implements the corresponding CPU normalization. `modeling/model.py` later divides these values by 255 before augmentation. Foreground tissue support, continuous normalized intensities and deterministic receive-field estimation are absent from these inspected paths.

The table contains adjacent augmentation evidence, not measurements of the proposed deterministic transforms. Historical configuration files preserve the actual seeds.

| Mechanism | Setting | Run | Macro delta and 80% interval | Source |
|---|---|---|---|---|
| Bilateral smoothing augmentation | ReXNet; combined_v1; scanner_v1 fold 2; seed 2464012069 | bilateral-smoothing-augmentation-20260926T021107Z-ea5abbac | +0.002039; [+0.000184, +0.003854] | outputs/autoresearch/augmentation/experiments/bilateral-smoothing-augmentation-20260926T021107Z-ea5abbac |
| Noise-referenced smoothing | ReXNet; combined_v1; scanner_v1 fold 2; seed 1704794368 | noise-referenced-bilateral-smoothing-20260926T081854Z-5c6c595c | +0.001813; [-0.000342, +0.004002] | outputs/autoresearch/augmentation/experiments/noise-referenced-bilateral-smoothing-20260926T081854Z-5c6c595c |
| Synthetic multiplicative bias fields | ReXNet; combined_v1; scanner_v1 fold 2; seed 2464012069 | smooth-bias-field-augmentation-20260926T011026Z-b7fa0f09 | +0.000571; [-0.001339, +0.002414] | outputs/autoresearch/augmentation/experiments/smooth-bias-field-augmentation-20260926T011026Z-b7fa0f09 |
| Global nonlinear intensity augmentation | ReXNet; combined_v1; scanner_v1 fold 2; seed 1704794368 | gin-intensity-augmentation-v3-20260926T074639Z-0d98d2fa | +0.001332; [-0.001163, +0.003813] | outputs/autoresearch/augmentation/experiments/gin-intensity-augmentation-v3-20260926T074639Z-0d98d2fa |

`docs/validation/results/2026-09-28-campaign-rescoring.md` reports that smoothing's saved comparison becomes approximately +0.0008, 80% [-0.0009, +0.0027], on rebuild_state_v2. It also reports the external `half-stack-vs-smoothing-dsu` comparison at approximately +0.0011 originally and +0.0020 after re-scoring, with the latter interval [-0.0005, +0.0046]. That is a bundled main-recipe result, not attribution to a specific normalization transform; its seed is not supplied by that table.

## Current curve interpretation
The incumbent's final ×1 macro AUC is 0.9244 despite training loss falling to 2.2860. Feature-statistics, GIN, contrast mixing and smoothing activity remain nonzero. Their activation does not establish that preaugmentation normalization preserves useful signal or removes acquisition-specific nuisance. Vendor differences are confounded by pathology prevalence and case mix.

## Closed directions and ownership
The packet closes multiplicative-bias-field augmentation, k-space-resolution degradation, sharpening, feature-statistics uncertainty and the RN18 regularization family. They stay closed. Estimating and removing a field deterministically is a different operation from injecting random fields, and must not become another augmentation-amplitude retry. `rn18-learned-intensity-calibration` is already proposed in the intensity-normalization family; leave it to its owner. The present topic has no learned affine calibration, smoothing-strength retry or metadata adversary. The plane-readout pool seed is outside this scope. No direct measured result for these five deterministic signal mechanisms is supplied.

## Seed precedents

| Seed | Complexity | Structural | Closest precedent | Difference |
|---|---|---|---|---|
| sig-foreground-quantile-bounds | low | yes | src/knee_abnormality/data/preprocessing/intensity.py | Compute bounds on estimated tissue support instead of every valid-rectangle pixel, while applying the mapping to the same unchanged image support. |
| sig-continuous-normalization | low | no | src/knee_abnormality/data/preprocessing/device.py | Retain the exact normalization mapping and 0–255 scale but remove rounding and uint8 storage on both CPU and device paths. |
| sig-magnitude-power-debiasing | medium | yes | noise-referenced-bilateral-smoothing | Estimate and subtract background signal power deterministically before normalization instead of using noise to scale a stochastic bilateral-smoothing kernel. |
| sig-slice-coherent-local-histograms | medium | yes | https://www.sciencedirect.com/science/article/pii/S0734189X8780186X | Aggregate tile histograms across selected unique slices of one MRI acquisition and share their mappings through depth, with valid-support handling and the incumbent's fixed image geometry. |
| sig-coarse-receive-field-correction | high | yes | smooth-bias-field-augmentation | Estimate a field from the current acquisition and divide it out deterministically before normalization; do not add synthetic fields or retry their rejected augmentation strength. |
