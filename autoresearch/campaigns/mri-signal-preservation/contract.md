# Objective
Improve all-label macro AUC through deterministic acquisition-local signal processing before augmentation. The inspected source computes intensity bounds from every pixel inside valid rectangles and rounds normalized values to uint8 on both CPU and device paths. Padding is excluded from the bounds, but air inside valid rectangles is not. The magnitude of clinically relevant quantization loss, residual shading and noise-floor bias has not been measured.

## Permitted mechanisms
Permit continuous normalized intensities, foreground-supported quantile bounds, supported magnitude-noise power correction, slice-coherent contrast-limited local histogram mapping, and coarse acquisition-local receive-field estimation. [Gudbjartsson and Patz](https://pmc.ncbi.nlm.nih.gov/articles/PMC2254141/) establish magnitude-noise bias under their acquisition assumptions. [Pizer et al.](https://www.sciencedirect.com/science/article/pii/S0734189X8780186X) motivate contrast-limited adaptive histogram processing. [Tustison et al.](https://iacl.ece.jhu.edu/proceedings/isbi2010/pdfs/0000708.pdf) motivate estimation of smooth intensity nonuniformity. These are mechanism precedents, not evidence of a knee-classification gain or universal validity of a Rician model.

## Fixed behavior and exclusions
Retain the live incumbent's backbone, readout, objectives, label source, folds, cohorts, 36-triplet recipe, 224-pixel images, 0.625-mm spacing, sampling, geometry and augmentation stack. No cropping, resampling change, extra frames, learned vendor-specific transform, new label processing or transductive fitting. Keep original selected-frame and valid-pixel support accounting. Treat only the existing unique frames once when estimating acquisition statistics; gathering repeated triplet channels must not reweight them.

Intensity transforms use the same code semantics in training and prediction, before randomized augmentation. Each acquisition estimates its own statistics. Invalid foreground, noise or field estimates yield an identity correction for that acquisition, followed by the retained normalization mapping; they do not drop an acquisition or study. Preserve the incumbent constant-series handling. Zero padding remains zero. Numerical support rules must be specified before launch and logged.

## Evidence and budget
Use the supervisor's incumbent-matched seed, fixed evaluation and sealed range budget, with assignment fractions 0.5–1.5 and inherited absolute limits authoritative. Initially retain ten epochs. Record preparation/training/prediction cost, configuration, source identity and full epoch curves. Diagnostics include changed-pixel fraction, normalization error or landmark movement, valid support, fallback fraction and transform-specific activity. Vendor and availability slices are exploratory; small vendors are not suitable optimization targets.

Follow [shared evaluation and adoption](../../protocol.md#shared-evaluation-and-adoption): valid macro improvement >= 0.001 and central 80% paired-bootstrap lower bound strictly above zero. No per-label objective, extra threshold or confirmation run. The supervisor owns scheduling, launches, verdicts and records.

## Integration
Preserve unrelated accepted improvements. Continuous storage is compatible with later mappings. Noise and receive-field correction precede the final intensity mapping. Foreground-global and local-histogram mappings are alternatives: a replacement removes superseded mapping branches and settings rather than stacking arbitrary equalizations. Each seed supplies any support-estimation helper it needs and has no dependency on another adoption. Reconcile these semantics with every newly accepted incumbent.

## Cited weaknesses

Values from the topic packet, checked by code:

- `/incumbent/slices/vendor/Philips/per_label/PF OA` = 0.842669: PF OA is particularly weak in the 267-study Philips slice. Scanner-dependent appearance is a hypothesis; prevalence and case mix also differ.
- `/incumbent/slices/vendor/Siemens/macro_over_defined_labels` = 0.917681: The largest vendor group, Siemens with 396 studies, has lower macro AUC than GE. Deterministic signal normalization upstream of augmentation remains unexplored.
- `/incumbent/slices/vendor/GE/macro_over_defined_labels` = 0.958428: GE's 198-study macro AUC contrasts with Siemens and Philips. This is descriptive evidence of heterogeneous difficulty, not an adjusted scanner effect.
- `/incumbent/per_label/Synovitis` = 0.896563: Synovitis remains weak despite active smoothing, intensity augmentation and scanner confusion. Preserving low-intensity and local contrast information is an untested alternative.

## Literature

- <https://pmc.ncbi.nlm.nih.gov/articles/PMC2254141/>
- <https://www.sciencedirect.com/science/article/pii/S0734189X8780186X>
- <https://iacl.ece.jhu.edu/proceedings/isbi2010/pdfs/0000708.pdf>

## Starting point

- Reference run: `outputs/autoresearch/radimagenet-rn18/experiments/rn18-more-triplets-20260928T235020Z-fd2c7bef/training`, recipe `outputs/autoresearch/radimagenet-rn18/experiments/rn18-more-triplets-20260928T235020Z-fd2c7bef/training/effective_config.json`.
- Chained on `rn18-more-triplets-20260928T235020Z-fd2c7bef`, the incumbent of [radimagenet-rn18](../radimagenet-rn18/INDEX.md), with its frozen source `outputs/autoresearch/radimagenet-rn18/experiments/rn18-more-triplets-20260928T235020Z-fd2c7bef/source`, seed and fold.
- Evaluation and adoption follow the [shared rule](../../protocol.md#shared-evaluation-and-adoption).
