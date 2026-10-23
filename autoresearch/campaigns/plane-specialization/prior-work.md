# Prior work

Where the seeds come from, the measured precedents and the closed directions.

- **Selection review, 2026-09-27:** the user wants each backbone to learn from one plane only.
  With one shared backbone the knowledge is spread across planes: given a single plane, the
  shared ReXNet control stays close to its full score, and fusion adds little over the best
  single plane for most labels. Plane-specialized backbones that must extract the most from their
  own plane should fuse into a better model. B0 keeps the runs fast; the ReXNet control stays the
  yardstick. Seeds: the user's random label-to-plane partition, its dense counterpart, learned
  fusion versus the present-plane mean, per-plane heads, the fixed radiology mapping (the user
  allowed label-to-plane routing for it), auxiliary per-plane readouts, competitive
  (winner-take-all) routing and a full independent head per plane.
- **Seed noise:** re-seeding the adopted smoothing recipe moved ×4 macro AUC by 0.0049
  ([augmentation result](../../../docs/validation/results/campaigns-and-queues/2026-09-26-augmentation-campaign.md#seed-noise));
  re-seeding the knee-geometry recipe moved it by −0.0063, with Contusion −0.0205, PF OA −0.0154,
  Fracture −0.0137 and MCL −0.0129. Treat single-label moves of that size in those labels as
  noise.

## Single-plane evidence

[Single-plane inference](../../../docs/validation/results/pooling-and-attention/2026-09-27-single-plane-inference.md)
on `scanner-confusion-control` (shared ReXNet, series dropout 0.2), fold 4, ×4: all series
0.9097, sagittal only 0.8709, coronal only 0.8672, axial only 0.8487. The best single plane per
label trails the full model by 0.020 on average, but by 0.008 or less for five labels (ACL
sagittal +0.003, Baker's axial −0.001, Fracture coronal −0.005, Effusion axial −0.006, MCL
coronal −0.008); only the menisci need several planes (−0.040, −0.086). Series dropout trains
the model to cope with missing planes, which inflates single-plane scores on its own. The
reference's own single-plane scores are not yet measured.

## Reference

`20260927-210053-b0-plane-backbones`: ×4 macro 0.898434, soft 0.7899, selected epoch 9 of 10,
training 1,534 s, peak 20.0 GiB, 2,220 optimizer steps. Against the ReXNet control (×4, same
fold and seed):

| Label | ReXNet | B0 planes | Δ |
|---|---:|---:|---:|
| ACL | 0.9591 | 0.9608 | +0.0017 |
| MCL | 0.9180 | 0.8952 | −0.0228 |
| Medial Meniscus | 0.9188 | 0.9145 | −0.0043 |
| Lateral Meniscus | 0.9079 | 0.8905 | −0.0174 |
| Medial OA | 0.9065 | 0.8989 | −0.0076 |
| Lateral OA | 0.8741 | 0.8608 | −0.0133 |
| PF OA | 0.8795 | 0.8669 | −0.0126 |
| Effusion | 0.8806 | 0.8770 | −0.0036 |
| Synovitis | 0.9058 | 0.9011 | −0.0047 |
| Baker's | 0.9369 | 0.9249 | −0.0120 |
| Contusion | 0.9075 | 0.8786 | −0.0289 |
| Fracture | 0.9221 | 0.9119 | −0.0102 |
| **Macro** | **0.9097** | **0.8984** | **−0.0113** |

The reference differs from the ReXNet control in backbone family, plane backbones, DSU (off),
series dropout (0), batch (16) and learning rates at once, so the gap has no single cause.

## Precedent table

Measured runs outside autoresearch; deltas are macro AUC against the control of their day.

| Mechanism | Setting | Run | Δ | Source |
|---|---|---|---|---|
| Plane/contrast readouts | six per-type heads replacing the shared readout, shared backbone | near-misses #11 | −0.0006 (Baker's −0.0103) | [result](../../../docs/validation/results/campaigns-and-queues/2026-09-25-near-misses-campaign.md) |
| Series dropout | 0.1, 0.2, 0.3 (ordinary BCE) | 2026-09-16 queue | flat between 0.2 and 0.3 | [overnight queue](../../../docs/validation/run-comparison-and-diagnostics.md#overnight-launcher) |
| Acquisition-count normalized attention | per-type count normalization | `count-normalized-attention` | −0.0035 | [result](../../../docs/validation/results/pooling-and-attention/2026-09-17-count-normalized-attention-result.md) |
| B0 shared backbone | batch 16, both rates ×2 (8e-4 / 1.2e-3), 10 epochs, ordinary BCE, `all_5fold_v3` fold 0, ×1 | `03-lr-linear` | +0.0034 vs its B0 control; −0.0044 vs ReXNet | [b0 queues](../../../docs/validation/results/campaigns-and-queues/2026-09-18-b0-hyperparameter-queues.md) |
| B3 shared backbone | current recipe, fold 3, `backbone_lr` 4.5e-4, two-epoch warmup | `03-warmup-2` | +0.0004 vs ReXNet | [B3 queues](../../../docs/validation/results/campaigns-and-queues/2026-09-27-b3-overnight-queue.md) |
| ReXNet + B0 ensemble | five folds each, ordinary BCE | 2026-09-19 folds | +0.0061 OOF over ReXNet | [folds](../../../docs/validation/results/validation-and-ensembles/2026-09-19-fold-ensembles-and-sigma-sweep.md) |

## Closed directions

- Redundancy penalties between plane embeddings (see the contract's exclusions).
- Per-label tuning from label results, gold-study weighting, series dropout, backbone or
  hyperparameter changes.
