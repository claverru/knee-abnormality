# 2026-09-12 accurate-centers result

Retain midpoint sampling and the submitted `crop140mm` model as the frozen
fusion control. The geometric-center candidate did not improve the selected
fold-0 checkpoint; its sampling path was removed from production while the
shared geometry index and preprocessing-identity enforcement were retained.
There was no Kaggle submission for this experiment.

| Selected epoch-8 checkpoint | Macro AUC | Soft-target BCE | Gold AUC | Ten-epoch training |
|---|---:|---:|---:|---:|
| `crop140mm`, midpoint control | 0.88612165 | 0.45296129 | 0.92698413 | 2,130.23 s |
| `accurate-centers`, geometric candidate | 0.88513698 | 0.45403857 | 0.92791005 | 1,864.39 s |

The paired predictions cover the same 904 study UIDs. All recorded input,
label, membership and split identities match apart from the newly added
preprocessing identity. Both runs used seed 42, 3,503 training studies, ten
epochs, 4,380 optimizer steps, the saved `crop140mm` configuration and fixed
2× validation density. Training budgets per contrast group were axial 18,
coronal 15 and sagittal 15 raw slices. At implementation start, the mutable
YAML had a different allocation; the saved control config remained authoritative
throughout. At handoff the YAML also has 18/15/15. No model, optimizer, crop
or image budget was changed for the comparison.

Macro AUC fell by 0.00098466 and BCE increased by 0.00107728. Six targets
improved and six declined. Largest declines were Fracture (−0.01242), Lateral
Meniscus (−0.00717) and ACL (−0.00489); largest gains were Medial OA (+0.00634),
Baker's (+0.00571) and Synovitis (+0.00565). The tiny gold increase concerns
only eleven studies. This single-seed result does not establish a significant
regression; it provides no gain supporting adoption under the plan's rule.

Peak allocated CUDA memory was identical at 20,017,470,464 bytes. Full training
was 12.5% faster, and post-training validation prediction took 36.47 versus
37.37 seconds. The control's archived source predates the separately verified
native-reader optimization, so the runtime difference is cumulative and cannot
be attributed to centers alone. See the bounded loader measurements and
128-study geometry/coverage audit in [data preparation](../../data-preparation/accurate-sampling-centers.md).
A training worker snapshot recorded 280–340 MB unique resident memory per
worker, without a matched control or a peak claim.

The [control manifest](../../../outputs/analysis/a/accurate-centers-20260912/control.json)
pins checkpoint, config, executed source, input identities, timing and prediction
hashes. The selected control is
`outputs/training/crop140mm/fold_0/best-epoch=07-val_macro_auc=0.88612.ckpt` (deleted on 2026-09-16), SHA-256
`f3fd7b72bed1315297d54ee92742965ebb9c892ae0683c717bf20ad0947d3a92`,
with frozen bundle `outputs/submission/exports/knee-crop140mm-20260912`.
The rejected candidate is
`outputs/training/accurate-centers/fold_0/best-epoch=07-val_macro_auc=0.88514.ckpt` (deleted on 2026-09-16), SHA-256
`bf23a0e92e860062c0e52dbebf58428f0b56cd21052ce3e1a338de3cd1274c50`.
Its source, configuration and epoch predictions remain under
`outputs/training/accurate-centers`; the checkpoints and the weights and wheels of its local bundle
`outputs/submission/exports/accurate-centers-candidate-20260912` were deleted on 2026-09-16.

The candidate passed normal post-training validation loading, export, and
resume identity validation on its actual completed-run checkpoint. Its bundle
recomputed the same identity without training inputs and predicted the three
example-test studies successfully. Historical checkpoints with missing identity
were rejected by prediction/export. After selecting midpoint sampling, the
candidate's recorded identity is also rejected by current code. Use the
matching archived implementation for either historical model; do not rewrite
their metadata. Fresh midpoint runs use the distinct
`midpoint_triangular_shared_geometry_v1` identity.

After removing the rejected sampler, a two-pass, 16-study production profile
matched the archived control's study order, input counts and every prepared
image hash. Static lint, formatting and catalog checks passed; no tests were
written or run. These checks do not establish unexercised metadata cases or
future local-alignment validity.
