# Sync-free training step

The training step issues thousands of small operations and waits at a few hundred points that
force a GPU synchronisation (boolean indexing, `nonzero`, `.item()`, `unique`, `linalg` error
checks, constants built from Python lists, Python branches on tensor values). After each one the
GPU queue is empty and the main thread cannot run ahead, so a host with slow single cores
leaves the GPU idle ([2026-10-08 measurements](../../modeling/training-runtime-and-memory.md#posed-crop-recipe-where-a-step-goes-and-what-was-removed-2026-10-08)).

- **Change:** replace the synchronising operations with masked tensor operations, device-side
  constants and asynchronous assertions, drawing the same random numbers in the same order.
- **Keeps:** every output and the random stream; proved by the same weights giving the same
  predictions and by the first training losses following the same trajectory.
- **List of sync sites:** `torch.cuda.set_sync_debug_mode("warn")` over a few real training
  steps (the audit script of 2026-10-14 records each call site with its source line).

## Audit of 2026-10-14

ResNet-50 student recipe (`configs/rn50-distilled-student.yaml`, Kaggle + fastMRI pairs), RTX PRO
6000 with an EPYC 7543 host (small-kernel launch 8.5 µs), five steps beside another training:
**281 synchronisations per step**, 199 ms of the 809 ms step blocked in them, 7,749 kernel
launches, GPU busy 71%. Alone, that host trains this recipe at 2.5 to 3.0 steps/s with the GPU
50 to 65% busy, against 3.7 steps/s for the ConvNeXt student on an i7-14700K host.

| Group | Per step | Sites | State |
|---|---:|---|---|
| A. Training logging | ~35 | `_log_crop_frames`, `_log_family_activity`, `_side_step`, `_meniscus_step`, `_anatomy_step` counts and `.item()` | Changed: masked sums with tensor weights, read once at the end of the epoch |
| C1. Linear algebra error checks | ~19 | `linalg.solve` (frame planes, frame pose), `linalg.inv` (feature geometry, crop rows, augmented-to-prepared), `matrix_exp` (frame rotation) | Changed: `solve_ex`, `inv_ex`; closed-form Rodrigues rotation in place of `matrix_exp` |
| C2. Constants per step | ~12 | `torch.tensor([...], device=...)` in the study frame, regions, crop rows, OA indices, anatomy rows, reflection and conversion matrices | Changed: device-side construction, cached per size, non-persistent buffer, pinned non-blocking rows |
| C4. Checks | ~6 | `_effective_presence`, finite pooled logits, anatomy `mask.any()` | Changed: `torch._assert_async`; masked anatomy loss |
| C3. Data-dependent selection in `_encode_crops` | ~45 | `keep_series` masks, `nonzero` of the slice mask, `aligned` rows | Partly changed: `keep_series` skipped when series dropout is off (every batched series is present); the slice-mask indices stay, since the device preparation drops constant images |
| B. Augmentation | ~95 | contrast-mixing loop per study and plane, region filters, affine, noise, intensity, resolution, bilateral selections | Changed: contrast mixing vectorised over study and plane groups with one synchronisation for the group counts, then the same `rand` calls in the same order; the image augmentations draw their parameters up front in the original order and take every selection count in two synchronisations, indices by `nonzero_static`; GIN normalises per series index instead of `unique`; cached kernels and matrices. Feature-statistics augmentation keeps its two (`unique`, group count), since a fixed group count would remap its draws |
| D. Backward | ~7 | autograd over the indexing above | Follows C3 |

## Parity and speed (2026-10-14, RTX PRO 6000, idle host)

| Check | Old code | Changed code |
|---|---|---|
| Same weights, 64 validation studies at ×4 (ConvNeXt student) | | mean difference 7e-8, max 6e-6; after merging main's model split 2.5e-6, max 7e-4 (ResNet-50 student 3.6e-6, max 4e-4) |
| Training step, ResNet-50 student recipe, 60 steps | 275 and 276 ms (two runs) | 258 ms; 260 ms after the merge (−6%) |
| First 12 losses | two runs differ by about 0.001 | within that spread |
| Synchronisations per step | 281 | 50; GPU busy 84% |

Remaining sites (about 50): kornia inside affine warping, gamma, contrast and noise (13); the slice-mask
`nonzero` in `_encode_crops` and `_augment_images` (5), which need the constant-image check moved
to the loader; feature-statistics augmentation (4); `_log_views`, the masked target loss, the
anatomy centre checks and the soft-tissue routing lists and finiteness checks (about 15); the
three count reads of contrast mixing and image augmentation; Lightning's own step logging.
Edits change the hashed sources, so earlier checkpoints need the current identity to load.
