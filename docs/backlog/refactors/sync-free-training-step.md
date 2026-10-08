# Sync-free training step

The training step issues about 5,500 small operations and waits at about 265 points that force a
GPU synchronisation (boolean indexing, `nonzero`, `.item()`, `unique`, Python branches on tensor
values), most of them in `modeling/augmentation.py` and the geometry of `modeling/model.py`. The
GPU executes about 180 ms of kernels in a 245 ms step, so the two cannot overlap
([measurements](../../modeling/training-runtime-and-memory.md#posed-crop-recipe-where-a-step-goes-and-what-was-removed-2026-10-08)).

- **Change:** replace the per-study and per-plane Python loops and the value-dependent branches
  with masked tensor operations that draw the same random numbers in the same order, so the
  step runs without forced synchronisation until the loss.
- **Keeps:** every output and the random stream; proved by the same weights giving the same
  predictions and by the first training losses following the same trajectory.
- **Expected:** a step near the GPU's own 180 ms (an estimate, not measured).
- **List of sync sites:** `torch.cuda.set_sync_debug_mode("warn")` under
  `scripts/tools/bench_training_steps.py`.
