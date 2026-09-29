# RadImageNet campaign (validated 2026-09-28, not started)

| Read when | Document |
|---|---|
| Checking the objective, budget, fixed and permitted mechanisms and evidence requirements | [Contract](contract.md) |
| Checking the runs, curves, drift and noise the seeds build on | [Prior work](prior-work.md) |
| Reading the configuration-only seeds (14 epochs, warmup, series dropout, DSU) | [Seeds 1](seed-ideas.yaml) |
| Reading the code seeds (layer-wise decay, linear probe then fine-tune, L2-SP, mean-initialized adjacent slices, frozen stem) | [Seeds 2](seed-ideas-2.yaml) |
| Reading the input, rate, multi-scale, schedule-shape and ResNet-10t batch seeds | [Seeds 3](seed-ideas-3.yaml) |
| Reading the plane-backbone seed (one ResNet-18 per plane with dense readouts and learned fusion) | [Seeds 4](seed-ideas-4.yaml) |
| Checking results | [Findings](findings.md) |

## Setup

| Read when | Document |
|---|---|
| Checking the sealed recipe, reference, budget, scope and role instructions | [Campaign definition](campaign.yaml) |

- **Code:** the `radimagenet` worktree's changes are in main's working tree (uncommitted); the
  configs, model, spatial tokens and RadImageNet loader equal the worktree byte for byte.
- **Reference:** the ReXNet control, with `reference_equivalences` for the two keys its recorded
  config lacks (`model.pretrained_weights: timm`, `model.slice_channels: adjacent`) and the
  `sources_sha256` identity exception below.
- **Budget:** range 0.284–0.851 of the reference's 1,912 s, sealed as 543–1,627 s of training
  (50–150% of the RadImageNet base run's 1,085 s), the only budget rule: epochs, batch size, steps
  and input per study are free, and a run past the upper bound is stopped as invalid.
- **Logging:** every run logs `backbone_drift_stage*`, `backbone_grad_rms_stage*` and
  `head_grad_rms` per epoch (Lightning module), so the researcher sees where learning stops.
- `validate` passed on 2026-09-28 with 15 forced seeds.

## Reference parity

The ReXNet reference was trained before the RadImageNet code touched two hashed preprocessing
sources (`modeling/model.py`, `modeling/spatial_tokens.py`), so its recorded `sources_sha256`
(`952e0d5d…`) differs from the current tree's. The user authorized a reference identity exception
instead of a retrain (2026-09-28). Proof:
[`check_reference_source_parity.py`](../../../scripts/adhoc/check_reference_source_parity.py) ran
the reference's selected checkpoint on CPU over the first 16 validation studies, once with the
reference's own `source_snapshot.zip` (which reproduces `952e0d5d…`) and once with the current tree
(`02ef3566…`, which also includes the uncommitted `data/splits.py` move); the 16 × 12 predictions
are bit-identical. Rerun on the final tree on 2026-09-28: still bit-identical, same `02ef3566…`.
Rerun it before starting if hashed sources change again:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= PYTHONPATH=<extracted snapshot>/src uv run --locked python -m scripts.adhoc.check_reference_source_parity --run outputs/training/20260928-150724-rebuild-state-v2-control --output old.npy
rtk proxy env CUDA_VISIBLE_DEVICES= uv run --locked python -m scripts.adhoc.check_reference_source_parity --run outputs/training/20260928-150724-rebuild-state-v2-control --output new.npy
```
