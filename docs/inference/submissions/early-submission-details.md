# Early submission details (S01–S32)

Bundle contents, runtime settings, checkpoint identities and the reason behind each early
submission, for the rows of the [ledger](ledger.md) that need more than a label. Scores and
Kaggle's order among ties are in the ledger. Bundles lived under `outputs/submission/exports/`, workflow
records under `outputs/submission/records/`, isolated runtimes under `outputs/submission/runtimes/`;
all fold-0 checkpoints cited here were deleted on 2026-09-16 (SHA-256 kept as identity). Kaggle's
API gives no official hidden-run inference time, and "launch to score" waits include queueing,
so neither is a throughput measurement. Larger analyses live in the validation results:
[temperature and count (S27–S32)](../../validation/results/pooling-and-attention/2026-09-18-temperature-and-count-submissions.md),
[ensembles (S33–S37)](../../validation/results/validation-and-ensembles/2026-09-19-ensemble-submissions.md) and
[learned-view and blend submissions (S94–S97)](../../validation/results/validation-and-ensembles/2026-10-02-blend-submissions.md).

## Baselines (S01–S04, 2026-09-11)

- **S01, `baseline3` ResNet-50d fold 0** (epoch 7, 0.849819; SHA-256 `b955aefe…`). Export
  state dictionary equal to the checkpoint (341 keys); fresh BF16 inference reproduced all 903 saved
  validation rows exactly (macro AUC 0.8498192406240211), 24.13 studies/s on the 3090. Bundle
  `baseline3-fold0-v3`, notebook version 3 ran offline in 82.12 s on Kaggle in FP32 (three-study
  prediction 11.54 s). Public 0.878.
- The offline Python 3.12 bundle carried `pylibjpeg==2.1.0`, `pylibjpeg-libjpeg==2.4.0` and
  `pylibjpeg-openjpeg==2.5.0`, which decoded pydicom's JPEG Lossless (`1.2.840.10008.1.2.4.70`) and
  JPEG 2000 Lossless (`1.2.840.10008.1.2.4.90`) fixtures. All 819,078 local training DICOMs are
  uncompressed Explicit VR Little Endian, so no local file exercised the compressed paths.
- The project then moved from the absent `lightning` distribution to Kaggle's
  `pytorch-lightning==2.6.5` (bundle `baseline3-fold0-v5`; 903 rows reproduced again). Training and
  prediction both use PyTorch's `high` float32 matmul policy, so the float32 target-attention
  computation stays consistent under BF16 autocast.
- **S02–S04, `baseline4` ReXNet-200 fold 0** (epoch 7, 0.872510, all 903 rows reproduced; wheel-free
  bundle `baseline4-fold0-v1`): S02 (×1) 54.14 s visible notebook, scored 23 min 41 s after
  submission; S03 ×2 and S04 ×3 are covered in
  [inference image count and memory](../inference-image-count-and-memory.md).

## Overnight fold-0 candidates (S05–S08, 2026-09-12)

Queue `overnight-20260911T234416Z-8du9a4nh`, all at ×2 and batch 2, notebook version 1.

| S | Run | Epoch | Submission |
|---|---|---:|---:|
| S05 | `02-longer` | 12 | 56181804 |
| S06 | `04-higher-lr` | 10 | 56181821 |
| S07 | `05-faster-head` | 10 | 56181919 |

The winner `04-higher-lr` (epoch 10, 0.88294, SHA-256 `900add74…`) was the 150.08 mm comparison
reference. **S08, `crop140mm`** (epoch 8, 0.88612, SHA-256 `f3fd7b72…`; 224-pixel inputs at 0.625 mm,
×2, batch 2) was the frozen 140 mm control; its manifest records bundled source that differs from the
training snapshot because export happened during preprocessing package changes.

## Spatial fusion (S09, 2026-09-12)

`spatial-fusion` (27 queries, attention width 64, eager; epoch 10, 0.884169; SHA-256 `e9e0154e…`) scored
0.005 below `crop140mm`, in line with its lower standalone validation. Its deployment gate failed
earlier: even the reduced, compiled nine-query/width-32 variant added 7.47% warm end-to-end inference
time at batch two (budget 5%), see the
[engineering comparison](../../validation/results/sampling/2026-09-12-spatial-fusion-engineering-result.md).
The initial 27-query version added about 16% model time at batch two: 2.0 ms mapping, 2.4 ms
geometry-bias construction, 0.9 ms projection, only 0.3 ms cross-attention (PyTorch 2.10 picked
memory-efficient SDPA with BF16/additive-bias inputs). Sparsity was therefore not justified by
attention-kernel cost. These were local profiles, not Kaggle runtimes.

## Spatial tokens and sampling (S10–S13, 2026-09-13)

| S | Model | Epoch, local AUC | Checkpoint SHA-256 | Notes |
|---|---|---|---|---|
| S10 | `spatial-tokens-five-embeddings` | 10, 0.88950565 | `c58fd47a…` | Source SHA-256 `6e68f337…`; first notebook name rejected with HTTP 400 after the dataset upload, retry reused the dataset version (no submission consumed) |
| S11 | `tonylin-soft-v3` | 9, 0.89230583 | `55a9bf41…` | 0.008 below S10, so S10 stayed the baseline ([label comparison](../../validation/results/labels-and-losses/2026-09-13-tonylin-label-comparison.md)) |
| S12 | `3dsampling` | 9 | `3a63b49d…` | +0.001 over S10; +0.000458 macro AUC locally, slightly worse BCE ([comparison](../../validation/results/sampling/2026-09-13-3dsampling-validation-comparison.md)) |
| S13 | `spatial-tokens-reflection` | 9 | `bc55feaa…` | +0.001 over S12; +0.001012 locally, lower BCE and lower eleven-study gold AUC ([comparison](../../validation/results/sampling/2026-09-13-reflection-validation-comparison.md)) |

## Shared heads and compartment attention (S14–S18, 2026-09-13 to 2026-09-14)

- **S14, 1B** (`meniscus-oa-classifiers`, epoch 7, 0.892022; `fc7ced81…`): two shared classifier pairs
  with independent attention. A later CPU-only audit found no bundle mismatch: manifest and file
  hashes, checkpoint identity and every bundled Python file matched; both exports had identical
  effective configuration, and both visible notebook runs were FP32
  ([audit](../../history/sharing/2026-09-13-1b-public-result-and-overnight-code-choice.md)).
- **S15, `02-central-sampling`** (queue `overnight-20260913T214822Z-74udtekr`, epoch 10, 0.892538;
  `4035f331…`): the 1C architecture with reflection probability 0.5 and sampling sigma 40 mm; tied
  S13 at 0.931.
- **S16/S17, original 1C** (`disease-compartment-attention`, sigma 80 mm, epoch 10, 0.889430;
  `fc34d318…`). S16 asked for ×2, batch 8 and four workers; its hidden rerun failed with an
  unhandled-error message and no score. The visible notebook had three studies, so batch 8 was never
  exercised, and GPU memory pressure is a possible but unconfirmed cause. Kaggle returned
  `status: COMPLETE` together with `errorDescription`, which the runner on branch `1C` treats as failure.
  The batch-4 retry (S17, script version `349717410`) scored 0.928, 0.003 above 1B and 0.003 below
  S13 and S15. These are separate trained models; the result does not isolate the compartment embedding.
- **S18, 1A** (`meniscus-classifier`, epoch 9, 0.891192; `d124774e…`; identity
  `spatial_tokens_shared_meniscus_classifier_v1`: only the meniscus classifier is shared). First
  submission with [independent GPU processes](../independent-gpu-inference-processes.md): ×2, four
  studies per GPU batch, four loader workers total. The isolated runtime started from 1A's training
  archive plus the branch-1C coordinator, worker, process module, inference metadata partitioning and
  exporter, with `runtime-provenance.json` pinning each overlaid file. The first attempt failed on request
  loading before inference (no submission); the `-v2` retry changed only
  `scripts/pipeline/predict_worker.py`. Visible notebook: both Tesla T4s in FP32, GPU 0 predicted two
  studies in 8.81 s, GPU 1 one in 4.89 s, worker wall time 24.64 s, whole prediction subprocess 83.65 s,
  notebook total 90.62 s; submission to score 16 min 07 s. These example timings are startup-dominated
  and show no speedup over single-GPU inference. Public 0.930
  ([cross-experiment investigation](../../history/sharing/generalization-investigation.md)).

## Reflection sigma 40 champion and its variants (S19–S22, 2026-09-14 to 2026-09-15)

- **S19, `reflection-sigma40`** (epoch 7, 0.892360168; `9526b289…`; architecture-by-sigma results in the
  [validation comparison](../../validation/results/sampling/2026-09-14-reflection-sigma-40-result.md)).
  Runtime started from the run's training archive with the dual-GPU coordinator, corrected worker, process
  module and exporter from branch `1C` (`3b8c2cc`); the coordinator, worker and process module match S18's
  `-v2` bundle. ×2, batch 4 per GPU, four loader workers total, two T4s in FP32. Launched 12:14:15 UTC,
  notebook 12:15:00, scored 12:34:41 (visible notebook 1 m 31 s, full workflow 20 m 26 s). +0.005 over
  S13 and S15 (0.931).
- **S20, `00-baseline`** (queue `overnight-20260914T220805Z-ozsz4qbu`, epoch 10, 0.892763, 0.000403 above
  S19; `da45267e…`): the champion recipe on current code with the token graph off (`graph_layers: 0`) and
  no additive 3D embedding. Public 0.934, 0.002 below S19 despite the higher fold-0 score. It differs
  from the champion by the removed 3D embedding and an unused graph still built each forward, and one
  run plus one public score cannot separate that from training variability.
- **S21, champion without 3D at inference**: unchanged S19 checkpoint with every token's 3D embedding
  replaced by the learned `unknown_3d` vector through one forward hook in `inference.py` (outside the
  preprocessing identity hash). `model.ckpt` byte-identical to S19's; on the three local examples (3090,
  BF16) predictions differed from the champion's by a mean of 0.008 and at most 0.024. Public 0.936,
  equal to S19, so the champion does not measurably rely on 3D at inference on public test data (it
  does not show whether 3D helped training). The first launch failed with HTTP 400 on a 52-character
  notebook title before any run; `submit.py` then rejected titles longer than 50 characters.
  Notebook 2 m 01 s, scoring 16 m 37 s.
- **S22, fast champion**: S19's checkpoint with the runtime selected by the
  [Kaggle inference profiling](../profiling/2026-09-15-kaggle-inference-profiling.md): 16 total loader
  workers and `auto` precision resolving to FP16 with 64-image channels-last backbone chunks on T4
  (eight loader workers per GPU). Public 0.936, equal to the FP32 champion; visible notebook 2 m 32 s, submission to score
  10 m 05 s against 18 m 08 s (S19) and 16 m 07 s and 16 m 37 s for the two FP32 four-worker submissions
  of that day: direction only, queueing included.

## Seed controls, EMA and ranked filtering (S23–S27, 2026-09-17)

Queue `overnight-20260916T214520Z-t8k_kw2z`, restored full-slice recipe, fast runtime (×2, four studies
per GPU, 16 loader workers, FP16 chunks on T4), launch to score 10–14 minutes.

| S | Run | Seed | Epoch | Fold 0 | Public | Checkpoint SHA-256 |
|---|---|---:|---:|---:|---:|---|
| S23 | EMA 0.999 of the seed-42 run | 42 | 10 | 0.893155 | 0.929 | `b87fc2df…` |
| S24 | control | 42 | 8 | 0.892995 | 0.928 | `66988d61…` |
| S25 | control | 43 | 10 | 0.889837 | 0.931 | `8cce373a…` |
| S26 | control | 44 | 10 | 0.891627 | 0.931 | `05060efd…` |
| S27 | `ranked-filtered-seed42` | 42 | 10 | 0.889573 | 0.933 | `194a1932…` |

- S23's bundle holds only the averaged weights; its live weights equal S24's, so EMA is the one paired
  comparison (+0.001 public, +0.00016 fold 0).
- Fold-0 order did not predict public order: among the three seeds the public order reversed the fold-0
  order, and the same recipe trained four times (three seeds plus `00-baseline`) scored 0.928–0.934
  publicly (seed mean 0.930) and 0.8898–0.8930 on fold 0. Validation and test triplet sampling break ties
  between repeated series of one type with the seed, so the differences mix training and
  evaluation-sampling noise. Public differences below about 0.006 between separately trained runs were
  treated as noise.
- S27 was the first submission of the series filtering and deterministic source ranking code
  (identity `ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10`,
  [result](../../validation/results/sampling/2026-09-17-ranked-filtered-result.md)); the new indexing, drops,
  guard and ranking ran on the full hidden test without failure. Its 0.933 lies inside the 0.928–0.934
  range of the same recipe on old code and above the three seed controls, with the lowest fold-0 score of
  the four. Visible notebook 3 m 32 s, scoring 9 m 04 s.

## S28–S32 (2026-09-18)

The temperature, count and `series-share` analysis is in
[temperature and count submissions](../../validation/results/pooling-and-attention/2026-09-18-temperature-and-count-submissions.md).
The three `ranked-filtered … + tau0.75` bundles carry τ 0.75 folded into the final scorer layer
(checkpoint `b4ab10f1…`), so their manifests record `attention_temperature: 1.0`.
