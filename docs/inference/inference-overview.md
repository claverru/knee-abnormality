# Inference pipeline overview

How a checkpoint became a scored Kaggle submission, as the pipeline stood at the end of the
competition. Every submission came from `main`; the separate inference-speed branch was removed on
2026-10-23 and efficiency submissions were exported from `main` too. Scores, settings and Kaggle's
order among ties for every submission are in the [ledger](submissions/ledger.md); the model itself is
in [current model](../modeling/current-model.md).

## Stages

1. **Export** ([`scripts/pipeline/export.py`](../../scripts/pipeline/export.py)): one or more fold
   checkpoints become an immutable bundle with the frozen config, a copy of the shared package, both
   prediction entry points, the notebook template and offline wheels. Image multiplier, attention
   temperature (one value or one per member), batch size, loader workers, mirror TTA and backbone
   mode (`graph` compiled and captured, or `eager`) are bundle settings, and a bundle rejects
   contradicting overrides at prediction time
   ([bundle](model-bundle-and-offline-environment.md), [recipes](justfile-recipes.md)).
2. **Submit** ([`scripts/pipeline/submit.py`](../../scripts/pipeline/submit.py), detached through
   `just submission-job`): versions a private dataset, builds a compile cache on Kaggle when the bundle
   lacks one, runs the offline GPU notebook on the visible test, verifies `submission.csv` and submits
   that exact notebook version; every remote response is written to a record
   ([workflow](kaggle-submission-workflow.md)).
3. **Kaggle rerun**: the notebook discovers studies from the mounted test data, one process per visible
   GPU decodes and scores a disjoint partition, members of an ensemble share each decoded batch, and the
   merged CSV is validated against the output contract
   ([GPU processes](independent-gpu-inference-processes.md),
   [output contract](submission-requirements-and-output-contract.md)).
4. **Scoring**: Kaggle reports a public score only; no runtime is exposed through the API. The
   [efficiency leaderboard](efficiency-leaderboard.md) is the only view of hidden runtimes.

## Settings that varied

- Image count and memory: ×1 to ×3 on the first baselines ([image count](inference-image-count-and-memory.md)),
  ×2 for the early fold-0 models, ×4 from 2026-09-18, ×6 in S103–S106 and ×8 in S190–S191.
- Attention temperature τ: 1.0 until 2026-09-18, 0.75 from S28, then 0.9 and tuned per ensemble
  (0.35 to 0.85 in the October ensembles).
- Runtime: FP32 on two T4s with 2 to 4 workers for the first dual-GPU submissions; FP16 chunked
  backbones and 16 workers from 2026-09-15; GPU image preparation, CUDA graphs, buffer pools and
  compile caches afterwards ([profiling records](profiling/INDEX.md)).

## Where the evidence is

| Question | Document |
|---|---|
| What each early submission contained | [Early submission details](submissions/early-submission-details.md) |
| Why runtime was priced in AUC | [Efficiency exchange rate](profiling/2026-10-16-efficiency-exchange-rate.md) |
| What competitors reported about runtime | [Runtime evidence and leads](runtime-evidence-and-leads.md) |
| The 2026-09-14 comparison of reflection, 1A, 1B and 1C inference clocks | [Generalization inference clocks](../history/sharing/generalization-inference-clocks.md) |
