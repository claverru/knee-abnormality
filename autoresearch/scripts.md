# Existing script integrations

Autoresearch reuses exactly these two existing scripts. Their implementations stay
in `scripts/`; this document records how the supervisor invokes them. Runtime commands
use the absolute Python executable from the launching locked environment, with
`PYTHONPATH` pointing to the appropriate frozen source, instead of creating a new
environment inside every snapshot. These equivalent manual templates use `uv`.

## Training

Use [scripts/pipeline/train.py](../scripts/pipeline/train.py) for every training run.
The supervisor launches it detached from an isolated, frozen candidate checkout.
The candidate configuration must carry a fresh recorded training seed and satisfy
the campaign contract before launch.

Invocation template, from that checkout's repository root:

```bash
rtk proxy uv run --locked python -m scripts.pipeline.train CANDIDATE_CONFIG \
  --folds 0 \
  --output-root outputs/autoresearch/CAMPAIGN_ID/experiments/EXPERIMENT_ID \
  --run-id training
```

Replace `CANDIDATE_CONFIG`, `CAMPAIGN_ID` and `EXPERIMENT_ID` with the recorded
configuration path, campaign ID and unique attempt ID. The resulting `training/`
directory must not already exist.
The supervisor must resolve data, environment, and artifact paths consistently
across isolated checkouts.

The script saves the effective configuration, source snapshot, provenance,
checkpoints, epoch metrics, selected-checkpoint predictions, timing, optimizer-step
count, and peak CUDA memory. Its source archive is provenance; execution isolation
and detached process ownership remain the supervisor's responsibility.

For campaign runs, the supervisor supplies `KNEE_AUTORESEARCH_LIMITS`, enabling the
trusted triplet-accounting callback. It writes `fold_0/triplet_accounting.json` and
rejects count, cohort, validation-determinism, or training-randomness violations.
Ordinary training does not enable that callback. All launches and validation remain
local; no submission or publication script is integrated.

## Control-versus-experiment comparison

Use [scripts/tools/compare_fold_runs.py](../scripts/tools/compare_fold_runs.py)
after training completes. It reads saved predictions and checkpoint tensors on CPU,
without constructing or running a model. It calculates paired report-group
bootstrap intervals for candidate-minus-reference AUC differences, with 2,000 draws
and 95% intervals by default, and writes `comparison.json` plus analysis sources.

Invocation template:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked python -m scripts.tools.compare_fold_runs \
  --reference CONTROL_RUN \
  --candidate EXPERIMENT_RUN \
  --output-dir outputs/autoresearch/CAMPAIGN_ID/experiments/EXPERIMENT_ID/comparison \
  --fold 0 \
  --allowed-config-differences name,seed,paths.output_root
```

`CONTROL_RUN` and `EXPERIMENT_RUN` are complete training run directories containing
`fold_0/`, not checkpoint files. The comparison output directory must be fresh.
Add only the candidate's reviewed keys permitted by the sealed campaign policy
to the configuration allowlist. Input-identity exceptions come from that policy's
`input_identity_keys`. Sampling permits `preprocessing_identity` after source
review; the loss campaign permits no input-identity differences.

The script requires frozen fold files, complete run artifacts including timing and
source snapshots, and the acquisition metadata specified by `--headers`. Its default
headers path is `outputs/analysis/sharing-disagreement-20260914/study_headers.parquet`.
Selected predictions must reproduce their recorded epoch AUC. Missing or inconsistent
artifacts are invalid evidence, not permission to weaken these checks silently.

Its `--seed` controls bootstrap resampling only; it does not fix the training seed.
Intervals exclude training-seed and checkpoint-selection variability. The script
produces comparison evidence; the runtime applies the fixed verdict after
orchestrator interpretation, under the selected [campaign contract](campaigns/INDEX.md).
