# Inference and efficiency

## Procedures and requirements

| Read when | Document |
|---|---|
| Checking submission state and inference context | [Inference and efficiency overview](inference-overview.md) |
| Checking submission rules and the output contract | [Submission requirements and output contract](submission-requirements-and-output-contract.md) |
| Interpreting the efficiency leaderboard | [Efficiency leaderboard](efficiency-leaderboard.md) |
| Building frozen inference bundles and offline environments | [Repository readiness and model bundle](model-bundle-and-offline-environment.md) |
| Preparing exports and detached submissions | [Fast submission preparation](fast-submission-preparation.md) |
| Running training-to-submission commands | [Justfile recipes](justfile-recipes.md) |
| Operating the Kaggle submission workflow | [Kaggle submission workflow](kaggle-submission-workflow.md) |
| Cancelling a queued or running Kaggle notebook from the API, or finding its session id | [Cancelling a notebook run](kaggle-submission-workflow.md#cancelling-a-notebook-run) |
| Configuring GPU processes, loader workers, backbone precision and warm-up | [Independent GPU inference processes](independent-gpu-inference-processes.md) |
| Checking physical-sampling identity and seed compatibility | [Physical-sampling runtime identity](physical-sampling-runtime-identity.md) |

## Runtime and memory

| Read when | Document |
|---|---|
| Choosing inference image counts and batch size | [Inference image count and memory](inference-image-count-and-memory.md) |
| Reviewing loader, precision and startup profiles | [2026-09-15 Kaggle inference profiling](2026-09-15-kaggle-inference-profiling.md) |
| Inspecting threaded DICOM headers, frame loading and cold-start stages | [2026-09-18 Kaggle x4 profile after the loading changes](2026-09-18-kaggle-x4-profile-after-loading-changes.md) |
| Inspecting series filtering, threaded listings, loader ordering and CPU attribution | [2026-09-18 Kaggle x4 profile after the index changes](2026-09-18-kaggle-x4-profile-after-index-changes.md) |
| Inspecting preparation CPU changes, resampling and the I/O-thread sweep | [2026-09-18 Kaggle x4 profile after the CPU cluster](2026-09-18-kaggle-x4-profile-after-cpu-cluster.md) |
| Checking the current inference runtime, startup, CUDA graph, GPU image preparation and row-trimmed reads | [2026-09-22 Kaggle x4 startup and GPU preparation](2026-09-22-kaggle-x4-startup-and-gpu-preparation.md) |
| Reviewing unverified runtime evidence and efficiency ideas | [Unverified runtime evidence and efficiency leads](runtime-evidence-and-leads.md) |

## Submission records

| Read when | Document |
|---|---|
| Finding submission records and the ledger | [Submission records](submissions/INDEX.md) |

## Common lookups

| Read when | Section |
|---|---|
| Finding train, resume, export and submit command syntax | [Recipe reference](justfile-recipes.md#recipe-reference) |
| Following a checkpoint export and submission example | [Export and submit](justfile-recipes.md#export-and-submit-a-checkpoint) |
| Inspecting detached submission progress and runner logs | [Submission status and logs](justfile-recipes.md#detached-submission-status-and-logs) |
