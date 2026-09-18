# Inference and efficiency

## Procedures and requirements

| Read when | Document |
|---|---|
| Getting the current submission state, best public score and the 2026-09-14 inference timing summary | [Inference and efficiency overview](inference-overview.md) |
| Checking official notebook, submission-limit, timeline, evaluation and efficiency-prize rules, or the submission CSV contract | [Submission requirements and output contract](submission-requirements-and-output-contract.md) |
| Building a frozen checkpoint/code/wheel bundle, choosing offline codecs and precision, or attaching verification evidence | [Repository readiness and model bundle](model-bundle-and-offline-environment.md) |
| Exporting a selected checkpoint without local prediction and launching detached submissions within Kaggle's GPU session limit | [Fast submission preparation](fast-submission-preparation.md) |
| Running training-to-submission recipes: setup, train, resume, export, infer, submit and status | [Justfile recipes](justfile-recipes.md) |
| Driving uploads, notebook runs, submissions and score polling from the Kaggle CLI, or assembling the notebook workflow | [Kaggle submission workflow](kaggle-submission-workflow.md) |
| Running or exporting dual-GPU inference: one process per visible GPU, four studies per GPU, 16 total workers, automatic FP16 backbone chunks on T4 and main-thread warm-up | [Independent GPU inference processes](independent-gpu-inference-processes.md) |
| Checking the physical-sampling preprocessing identity, seed compatibility and realized input counts | [Physical-sampling runtime identity](physical-sampling-runtime-identity.md) |

## Runtime and memory

| Read when | Document |
|---|---|
| Choosing the inference image multiplier and batch size from public scores and measured VRAM | [Inference image count and memory](inference-image-count-and-memory.md) |
| Choosing loader workers or precision on Kaggle's two T4s, or finding what slows inference: FP32/FP16, 4–24 workers, startup, indexing and per-stage DICOM loading costs over NFS | [2026-09-15 Kaggle inference profiling](2026-09-15-kaggle-inference-profiling.md) |
| Reviewing unverified submission timings, runtime leads, training-throughput anecdotes and open efficiency questions | [Unverified runtime evidence and efficiency leads](runtime-evidence-and-leads.md) |

## Submission records

| Read when | Document |
|---|---|
| Finding any of the 22 Kaggle submissions by date, public score or submission ID, and the dated record with its bundle, runtime and workflow details | [Submission ledger](submissions/INDEX.md) |
