# Inference and efficiency

| Read when | Document |
|---|---|
| Getting the end-to-end picture of how a checkpoint became a scored submission and which settings varied | [Inference pipeline overview](inference-overview.md) |
| Checking submission rules and the output contract | [Submission requirements and output contract](submission-requirements-and-output-contract.md) |
| Building frozen inference bundles, ensemble bundles, mirror TTA, backbone mode and the offline environment | [Model bundle and offline environment](model-bundle-and-offline-environment.md) |
| Finding train, export, submit and status command syntax, the no-local-prediction export routine and the batch-GPU concurrency limit | [Justfile recipes and submission preparation](justfile-recipes.md) |
| Operating the Kaggle CLI submission workflow, its known failures (stalled GPU notebooks, queue) and cancelling a notebook run | [Kaggle submission workflow](kaggle-submission-workflow.md) |
| Configuring GPU processes, loader workers, backbone precision and warm-up | [Independent GPU inference processes](independent-gpu-inference-processes.md) |
| Looking up the baseline4 image-count comparison and the measured inference memory per multiplier | [Inference image count and memory](inference-image-count-and-memory.md) |
| Checking what could differ on the hidden test set compared with training data and the three local examples | [Hidden-test risks](hidden-test-risks.md) |
| Interpreting the efficiency leaderboard: which selected submission it ranks, its one-day lag, the AUC-to-runtime rate, our rank history | [Efficiency leaderboard](efficiency-leaderboard.md) |
| Reading what competitors reported about runtime, and public-notebook inference techniques (all unverified) | [Competitor-reported runtime evidence and leads](runtime-evidence-and-leads.md) |
| Reading a dated Kaggle inference profiling or speed record: loader, startup, TensorRT, compile, buffer pool, bundle size, efficiency exchange rate and loading-cost attribution | [Inference profiling records](profiling/INDEX.md) |
| Finding a submission's ID, score, rank among ties or bundle contents | [Submission records](submissions/INDEX.md) |

## Common lookups

| Read when | Section |
|---|---|
| Finding train, resume, export and submit command syntax | [Recipe reference](justfile-recipes.md#recipe-reference) |
| Following a checkpoint export and submission example | [Export and submit](justfile-recipes.md#export-and-submit-a-checkpoint) |
| Inspecting detached submission progress and runner logs | [Submission status and logs](justfile-recipes.md#detached-submission-status-and-logs) |
| Cancelling a queued or running Kaggle notebook from the API, or finding its session id | [Cancelling a notebook run](kaggle-submission-workflow.md#cancelling-a-notebook-run) |
| Naming a submission and regenerating the ledger | [Naming](submissions/ledger.md#naming) |
