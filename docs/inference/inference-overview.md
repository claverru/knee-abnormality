# Inference and efficiency

Evidence: official submission research and repository inspection through
2026-09-11, completed local submissions checked on 2026-09-14, plus unverified working notes;
see [provenance](../research/provenance.md).

Each current submission uses one trained fold. The implemented exporter
packages the checked-in offline Kaggle notebook with a frozen model bundle; the notebook runs the
shared inference path on mounted test data and writes `submission.csv`.
Baseline3 submission `56164711` scored **0.878** publicly, and baseline4
submission `56167669` scored **0.910** publicly at 1× image limits. Baseline4
at 2× scored **0.913** (`56169438`), ahead of 3× at **0.912** (`56169445`).
The latest submission uses **2× images, four studies per GPU, 16 total loader workers and
automatic FP16 with fixed backbone chunks** through the isolated dual-GPU runtime; it rescored
the champion at 0.936 ([record](submissions/2026-09-15-fast-champion-submission.md)).
The best recorded public result is **0.936**, from `reflection-sigma40`
submission `56230573` on 2026-09-14. See the
[validation comparison](../validation/results/2026-09-14-reflection-sigma-40-result.md)
and [submission artifact](submissions/2026-09-14-reflection-sigma-40-submission.md). The older commands
in this branch still describe its original batch-2 exporter; reproduce the new
submission with its recorded runtime or frozen bundle.

## 2026-09-14 inference timing investigation

The [generalization and inference investigation](../history/sharing/generalization-inference-clocks.md)
compares reflection, 1A, 1B and original 1C using their actual GPU, worker and
batch settings. Submission-to-completion waits of 13m36s–17m38s include queue
and scoring delays; they do not establish hidden-data inference speed or the
efficiency-prize runtime. The saved visible notebooks contain only three studies.

Static inspection identifies duplicated full-test metadata indexing per GPU,
shared CPU/I/O limits, process startup and uneven study costs as scaling
hypotheses. All four Kaggle runs use FP32 on T4. The report proposes a controlled
Kaggle profiling matrix and explains the official accuracy/runtime tradeoff.
No local GPU work or new remote benchmark was run for this investigation.
