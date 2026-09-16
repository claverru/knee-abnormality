# Ensemble interpretation

Earlier notes identified prediction-scale differences between losses as a
problem for combining logits or probabilities. Rank ensembling was mentioned
as an approach already tried, but its implementation and result are absent.
Hard voting was also considered without a resolved method; that speculation
should not be used as its definition.

Track per-target behavior and complementary errors alongside aggregate scores.
The [supervision notes](../modeling/supervision.md#downstream-benefit-changed-over-time)
also record an observation that improved labels could make
models more correlated and reduce their ensemble contribution.

A strictly increasing transform of one model's predictions preserves each target's ROC
AUC, so calibration or threshold tuning alone is not an AUC gain. Clipping, averaging and
different transforms across models can change rankings; evaluate them on held-out
predictions. A stacker fitted to OOF predictions needs held-out or nested evaluation of
its own, and every learned component that uses evaluation labels must be fold-isolated.
The [code review](../research/code-and-notebooks.md) shows where artifact names or fold
loops alone do not establish that isolation.
