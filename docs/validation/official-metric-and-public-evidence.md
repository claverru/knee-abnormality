# Official metric and public evidence

The [official evaluation page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)
specifies the unweighted mean of twelve per-target ROC-AUC values. Submit one
row per `StudyInstanceUID` with confidence scores for the twelve target columns
in `sample_submission.csv`; retain continuous scores rather than thresholding
for F1.

Public experiments demonstrate why the evaluation label source matters:
hard-label training can win against a hard-label surrogate while soft-label
training has a better point estimate on independent radiologist labels. See
[public supervision evidence](../research/public-research.md#report-derived-supervision)
and [validation evidence](../research/public-research.md#validation-and-ensembling) for the
paired results, uncertainty, duplicate-report grouping, scanner stress tests,
and public-LB tuning failures. These observations informed the
[project protocol](project-validation-protocol.md); their reported scores have
not been reproduced locally.
