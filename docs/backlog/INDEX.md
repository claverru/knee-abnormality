# Backlog

Ideas not yet started, one document per experiment type. The
[work-tracking rules](../../AGENTS.md#work-tracking) say where a new idea goes and how entries
leave; active work lives in [ongoing work](../ongoing/INDEX.md).

| Read when | Document |
|---|---|
| Planning or reviewing the next overnight queue: higher head learning rate, denser training inputs, other hyperparameter-only candidates and the runs already measured | [Overnight queue](overnight.md) |
| Recording or picking a small code addition behind one new option, implemented only when the user asks: superseded retakes, the rejected budget reallocation and head changes, several heads per label, gradient-norm logging, 3D-embedding training effect, label-mask and target tweaks, BCE weighting and study/batch sampling, ranking loss, late affine shutdown, augmentation ordering and simulated low resolution, contrast slot relabel, per-plane spacing | [Small changes](small-changes.md) |
| Recording or planning an investigation-sized effort for a focus session with the user: position and knee side, medial/lateral transfer, learned and sampling-independent intensity bounds, robust study center, stride-16 feature fusion, and parked ideas (adaptive training allocation, external data, label refinement, 3D and dense tasks) | [Big bets](big-bets.md) |
| Accelerating DICOM loading over Kaggle's NFS: where the cold-start time goes and what the profiler gets wrong, the open candidates by retrain cost, the closed regressions, how to measure one, and prior art from past RSNA competitions | [Loading speedup](loading-speedup/INDEX.md) |
| Preparing the final phase: fold and seed ensembles, full-data fits, bigger images, test-time augmentation, EMA or SWA weight averaging and distillation | [Endgame](endgame.md) |
| Debugging a failed or surprisingly low Kaggle submission, or adding a rule that depends on test data: known hidden-test risks (crashes on unusable studies or unseen contrast flags, header gaps, series filters, numerics, public-score noise) | [Test-time risks](risks.md) |
