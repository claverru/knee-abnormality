# 2026-09-14 sigma-60 result

`outputs/training/sigma60` completed ten epochs and selected epoch 10 (index 9), scoring
0.892378929 full-fold AUC versus 0.892538106 for the overnight sigma-40 run.
The -0.000159177 difference is too small for a reliable ranking from one fold
and seed. Retain sigma 40 as the submitted incumbent, without treating sigma 60
as an established regression. Both outperform the original sigma-80 1C control
locally (0.889429588 AUC).

| Metric | Sigma 40 | Sigma 60 | Delta |
|---|---:|---:|---:|
| Full-fold AUC, 904 studies | 0.892538 | 0.892379 | -0.000159 |
| Soft-target BCE | 0.448244 | 0.448567 | +0.000323 |
| Gold AUC, 11 studies | 0.917229 | 0.927414 | +0.010185 |
| Training seconds | 1739.16 | 1683.84 | -55.32 |

Seven targets improve and five decline. Largest gains are MCL (+0.011223),
synovitis (+0.005180) and Baker's (+0.004583); largest declines are contusion
(-0.011075), effusion (-0.006212) and fracture (-0.005738). Both meniscus targets
improve. Gold remains diagnostic because it contains only eleven studies.
Peak allocated CUDA memory was 20,392,964,096 bytes versus 20,681,703,936 for
sigma 40; runtime and memory differences are descriptive, not isolated effects.

Saved effective configs differ only in sigma (60 versus 40), names and output
paths. Complete input identities match. Both use 1C, seed 42, Steven v4, fold 0,
224 pixels, 140 mm, reflection 0.5, baseline LRs/normalization, 2× validation
images and 4,380 optimizer steps. Package source differences are confined to
`inference.py` and the added `inference_processes.py`; model/training source
matches. Evidence is each run's effective config, source snapshot, input
identity, selected-checkpoint record, epoch metrics and timing JSON. Sigma 60's
selected checkpoint is `fold_0/best-epoch=09-val_macro_auc=0.89238.ckpt`.
No new training, inference or tests ran for this comparison; sigma 60 has not
been submitted in this workflow.
