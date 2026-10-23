# Site-generalization campaign contract

## Objective

Make the model generalize to imaging sites it never trained on. The hidden test probably holds
sites absent from training ([sites and test composition](../../../docs/metadata/sites-and-test-composition.md)).
Trained without five inferred sites, the current recipe loses about 0.01 macro AUC on those sites
while losing almost nothing from the 23% less training data
([sites-out diagnostic](../../../docs/validation/results/validation-and-ensembles/2026-09-29-sites-out-diagnostic.md)).
Find training-time mechanisms that close that gap without costing seen sites.

## Recipe and reference

- **Recipe:** `configs/sites-out.yaml`, which is `configs/training.yaml` (RadImageNet ResNet-18,
  scanner confusion head at 0.1) with the split `all_5fold_scanner_v2_sites_out`.
- **Split:** `all_5fold_scanner_v2` without every study whose report language is Bulgarian,
  German, Greek, Dutch or French. Those five single-site clusters hold 1,036 studies. Training
  uses 2,683 studies; fold 0 holds 688 seen-site studies for checkpoint selection.
- **Reference and starting incumbent:** `sites-out-control` (`outputs/training/sites-out-control`,
  seed 1533697213, fold 0).
  - Scores: ×4 0.929085 on the 688 seen-site studies, ×4 0.899579 on the 1,036 held-out-site studies.
  - Timings: training 916 s, epoch 7 selected.

## Evaluation

The held-out-site evaluation of the
shared evaluation and adoption rule (its held-out-site section is in `autoresearch/protocol.md` on
branch `sites-out`), sealed in
`campaign.yaml`:
- **Primary:** ×4 macro AUC on all 1,036 held-out-site studies against the `rebuild_state_v2`
  evaluation targets, with the shared thresholds: +0.001 and a positive central 80% paired
  report-group bootstrap lower bound.
- **Guard:** the seen-site fold-0 ×4 macro AUC must not fall by more than 0.001.
- **Checkpoint selection** never sees the held-out sites.
- **Noise yardstick:** two replicate runs of the recipe on different hardware differ by 0.0012 on
  the primary metric.

## Budget

Range: each candidate's training time lies within 50–150% of the reference's 916 s, that is
458–1,374 s. The epoch count, batch size and optimizer steps may change within it. The studies
seen per epoch may not.

## Fixed

These stay unchanged unless the idea itself is the scanner-confusion retargeting, or the forced
ablation that removes the scanner-confusion term (user, 2026-09-29):
- **Evaluation:** split, fold, evaluation labels, gold overrides, validation and prediction
  multipliers.
- **Losses:** the disease loss and its class-balance weights, the half-knee, report-attribute and
  scanner-confusion terms and their weights.
- **Input:** image size 224, crop and sampling geometry.

## Permitted

- **Appearance randomization across training sites:** Fourier amplitude or feature-statistics
  exchange between studies of different sites, contrast, noise and resolution randomization, and
  field-strength simulation.
- **Class-conditional feature alignment:** CORAL or MMD between training sites within each label
  state.
- **Metric and contrastive terms:** angular-margin prototypes shared across sites, and cross-site
  supervised contrastive terms.
- **Robust objectives over training sites:** site-balanced sampling, GroupDRO, V-REx and Fishr.
- **Per-study normalization** of feature statistics.
- **Protection of pretrained features:** anchored decay toward the pretrained weights,
  freezing, linear-probe-then-fine-tune and per-stage rates.
- **Consistency regularization** between augmented views of a study.
- **Scanner or site confusion variants**, including retargeting the confusion head from scanner
  groups to site clusters.

## Training-site labels

Ideas that need a training site take it only from
`outputs/analysis/t/training-sites-20260929/study_languages.parquet`
(SHA-256 `8cc7d5628daf5a5e0cdeb70b4dc11b6eba84f0c9281fce4e0bc7a5aa81856fc3`):
- **Cluster rule:** each cluster is `language/vendor`, with Toshiba, Canon, FUJIFILM and Hitachi
  pooled as `minor`. Clusters under 30 training studies are pooled as `other`, which leaves 12
  clusters plus `other`.
- **Use:** training only, never read at prediction.
- **Validity:** the clusters are proxies. A language or vendor pair can span several institutions
  (English, Spanish, Turkish, Croatian).

## Excluded

- **Site, vendor, scanner model or field strength as a model input or conditioning signal at
  prediction.** An unseen site has no value to condition on.
- **Already measured and closed:**
  - gradient reversal against the scanner;
  - post-hoc per-site output centering;
  - per-model logit biases;
  - confusion-weight sweeps;
  - generic DSU, which is already in the recipe.
- **Project-wide exclusions:** weight averaging (SWA, EMA), image sizes above 224, ensembling,
  test-time augmentation, label or label-source changes, gold-study weighting, trained models in
  the loader, multi-seed or confirmation runs.

## Evidence requirements

- **Mechanism scalar:** every candidate logs its mechanism's own scalar per epoch, for the
  activation signature.
- **Per-site results:** the comparison records per-vendor strata on the held-out cohort. The
  researcher also reads each run's per-held-out-site AUC from `held_out_site_predictions.csv`.
- **Transfer:** adopted mechanisms reach main only through the user's usual full-data control on
  the production split. The public leaderboard can confirm them only if its share of the test
  holds unseen sites.
