# Validation–Kaggle correlations: reproduction and limits

Data and calculations: NumPy/Polars/SciPy/scikit-learn, `CUDA_VISIBLE_DEVICES=''`, no torch/model
execution. Join frozen references and per-epoch CSVs by ID, assert identical 904-study membership,
then compute per-label ROC AUC and its unweighted mean. Pair-block contribution is
`mean_j[(P_block,j * N_block,j)/(P_j * N_j) * (AUC_new,block,j - AUC_old,block,j)]`.
Coverage-based blocks use the original v1 indicator; alternative-source scores replace gold cells
with frozen official labels. Subgroup scores retain all twelve labels and require both classes.

Fixed difficulty anchors are selected predictions from `ranked-filtered-seed43`,
`ranked-filtered-seed44`, and `min_lr_01`. Difficulty is the average fraction of opposite-class
rank errors per label, half credit for ties; its median is approximately 0.0997. Subgroup
permutations shuffle the same 14 public scores jointly across all diagnostics, seed 20260920.
Cached pooling uses attention `softmax(score / τ)`, weighted saved contributions, saved readout
bias and sigmoid; per-type ×2 budgets are (12,12,10,10,10,10), doubled for ×4 and capped by availability.

Primary artifact locations, relative to repository root:

- `data/folds/all_5fold_v3/{studies.csv,targets.csv,protocol.json}`; SHA-256 respectively
  `c709cd27ff54dd10d833abfa966622f9a5ecadef5d37670da658d81c26f7b3ed`,
  `0e48fbc11f2824db2d04c83a55f8a2d0067f58ebd43cf4142d8166c2ace91a7c`,
  `facd4494eeab7d5ff58f0f5b5440a634256bc64c12f99175e02f27e3cf6b6aab`.
- Historical runs and scores: `scripts/adhoc/submission_public_scores.json`, SHA-256
  `cc2137685e10632fd3c5a57f4cf7b8b5b10cb5f31156114f240367fc1fdb9cc3`.
  Every run supplies `effective_config.json`, `fold_0/epochs/metrics_epoch_*.json` and aligned
  `validation_epoch_*.csv`; use saved source snapshots/input identities rather than current source.
- Main ordinary CSV: `outputs/training/overnight-20260918T232636Z-mt7ouybk/runs/rexnet-fold0/fold_0/epochs/validation_epoch_009.csv`,
  SHA-256 `c34a26ba28bbca1b770f363b658aa1075901f4d9790f0707a8f45833f0d654a8`.
  Control-B CSV: `outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/epochs/validation_epoch_008.csv`,
  SHA-256 `c523e3397cdd070446df68be2b122b377108bf53dd8a07562236181768a4652b`.
- Alternative labels: `data/labels/public/stevenleehans/version-6/llm_labels_full.csv`,
  `data/labels/public/pilkwang/version-1/report_labels_v2.csv`,
  `data/labels/public/lixin73/version-1/labels_llm_gpt56sol.csv`,
  `data/labels/prepared/tonylin-soft-v3/labels.csv`; source identities in
  [public-label configuration](../../../../configs/labels/public-label-sources.json).
- Study headers: `outputs/analysis/s/sharing-disagreement-20260914/study_headers.parquet`;
  counts: `data/metadata/train_series_image_counts.parquet`; geometry/slabs:
  `outputs/analysis/p/peripheral-coverage-20260918/`; tokens/selections/series:
  `outputs/analysis/t/token-cache-20260917/`. Headers represent one series per study, which limits
  scanner/geometry interpretations. Report summaries use `data/raw/train.csv`; no raw text is reproduced.

Public scores are rounded to three decimals; models, checkpoints, labels and diagnostics were
adaptively selected and are not independent observations. The newer cohort has only five models.
Masked endpoints alter the case mix and rare-label sample sizes; alternative report labels are
not independent image truth. Downloaded notebook submission CSVs contain only three example
studies, not hidden-test predictions, so neither hidden-test calibration nor per-label attribution
can be measured. No private-leaderboard behavior is established. The useful result is a concrete
local failure mode to test prospectively, plus evidence against replacing validation with a
retrospectively favorable label or study subset.
