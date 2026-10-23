# Second round: averaged labels, fastMRI bone masks and distilled students (2026-10-13 to 2026-10-14)

**ConvNeXt-Tiny pretrained on fastMRI and then trained jointly beat the ResNet-50 joint model on
every fold (+0.0044 mean ×1); the five ConvNeXt folds scored 0.964 public (S155), the ten
ConvNeXt + ResNet-50 models also 0.964 and sorted first (S154), and a single ConvNeXt student
distilled from them 0.964 (S156). τ and the image multiplier were flat locally, so ×4 and τ 0.85–0.9
stayed.**

## Setup

- **Pseudo-labels:** the five teachers of main's recipe at `288e50a`, ×4, τ 0.9, mirror TTA, 41
  minutes; `/workspace/predictions/fastmri/rn50-main-288e50a-5fold-x4-tau09-tta/` (scripts in
  `source/`) and `data/labels/prepared/fastmri-rn50-main-288e50a-x4-tau09-tta/`. Against the first
  round, same study and fold: correlation 0.978 to 0.995 per label, mean absolute difference 0.015
  to 0.026 (the spread between teachers inside one round is 0.018 to 0.030).
- **Averaged labels (user decision):** `training_table_averaged.parquet`, the mean of the five
  teachers per study repeated for every fold, 9,828 studies. Not held out: fold scores of runs on
  it are inflated through the four teachers that trained on the fold and are not comparable with
  earlier runs.
- **Class weights in a fastMRI-only stage (user decision):** from the fastMRI labels, each label's
  mean soft target as prevalence (`fixed_class_balance(..., soft_prevalence=True)`, the formula of
  the targets-and-inputs soft-prevalence candidate). Joint stages keep the Kaggle rows with the
  thresholded count.
- **Bone masks:** every run reads `anatomy-mrannotator-sample550-fastmri1230-joints`. In the
  fastMRI-only stage's first epoch the bone map reaches 0.90 agreement on 1.81 million mask tokens
  and the crops sit at +0.4 px (sagittal) and +2.7 px (coronal) with 3.7° turns, against 0.08, −33
  px, −28 px and 8.4° without fastMRI masks.
- **Chain on the rented machine** (`/workspace/outputs/logs/night_chain.sh`, state in
  `night_chain.state`), the `fastmri-pretraining` worktree's merged code, seed 1110484336:
  1. `convnext-pretrain-fastmri-avg-labels-masks`: ConvNeXt-Tiny tuned recipe, fastMRI only, ten
     full passes, nominal fold 0;
  2. `rn50-joint-avg-labels-masks-folds-0-4`: main's ResNet-50 recipe, joint, five folds;
  3. `convnext-pretrained-joint-avg-labels-masks-folds-0-4`: ConvNeXt joint from the pretraining's
     last epoch, five folds;
  4. τ search (`tau_predict.py`, `tau_score.py` in `/workspace/outputs/logs/`): τ 0.70 to 1.00 by
     0.05 at ×4, ×5, ×6, no mirror, out-of-fold, per backbone and for their equal blend, results in
     `/workspace/outputs/analysis/tau-search/`.
- **τ and caching:** τ also moves the first look (frame and crop poses; up to 2 px at τ 0.7), so
  pooling cached tokens at another τ is off by up to 0.29 logits. Sharing only the first look's
  backbone pass between τ values is not exact either: it failed its check on ResNet-50 fold 0
  (0.0038 logits; per τ mostly under 0.0002 with single jumps of 0.0065 and 0.028, a discrete pose
  choice flipping), while two direct passes agree within 0.00005. It saved only ~16% of the
  backbone images (224 of ~1,200 per batch), so the search runs the direct path for every τ
  (`tau_predict.py`; the cached version kept as `tau_predict_cached.py`).

## Results

Fold scores are inflated by the averaged labels (four of the five teachers saw each validation
fold); compare runs only within this round.

| ×1 macro AUC | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean |
|---|---|---|---|---|---|---|
| ConvNeXt pretrained, then joint | 0.940925 | 0.932857 | 0.937863 | 0.933431 | 0.934253 | 0.935866 |
| ResNet-50 joint | 0.936159 | 0.928819 | 0.934051 | 0.928780 | 0.929416 | 0.931445 |

- ConvNeXt ahead on every fold, +0.0038 to +0.0048 (mean +0.0044). Training 47.6 to 50.1 min per
  fold for the ConvNeXt, 43.8 to 46.0 for the ResNet-50; pretraining 62.5 min, ×1 0.932150 on fold
  0 (its own fastMRI-only model). The ResNet-50's own ×4 per fold: 0.941205, 0.936327, 0.939919,
  0.934879, 0.935158 (mean 0.937498). No windowed-resampling fallback in any run.
- **τ and multiplier (ResNet-50, five folds, mean of folds):** flat. Best ×4 τ 0.85 0.937275
  against ×4 τ 0.9 0.937215 (+0.00006); the whole grid spans 0.0003; ×5 and ×6 never beat ×4.
  The same weights at ×4 τ 0.9 score 0.937498 in the runs' own prediction, 0.0003 apart from the
  search's (different image draws): the draw noise is as large as the whole τ effect.
- **Per fold** the best setting moves without a pattern shared with the model: ResNet-50 folds 0
  to 4 prefer τ 0.70, 1.00, 0.85, 0.70 (×4) and 0.90 (×6), gains over ×4 τ 0.9 of +0.00031,
  +0.00005, +0.00009, +0.00007, +0.00065. ConvNeXt folds 0 to 2 prefer ×4 τ 0.75, 1.00, 0.85
  (+0.00013, +0.00009, +0.00008); ×4 beats ×5 and ×6 by 0.0005 to 0.0008 on each. The leaning of
  τ follows the fold (fold 0 low, fold 1 high, fold 2 middle for both backbones). Decision: keep ×4
  τ 0.9.
- **Equal blend vs the ConvNeXt alone (×4 τ 0.9, folds 0 to 2):** 0.944084 vs 0.944282, 0.938865
  vs 0.938333, 0.942621 vs 0.942928; the blend does not beat the ConvNeXt alone here (−0.0002,
  +0.0005, −0.0003).
- The ConvNeXt τ predictions were stopped by the user after folds 0 to 2 (×4, ×5, ×6) and fold 3
  (×4); scores in `/workspace/outputs/analysis/tau-search/` (per-fold tables from
  `tau_per_fold.py` there). `last.ckpt` removed from the two five-fold runs; the pretraining's
  `last.ckpt` kept, it is the joint run's recorded starting weights.

## Distilled students on all Kaggle folds

Single students trained on all five Kaggle folds at 0.25 teacher share from the ten-model S154
ensemble, on the fastMRI mean-of-10 labels; ×4, τ 0.85, no TTA, batch 4, 8 workers. Fold-0
validation is inflated by training on that fold.

- **S156, the ConvNeXt-Tiny student: 0.964 public**, second of the three 0.964 in the
  score-sorted list (S154 first, S155 third).
- **S157, the ResNet-50 student (`rn50-distilled-student-full-data`, ×1 0.951254 on fold 0): 0.962
  public**, first of the 0.962 (above S148), below S152 (0.963). The ConvNeXt student is ahead
  by one rounding step. Two notebooks hung in `notebook_running` (24 and 7 min, against 1.5 to
  2.5 min for earlier single ResNet-50s at ×4) and were deleted, and one relaunch failed on the
  deleted slug; the fourth (`claverru/s157-resnet-50-distilled-student-x4-tau0-85-c`) ran in
  91 s.

**S154 and S155:** the ten ConvNeXt + ResNet-50 joint models (`convnext-pretrained-joint-avg-labels-masks-folds-0-4`,
`rn50-joint-avg-labels-masks-folds-0-4`) and the five ConvNeXt alone, ×4, τ 0.85, mirror TTA,
batch 2, 8 workers: both 0.964, S154 sorted above S155 although out of fold adding the five ResNet-50s read
silver −0.0004 and gold −0.0003 (S155 0.940901 / 0.947541, S154 0.940478 / 0.947291). S153, the same ten, hit a Kaggle system error
after the output was verified and was resubmitted by hand from the same notebook version as S154.
