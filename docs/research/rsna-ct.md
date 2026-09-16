# Transferable methods from RSNA CT competitions

Borrow the separation between image encoding and volume aggregation. CT
competition solutions provide useful implementations for this, but their
windowing, slice labels, and organ masks do not transfer directly to MRI.
For closer imaging comparisons, read [RSNA MRI competitions](rsna-mri.md).

## Intracranial hemorrhage, 2019

Darragh Hanley and Dmitry Larko's second-place NoBrainer solution trained
single-slice CNNs, extracted global-average-pooled embeddings, and modeled
sequences with an LSTM. The sequence stage used differences between neighboring
embeddings as additional features. The authors describe CT-specific intensity
windows, five split folds with three used for their models, and a crop change
that preserved score. They list image/embedding mixup, partial or reversed
sequence augmentation, and 1D convolutions among unsuccessful experiments.
[1](https://github.com/darraghdog/rsna)

**Transfer proposal.** Cache embeddings from a fixed image encoder to compare
mean/max pooling, attention, and a small RNN cheaply. If the encoder is trained
on competition labels, build the cache separately for each validation fold.
A single cache from a model trained on all studies cannot support held-out
evaluation of the head. A frozen-feature screen also cannot establish the ranking after
end-to-end fine-tuning.

Neighbor differences are an inexpensive optional feature, but compute them only
within a consistently ordered series. Their interpretation changes with slice
spacing and contrast. Treat series boundaries, padding, and missing slices
explicitly; do not treat a bag of MRI series as one CT sequence.

There is also a supervision mismatch: hemorrhage training includes image-level
labels. Here, copying a study's positive label onto every slice would assert
that the finding is visible everywhere. A study-level loss over pooled window
features avoids making that assumption. It does not eliminate report-label noise.

## Abdominal trauma, 2023

Theo Viel's second-place solution combines 2D models and RNN aggregation with
organ crop models. Three adjacent frames form the input channels. Sampling uses
available organ or positive-frame information; organ crops are processed as
short sequences. The aggregation stage incorporates classification and
segmentation probabilities and their products, then uses organ-aware pooling.
The author reports that crop models helped solid-organ targets. These are
solution reports, not isolated knee-relevant AUC ablations.
[2](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma)

Team Oxygen's first-place repository independently documents a segmentation
preparation stage, organ volumes, 2D/2.5D encoders with GRU aggregation, and
patient-level aggregation before ensembling. It reports best single-model
four-fold OOF loss **0.326** and ensemble OOF loss approximately **0.31x**.
Those are the trauma competition's losses, not macro AUC; the ensemble also
changes model count and training membership.
[3](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution)

**Transfer proposal.** Distinguish finding evidence from anatomy presence. If
a soft anatomical gate identifies a meniscus or compartment, use it to weight
features while retaining uncertainty and a global fallback. Measure whether
the gate discards positives. A segmentation score alone cannot establish that
the selected windows retain the evidence needed for knee classification.

For a low-cost first comparison, use deterministic plane/contrast slots and
physical windows rather than train a new segmenter. Escalate to a learned
localizer only if baseline errors repeatedly show missed anatomy. Any
auxiliary segmentation branch needs actual masks or explicitly qualified
pseudo-masks; it cannot inherit trauma's supervision just by copying the model.

## What the inspected code establishes about OOF

The reviewed `kfold_inference()` loads a fold-specific model and restricts image
rows to that fold. It saves fold predictions and averages image predictions by
`patient_id` for one diagnostic evaluation. `predict_distributed()` can return
features as well as predictions. This verifies a fold-held-out inference path;
it does **not** prove the training checkpoint, upstream segmenter, or all
downstream features are free of leakage.
[4](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma/blob/6e9d193cb60b11c2a3c1597d430d53615173b441/src/inference/extract_features.py)

For this repository, the [current split](../validation/project-validation-protocol.md)
assigns every study to one validation fold and can produce OOF predictions for
all 4,407 studies. The older `gold_5fold_v1` protocol shared most report-only
studies across training folds; predictions from that historical pool must not
be relabeled as OOF. This distinction matters more than choosing the historical
solution's exact RNN.

## Sources

1. Darragh Hanley and Dmitry Larko. *RSNA 2019 Intracranial Hemorrhage Detection — NoBrainer*, second-place author repository; README reviewed.
2. Theo Viel. *RSNA Abdominal Trauma Detection*, second-place author repository, 2023; README reviewed. The associated [discussion](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447453) is a navigation link; its body was not independently retrieved.
3. Nischay Dhankhar / Team Oxygen. *RSNA-2023-1st-place-solution*, author repository, 2023; README reviewed.
4. Theo Viel. `src/inference/extract_features.py`, commit `6e9d193cb60b11c2a3c1597d430d53615173b441`; source inspected, not executed.

Reviewed on 2026-09-10. CT Hounsfield-unit windows, class weights, and
competition-specific consistency rules are deliberately not proposed as MRI
defaults.
