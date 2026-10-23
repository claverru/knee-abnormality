# Public code and notebooks worth reusing

These are source inspections, not executed notebooks or reproduced solutions.
The [manifest](source-manifest.json) pins downloaded bytes to Git revisions.
Notebook cell numbers are **zero-based source-cell indices**, including Markdown
cells. Historical Kaggle scores and attached artifact lineage remain separate
claims.

Start with the smallest relevant component: series aggregation, crop generation,
or annotation overlays. Preserve the project's Polars and Lightning choices
when implementing an idea; an old notebook's framework is not a project requirement.

## Lumbar coordinate crops

**Source:** Theo Viel, `notebooks/CoordsInf.ipynb`, commit
`0e4dbea8ce16249792ec64cefa4cd5510eda7cdb`.
[Notebook](https://github.com/TheoViel/kaggle_rsna_lumbar_spine/blob/0e4dbea8ce16249792ec64cefa4cd5510eda7cdb/notebooks/CoordsInf.ipynb).

Cell 7 loads each of four coordinate checkpoints, selects rows with the same
held-out fold, and saves predictions. Cells 13–17 deliberately create two crop
variants. With `FIX=True`, `fix_predictions()` replaces a sufficiently inaccurate
predicted coordinate with its supplied reference coordinate; `FIX=False` retains
predictions. Cell 17 then crops each slice stack around the coordinates.

This establishes both a fold-held-out prediction path and a reference-assisted
crop path. It does **not** establish that the final classifier's reported
validation used corrected crops. Reuse requires tracing which directory each
training and validation dataset reads. Evaluate deployment performance using
predicted crops; an oracle-coordinate experiment is a separate upper-bound
diagnostic. The current knee dataset has no analogous coordinate table.

## Lumbar second-stage aggregation

**Source:** same revision, `notebooks/Training_Lvl2.ipynb`.
[Notebook](https://github.com/TheoViel/kaggle_rsna_lumbar_spine/blob/0e4dbea8ce16249792ec64cefa4cd5510eda7cdb/notebooks/Training_Lvl2.ipynb).

Cells 8–11 join study folds and build a feature dataset from multiple experiment
outputs. Cells 15 and 19 select a `simple` aggregation model; cell 19 specifies
seed 42, four folds, and the task's weighted study loss with its any-severe
component. Despite an introductory RNN description, this notebook configures
the later aggregation stage. Cells 20–22 change execution with `DEBUG` and
enable full-data fitting when debug is disabled.

Borrow the explicit separation between base-model outputs and final aggregation.
Do not copy its metric-specific loss or assume files named `oof` are independently
generated. The upstream models, crop variants, feature arrays, and study joins
must all have the intended membership. The inspected notebook depends on
external experiment folders; their contents were not audited.

## Aneurysm regional pooling and submission

**Source:** tomoon33 / uchiyama33, commit
`e1dcdf0058e1e0d0044d8053e92243b4b4794555`.

The [`RegionMaskedPooling3D` source](https://github.com/uchiyama33/rsna2025_1st_place/blob/e1dcdf0058e1e0d0044d8053e92243b4b4794555/src/models/components/region_mask_pooling.py)
supports region-weighted mean/GeM/max features, optional mask dilation, and
global pooled features. It resizes masks to the feature grid and protects the
mean denominator against empty masks. These are implementation capabilities,
not evidence that every option was active or beneficial in the final solution.
A zero-denominator safeguard also does not establish useful predictions when
an anatomical region is missing.

The [submission notebook](https://github.com/uchiyama33/rsna2025_1st_place/blob/e1dcdf0058e1e0d0044d8053e92243b4b4794555/rsna2025-submission-1st-place.ipynb)
packages dependencies offline in cells 1–4. Cell 7 specifies segmentation/ROI
settings, classifier folds 0–3, two-way TTA, failure thresholds, and fixed fallback
probabilities. Cell 8 starts the competition-specific inference server and uses
`submission.parquet` in local evaluation. The knee competition instead has its
own [CSV submission contract](../../inference/submission-requirements-and-output-contract.md#verified-submission-requirements).

Borrow configurable stages, shared regional features, and explicit failure
handling. Re-estimate any fallback from permitted training data; do not import
aneurysm prevalence or return normal predictions for failed knee decodes.
Neither the configured TensorRT artifacts nor timing claims were reproduced.

## fastMRI+ annotation overlay

**Source:** Microsoft/fastMRI+ authors, `ExampleScripts/example.ipynb`, commit
`67ed9a68ee33d3197f93893e8125688470c4a577`.
[Notebook](https://github.com/microsoft/fastmri-plus/blob/67ed9a68ee33d3197f93893e8125688470c4a577/ExampleScripts/example.ipynb).

Cells 1–2 select annotations by image file. Cell 3 reads `reconstruction_rss`
and reverses the image's row axis. Cells 4–5 overlay slice-specific boxes using
`x`, `y`, width, and height. This provides a concrete coordinate-convention
reference for the annotation release; the flip is not a universal MRI transform.
It is an annotation visualization notebook, not a trained classification baseline.

Borrow the practice of displaying labels on the final transformed pixels.
If adapting this data, transform boxes along with the images and preserve source
slice identity. The notebook requires separately acquired fastMRI images; no
such acquisition or visual overlay was run here.

## Brain tumor input selection

**Source:** Firas Baba, commit `db805687a453d3f4174301bc3385d9c3e288482e`;
[`working/dataset.py`](https://github.com/FirasBaba/brain-tumor-radiogen/blob/db805687a453d3f4174301bc3385d9c3e288482e/working/dataset.py)
and [`working/config.py`](https://github.com/FirasBaba/brain-tumor-radiogen/blob/db805687a453d3f4174301bc3385d9c3e288482e/working/config.py).

The defaults specify 64 images at 256 pixels. `_prepare_biggest_images()` chooses
an index using a cropped-image-size helper. `load_dicom_images_3d()` uses that
index for the training path and the stack midpoint for inference, then pads
short volumes with zeros. Both paths numerically sort filenames. The helper's
precise image-content criterion was not inspected.

This is useful evidence that a winning repository can contain asymmetric
sampling assumptions. It is not a reason to copy filename ordering into the knee
pipeline. Use [physical DICOM geometry](../../data-preparation/unverified-preprocessing-notes.md#intensity-and-geometry),
and explicitly record any intentional train/evaluation sampling difference.

## Trauma fold-held-out feature extraction

**Source:** Theo Viel, commit `6e9d193cb60b11c2a3c1597d430d53615173b441`;
[`src/inference/extract_features.py` (removed)](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma/blob/6e9d193cb60b11c2a3c1597d430d53615173b441/src/inference/extract_features.py).

Inspect `predict_distributed()` for feature-returning inference and
`kfold_inference()` for loading per-fold checkpoints and selecting validation
rows. The [CT review](rsna-ct.md#what-the-inspected-code-establishes-about-oof)
records exactly what this does and does not prove. Trace study IDs through every
stage before training a head or repairing labels with the resulting features.

## Experiment-tracking workflow notes

A dashboard workflow linked ideas, experiments, local metrics, and Kaggle submissions. Models described as `tabfpn3` and simpler alternatives were fit to local/public results to predict promising combinations. No training data, model definition, or predictive-accuracy result for this meta-modeling workflow was recorded. Interactive notebooks were useful for inspecting image/label pairs and preprocessing. A reported failure mode of extended automated experimentation was producing more benchmark runs without explaining why models with similar metrics behaved differently. The actionable lesson is to compare their predictions and failure cases, not only collect aggregate scores.

## Provenance and limits

The four notebooks and four Python files above were downloaded as text on
2026-09-10. The manifest also pins two related README/write-up files. Only the
named cells/functions and surrounding control flow support the observations;
this is not a full audit of the authors' repositories. No external code,
training, inference, or tests were run.

Immutable source links identify the inspected artifacts, while the
[research index](../INDEX.md) and topic bibliographies identify their associated
papers and author-reported results. A future Kaggle pull may differ from these
GitHub releases.
