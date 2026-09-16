# Repository readiness and model bundle

## Repository readiness

The shared dataset, model and preparation code are reused for inference with
deterministic sampling, disabled augmentation and series dropout, and strict
checkpoint/config/target validation. `predict.py` validates mounted membership,
IDs, schema and probabilities; it accepts direct competition mounts and records
resolved device, precision and timing. `export.py` freezes those components and
offline dependencies, while `submit.py` versions the private Dataset, launches
the notebook, verifies its output and submits that exact version.

Local and visible Kaggle checks establish the baseline3 artifact described
below. They do not establish hidden-data mounts, compressed-image mix, full
runtime or a successful scored submission. Unsupported series are logged and
ignored; a study with no decodable supported acquisition fails explicitly.

Code: [prediction CLI](../../scripts/pipeline/predict.py),
[inference engine](../../src/knee_abnormality/inference/engine.py),
[checkpoint loading](../../src/knee_abnormality/inference/checkpoint.py),
[series index](../../src/knee_abnormality/data/sources.py),
[dataset](../../src/knee_abnormality/data/dataset.py),
[model](../../src/knee_abnormality/modeling/model.py), and
[configuration](../../src/knee_abnormality/config.py).

## Model bundle and offline environment

Freeze a selected fold checkpoint after reviewing its complete validation
results. A five-fold ensemble is optional; it is not required for the first
submission. Start with one checkpoint so the first submitted run measures a
small, reproducible pipeline.

The generated bundle contains:

- The selected checkpoint, or an export retaining `state_dict` plus the saved
  `hyper_parameters` config, targets and fold. The loader rejects bare weights
  without semantic metadata. Keep the training checkpoint locally; optimizer
  state is unnecessary for inference.
- The config recorded with that checkpoint and its ordered target list. Build
  this from saved run/checkpoint metadata, not a later edit of
  `configs/training.yaml`. Freeze model/data settings; permit explicit path and
  runtime overrides with their values recorded. The bundle `config.json` must
  carry an `inference` section with the frozen image multiplier; configs without it
  are rejected.
- A versioned copy of the shared Python package and both prediction entry
  points (`predict.py`, `predict_worker.py`), plus the checked-in
  [`scripts/pipeline/inference.ipynb`](../../scripts/pipeline/inference.ipynb) as the
  notebook. Export copies one shared file list and compares the same list with the
  training source snapshot.
- Offline wheels/requirements for dependencies missing from the selected
  Kaggle image, plus file hashes, dependency versions, source identity,
  checkpoint hash, label/split provenance and validation results.

Test-mode prediction reads the submission template and test metadata; it does
not call `load_fold_labels()`. Training CSVs, report-label sources, fold files,
training DICOMs and the original ImageNet download need not accompany the
inference bundle. Preserve their provenance with the trained artifact.

The exporter checks checkpoint metadata, reconstructs the model strictly on
CPU, and hashes packaged files. Readiness requires target Python/platform values
and all three decoder wheels. Export runs no local inference and records no
verification evidence, so a bundle claims neither numerical parity nor fresh codec
execution; the on-demand
[inference verification](fast-submission-preparation.md) is a separate diagnostic.
A required preprocessing identity matches the checkpoint
to the copied dataset/preprocessing sources, algorithm and reliability rules;
missing or mismatched identities fail export. Bundled prediction recomputes
that identity from its own sources without training-data access. Full-source
correspondence remains separately recorded. Historical checkpoints require
their archived source/bundles rather than export through current preprocessing.

`submit.py` still checks bundle integrity and refuses unmet packaging prerequisites.

For compressed images, a candidate offline decoder set is `pylibjpeg`,
`pylibjpeg-libjpeg` and `pylibjpeg-openjpeg`, with versions/wheels selected for
the target Python and NumPy environment. The first plugin covers JPEG Lossless
and the second JPEG 2000 according to the
[pydicom decoder table](https://pydicom.github.io/pydicom/stable/guides/user/image_data_handlers.html).
Verify real images using the listed syntaxes; the tiny example set need not
exercise every codec.

Inspect the notebook's Python, Torch/CUDA build, GPU compute capability and
compiled architecture support before selecting wheels. A local RTX 3090
bfloat16 run does not establish compatibility with another accelerator.
The installed Torch API provides
`torch.cuda.is_bf16_supported(including_emulation=False)` to distinguish native
support. Use a validated FP16 path or FP32 fallback when needed, and verify
that the Torch build itself supports the chosen GPU. The
[environment procedure](../architecture/environment.md#confirm-the-selected-kaggle-notebook)
records the relevant package inventory. Resolve/install from attached local
wheels with package-index access disabled; avoid blind upgrades of the whole
training environment inside a scored notebook.
