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
results. Start with one checkpoint so the first submitted run measures a
small, reproducible pipeline. `export.py` also accepts more than one
checkpoint to build a multi-checkpoint ensemble bundle; see
[below](#multi-checkpoint-ensemble-bundles).

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
  are rejected. Bundle-mode prediction applies this frozen multiplier to both
  the test and validation splits (unlike direct `--checkpoint` prediction,
  which uses the config's saved validation multiplier for the validation
  split); export with `--image-multiplier 1` to reproduce a true x1
  validation score through a bundle.
- A versioned copy of the shared Python package and both prediction entry
  points (`predict.py`, `predict_worker.py`), plus the checked-in
  [`scripts/pipeline/inference.ipynb`](../../scripts/pipeline/inference.ipynb) as the
  notebook. Export copies one shared file list and compares the same list with the
  training source snapshot.
- Offline wheels/requirements for dependencies missing from the selected
  Kaggle image, plus file hashes, dependency versions, source identity,
  checkpoint hash, label/split provenance and validation results.

### Multi-checkpoint ensemble bundles

`export.py --checkpoint` takes a comma-separated list of one or more
checkpoints. One checkpoint keeps
every current behavior exactly: `model.ckpt` at the bundle root and
`manifest.json`'s singular `checkpoint` record. More than one checkpoint
builds an ensemble bundle instead:

- Member checkpoints are stored as `models/member-000.ckpt`,
  `models/member-001.ckpt`, ... in export order, and `manifest.json` carries a
  plural `checkpoints` list (one record per member: source path/hash, fold,
  its own `model` config, its baked `attention_temperature` and
  `preprocessing_identity`, results, provenance) instead of the singular
  `checkpoint`/top-level `preprocessing_identity` keys. `config.json`'s
  `experiment` section carries the first member's full config, since
  `ExperimentConfig` has one `model` section; each member's own config lives
  in its checkpoint and in the manifest.
- Every member must agree on the whole `data` section (`reference_field_mm`,
  `image_size`, `series_types`, `study_slice_total`,
  `sampling_sigma_fraction` and the rest) because they share one data loader
  and one decoded batch; export refuses a mismatch and names the differing
  field. The `model` section, including `backbone`, may differ per member —
  the spatial token mapping is backbone-specific and lives downstream of the
  CNN, so mixed backbones share the decode correctly.
- Members may come from different folds. This is expected for a real
  ensemble: a five-fold bundle has one member per fold, each held out on a
  different split, meant for test-time prediction rather than for scoring
  against any single validation fold. Requesting `--split validation` against
  an ensemble bundle validates every member against the requested fold and
  fails loudly for any member not actually held out on it.
- `predict_worker.py` loads every member once per worker (kept resident for
  the worker's life) and runs all of them against the same decoded,
  device-resident batch, averaging their per-label sigmoid probabilities
  (mean of probabilities, not ranks or logits — measured 0.88269 vs 0.88244
  macro AUC on this project's 5-fold OOF). Decoding stays single-pass, so
  extra members cost close to their own forward time rather than a repeated
  load; see the
  [x4 loading profile](2026-09-18-kaggle-x4-profile-after-index-changes.md)
  for the loader-wait/forward split this assumes. A one-member bundle is a
  mean over one value, bit-identical to a plain forward pass.
- `--attention-temperature` accepts one value (applied to every member) or
  exactly one value per checkpoint, matched by position. A one-member bundle
  applies it at load time exactly as before. An ensemble bundle bakes each
  member's temperature into its exported weights at export time — the same
  division of the attention scorer's weight and bias that load-time
  application performs today, just applied once before saving — so
  `manifest.json`'s `runtime.attention_temperature` is recorded as `1.0` (an
  honest no-op at load time, since the scaling already happened); each
  member's real baked value is in `manifest.checkpoints[i].attention_temperature`.
- Verified locally (2026-09-19, `outputs/training/overnight-20260918T232636Z-mt7ouybk`):
  exporting a 2-member bundle from `rexnet-fold0` (val 0.88367) and
  `b0-fold0` (val 0.88635) at `--image-multiplier 1` and predicting fold 0's
  904 validation studies reproduced the mean of the two members' own
  single-checkpoint bundle predictions to within 7e-8 (float32 rounding,
  not a real difference). Macro AUC (`roc_auc_score` against
  `data/folds/all_5fold_v3/targets.csv` binarized at 0.5) was 0.8917 for the
  mean against 0.8837 and 0.8863 for the two singles, which closely reproduce
  their checkpoint filenames' recorded validation scores. A 1-member export
  was confirmed to keep the pre-ensemble `manifest.json`/`config.json` schema
  and to reproduce a plain forward pass bit for bit.

### Reflection test-time augmentation (TTA2)

`export.py --reflection-tta` (off by default) turns on mirror test-time
augmentation for prediction. When on, each member also runs a second forward
pass on the reflected view of the same decoded batch —
`reflect_batch` mirrors the prepared batch, forcing reflection
for every reflection-eligible study (the per-study `reflection_eligible` flag
already carried in every inference batch, computed per series at load time,
not sampled) and leaves reflection-ineligible studies unreflected, reusing
`apply_reflection`/`reflection_positions`/`reflection_channels` exactly as
training does. `predict_loader` (`src/knee_abnormality/inference/engine.py`)
averages, in probability space after the sigmoid, each study's probabilities
over whichever passes actually ran for it: one pass per member for an
ineligible study, two per member for an eligible one — never a fixed
division by 2, which would halve every ineligible study's prediction. The
flag is recorded in `manifest.json`'s `runtime.reflection_tta` and the
bundle's `config.json` `inference.reflection_tta` for both one-member and
ensemble bundles, so `predict.py` in bundle mode applies it automatically
with no extra CLI argument; `verify_bundle` (`inference/bundle.py`) rejects a
config/manifest mismatch. With the flag off, export and prediction stay
byte-identical to before, for both a single checkpoint and an ensemble.

Reflection TTA is a pure transform on the prepared batch, `reflect_batch`
(`src/knee_abnormality/inference/reflection.py`): for every reflection-eligible study
it mirrors non-sagittal images in x and reverses sagittal channel order, mirrors the
matching `valid_rects` (`left` becomes `width - columns - left`, sagittal rects
reversed by channel) and negates sagittal `slice_positions`. `predict_loader` feeds the
result to the unmodified model, which at eval draws no reflection of its own, so the
second pass sees a genuinely mirrored study and nothing in the model changes.

It is written this way because `modeling/model.py` is one of the files hashed into
every checkpoint's `preprocessing_identity` (`identity.PREPROCESSING_SOURCES`), and
that hash is byte-exact rather than semantic: adding even a no-op `force_reflection`
argument there invalidates every existing checkpoint and makes `export.py` refuse it.
`inference/` is not hashed, so `model.py` stays byte-identical and every checkpoint
remains valid.

Verified equal to forcing reflection inside the model: on two fold-0 studies the two
routes agree to **exactly 0.0**, while both differ from the unreflected pass by 0.087.
The transform mirrors rects directly where the in-model path mirrors the anchor grid
through `reflection_inverse`; the two conventions must stay in step, which that
equality check is what guards.

Measured on 2026-09-19, loading `b0-fold0` directly and running the same
averaging formula as `predict_loader` over fold 0's 904 validation studies at
`--image-multiplier 1`:

- Macro AUC 0.886348 with TTA off, reproducing the checkpoint's recorded
  0.88635 validation score.
- Macro AUC 0.886805 with TTA on (+0.00046).
- Reflection-eligible share for this cohort was 904/904 (100%), read from
  the batch flag, so no naturally ineligible study was available to prove
  the divide-by-1 branch on real data. That branch was instead proven with a
  synthetic batch (`reflection_eligible` forced to all-`False`): max
  `|combined - primary|` was exactly `0.0`. The divide-by-2 branch was
  confirmed the opposite way — all 904 real (eligible) studies' TTA-on
  predictions differed from their TTA-off predictions.

This confirms the averaging logic and the TTA-off no-op, but it is not a
substitute for running the actual bundle: `export.py`'s new flag,
`config.json`/`manifest.json` field placement, `bundle.py`'s
`verify_bundle` consistency check, and the `PredictionRequest`/
`predict_worker.py` plumbing that carries `reflection_tta` through bundle
mode are verified only by code review, not by an executed export or Kaggle
run.

Test-mode prediction reads the submission template and test metadata; it does
not call `load_fold_labels()`. Training CSVs, report-label sources, fold files,
training DICOMs and the original ImageNet download need not accompany the
inference bundle. Preserve their provenance with the trained artifact.
Because labels and folds are not bundled, reproducing validation through a bundle needs the
original data root and the checkpoint's label manifest, run from the bundle's own code:

```bash
rtk proxy uv run --locked env PYTHONPATH=outputs/submission/exports/BUNDLE/code \
  python -m scripts.pipeline.predict outputs/submission/exports/BUNDLE/config.json \
  --bundle outputs/submission/exports/BUNDLE --data-root data \
  --label-manifest configs/labels/combined-label-sources.json \
  --output outputs/analysis/BUNDLE-reproduced.csv --split validation --fold 0
```

Like the image multiplier, a bundle's attention temperature is frozen: a differing
`--attention-temperature` is rejected, so export a new bundle instead.

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

**Kaggle's notebook image runs Python 3.13 since 2026-10-03** (S98's first attempts failed on the
cp312 codec wheels, then on a missing `pydantic-settings`). The wheels that work are in
`outputs/submission/dependencies/kaggle-py313`: `pylibjpeg` 2.1.0, `pylibjpeg-libjpeg` 2.4.0 and
`pylibjpeg-openjpeg` 2.5.0 (cp313), `pydantic_settings` 2.15.0 and `python_dotenv` 1.2.3. They
are the export defaults since 2026-10-04 (`KAGGLE_PYTHON`, `KAGGLE_WHEELS` and the pinned
`OFFLINE_WHEELS` in `src/knee_abnormality/kaggle.py`), `prepare-codecs` downloads all five, and
export refuses a wheel directory that lacks one. S98 to S102 passed the folder and version as
flags. The local project stays on Python 3.12.

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
