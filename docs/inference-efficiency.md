# Inference and efficiency

Evidence: official submission research and repository inspection through
2026-09-11, completed local submissions checked on 2026-09-12, plus unverified working notes;
see [provenance](INDEX.md#provenance).

Each current submission uses one trained fold. The implemented exporter
creates an offline Kaggle notebook that loads a frozen model bundle, runs the
shared inference path on mounted test data and writes `submission.csv`.
Baseline3 submission `56164711` scored **0.878** publicly, and baseline4
submission `56167669` scored **0.910** publicly at 1× image limits. Baseline4
at 2× scored **0.913** (`56169438`), ahead of 3× at **0.912** (`56169445`).
Use **2× images and batch size 2** for current submissions.
The best recorded public result is now **0.925**, from `crop140mm`
submission `56182948` on 2026-09-12. See the
[crop comparison](validation.md#2026-09-12-140-mm-crop-result) and
[submission artifact](#140-mm-crop-submission).

## 140-mm crop submission

`crop140mm` epoch 8 (checkpoint index 7) completed scoring at **0.925** in
submission `56182948`, notebook version 1. The
[workflow record](../outputs/submissions/knee-crop140mm-20260912.json)
pins the dataset version, notebook version, submission response and API score.
The [immutable bundle](../outputs/exports/knee-crop140mm-20260912) uses
224-pixel inputs at 0.625 mm spacing, 2× images and batch size two.

Checkpoint:
[`best-epoch=07-val_macro_auc=0.88612.ckpt`](../outputs/crop140mm/fold_0/best-epoch=07-val_macro_auc=0.88612.ckpt),
SHA-256 `f3fd7b72bed1315297d54ee92742965ebb9c892ae0683c717bf20ad0947d3a92`.
Bundled-source SHA-256:
`babcebfccc5561c7f813acd8c1f68effaf8652e30d3b01428b8f5e2c9b6195ff`.
The manifest records that bundled source differs from the training snapshot;
export occurred during preprocessing package changes. Metadata, strict loading
and bundle integrity checks passed; optional local inference was not requested.
The completed score confirms this artifact's hidden-data execution. Workflow
elapsed times include scheduling and polling, and are not official inference
runtime. This is the preferred scored checkpoint; the overnight higher-LR
artifact remains the 150.08-mm comparison reference.

## Overnight fold-0 submissions

All three submissions from `outputs/overnight-20260911T234416Z-8du9a4nh`
completed successfully, confirmed through the Kaggle API on 2026-09-12.
They use notebook version 1, 2× images and batch size two. Epochs below are
numbered from one. Each record pins the uploaded bundle manifest and dataset
version, notebook version, submission ID, API score and workflow history.

| Run | Selected epoch | Submission ID | Workflow record |
|---|---:|---:|---|
| `02-longer` | 12 | 56181804 | [Record](../outputs/submissions/knee-overnight-02-longer-20260912.json) |
| `04-higher-lr` | 10 | 56181821 | [Record](../outputs/submissions/knee-overnight-04-higher-lr-20260912.json) |
| `05-faster-head` | 10 | 56181919 | [Record](../outputs/submissions/knee-overnight-05-faster-head-20260912-v2.json) |

The selected checkpoint for the overnight winner is
[`04-higher-lr/fold_0/best-epoch=09-val_macro_auc=0.88294.ckpt`](../outputs/overnight-20260911T234416Z-8du9a4nh/runs/04-higher-lr/fold_0/best-epoch=09-val_macro_auc=0.88294.ckpt),
SHA-256 `900add74a5473bf6c220b6287f90c96e46881a37776abbc53ec1d7d20a24e670`.
Its immutable bundle is
[`outputs/exports/knee-overnight-04-higher-lr-20260912`](../outputs/exports/knee-overnight-04-higher-lr-20260912).
The other bundles use the corresponding run names under `outputs/exports/`.
All exports passed metadata, strict model-loading and bundle integrity checks;
optional local inference was not requested. Successful scored submissions
establish hidden-data execution for these artifacts, but the API does not
provide official hidden-run runtime and no efficiency ranking is inferred.

## Verified submission requirements

Reviewed on 2026-09-10, with the indexed official rules, timeline, evaluation
and data pages rechecked on 2026-09-11. The code-requirements page still returned
no readable body, so recheck that interface before final submission selection,
especially the runtime hardware and execution limits.

- The [official code requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements)
  require notebook submissions, disabled Internet, a maximum of nine hours
  for either CPU or GPU execution, and an output named `submission.csv`.
- The [rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
  allow five submissions per day and two final selections. External data/models
  must meet the competition's accessibility and licensing conditions. Retain
  the public-label and pretrained-weight source/version records already used
  by training.
- The [timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline)
  currently sets entry/team-merger deadlines at October 15, 2026, final
  submission at October 22, and winners' deliverables at November 5, all at
  23:59 UTC. Winning solutions require code, weights and documentation, with
  additional presentation/public-release obligations described by the host.
- The [evaluation page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)
  scores the mean ROC-AUC across twelve targets. Write study-level confidence
  scores, preserving the official column names and order; do not threshold
  predictions into binary decisions.
- The [official efficiency evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation)
  considers final-selected (or automatically selected) submissions that beat
  the sample-submission benchmark on the private leaderboard. It minimizes:

  ```text
  Efficiency = AUC / (Benchmark - maxAUC) + RuntimeSeconds / 32400
  ```

  Here `maxAUC` is the best private-leaderboard AUC and `Benchmark` is the
  sample-submission score. This is not an AUC-per-second ratio. The public
  efficiency notebook updates daily and exposes ranks rather than full
  efficiency scores during development.

The tiny example-test run displayed on a public notebook page does not measure
hidden-test runtime. Measure the full path: imports, input discovery, decoding,
preprocessing, model loading, forward passes, ensembling, and CSV writing.
Training caches can save repeated experiments but do not predecode unseen
test studies. See [public runtime evidence](public-research.md#data-access-and-runtime-lessons).

## Inputs and output contract

The [official data description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
states that scoring replaces the three example studies, series descriptors
and DICOMs with the actual test data, approximately 1,300 studies. Reports
are absent at test time. Discover study IDs from the mounted test files on
every execution; do not package example predictions or hard-code their IDs.
The two contrast flags are not guaranteed to coincide. The listed image
encodings include uncompressed little-endian, JPEG Lossless and JPEG 2000.

The local official template was also inspected on 2026-09-10. Its header is:

```text
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
```

The output should have exactly one row for every template study, twelve finite
probabilities in `[0, 1]`, no missing/duplicate IDs and no extra index or fold
column. Use the mounted template to order rows and columns. Write the final
file to `/kaggle/working/submission.csv`.

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

Code inspected: [prediction CLI](../scripts/pipeline/predict.py),
[inference](../src/knee_abnormality/inference.py),
[dataset/index](../src/knee_abnormality/dataset.py),
[model](../src/knee_abnormality/model.py), and
[configuration](../src/knee_abnormality/config.py).

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
  `configs/baseline.yaml`. Freeze model/data settings; permit explicit path and
  runtime overrides with their values recorded.
- A versioned copy or wheel of the shared Python package and prediction entry
  point. Include the source needed by imports; reuse the same preparation code.
- Offline wheels/requirements for dependencies missing from the selected
  Kaggle image, plus file hashes, dependency versions, source identity,
  checkpoint hash, label/split provenance and validation results.

Test-mode prediction reads the submission template and test metadata; it does
not call `load_fold_labels()`. Training CSVs, report-label sources, fold files,
training DICOMs and the original ImageNet download need not accompany the
inference bundle. Preserve their provenance with the trained artifact.

The exporter checks checkpoint metadata, reconstructs the model strictly on
CPU, and hashes packaged files. Readiness requires target Python/platform values
and all three decoder wheels. Local inference verification is **optional** and
does not block routine submissions. Without it, `verification` is null and
`checks.local_inference` is `not_requested`; this does not claim numerical parity
or fresh codec execution. Training-source correspondence remains recorded.

For an explicitly requested investigation, `--verification-record` can attach
previously measured evidence with this schema:

```json
{
  "checkpoint_sha256": "64 lowercase hex characters",
  "bundled_source_sha256": "64 lowercase hex characters",
  "validation_max_abs_difference": 0.0,
  "validation_macro_auc": 0.0,
  "inference_image_multiplier": 2,
  "example_submission_rows": 3,
  "example_submission_valid": true,
  "jpeg_lossless_decoded": true,
  "jpeg_2000_decoded": true
}
```

The exporter validates the checkpoint/source hashes, image multiplier and
successful booleans before embedding supplied evidence. Enter actual measured
values. Sample evidence sets `validation_scope` to `sample`, records
`validation_studies`, and leaves `validation_macro_auc` null. `submit.py` still
checks bundle integrity and refuses unmet packaging prerequisites.

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
[environment procedure](environment.md#confirm-the-selected-kaggle-notebook)
records the relevant package inventory. Resolve/install from attached local
wheels with package-index access disabled; avoid blind upgrades of the whole
training environment inside a scored notebook.

### Baseline3 fold-0 submission artifact

Local preparation on 2026-09-11 selected
`outputs/baseline3/fold_0/best-epoch=07-val_macro_auc=0.84982.ckpt`, SHA-256
`b955aefe710905e47abf8e5ec8f65d2c6ec009108e61eeae02cb448d31bcaf54`.
The exported state dictionary has the same 341 keys and tensor values as the
training checkpoint, with the same fold and ordered targets. Fresh BF16
inference reproduced all 903 saved validation rows exactly, with maximum
absolute probability difference `0.0` and macro-AUC
`0.8498192406240211`. Prediction took 37.42 seconds on the local RTX 3090,
or 24.13 studies per second, excluding the 0.76-second model load.

The three-study example prediction matches the official 13-column schema,
study order and probability range. Repository, frozen-bundle and generated
notebook execution produced identical probabilities. The notebook installed
its wheels with index access disabled, reported 0.69 seconds for setup and
5.43 seconds for the prediction subprocess, and validated its output. These are
local notebook measurements.

Kaggle notebook version 3 installed the same wheels offline and completed on a
CUDA runtime using FP32. Its three-study prediction took 11.54 seconds and the
whole visible notebook took 82.12 seconds. This validates the visible Kaggle
path, not hidden-test throughput.

The Python 3.12 manylinux2014 x86-64 bundle contains `pylibjpeg==2.1.0`,
`pylibjpeg-libjpeg==2.4.0` and `pylibjpeg-openjpeg==2.5.0`. Those exact wheels
decoded pydicom's real JPEG Lossless fixture with transfer syntax
`1.2.840.10008.1.2.4.70` and JPEG 2000 Lossless fixture with transfer syntax
`1.2.840.10008.1.2.4.90`. A complete header scan found all 819,078 local
training DICOMs use uncompressed Explicit VR Little Endian, so no local
competition file could exercise the compressed paths. Hidden-test codec mix and
full runtime remain remote checks.

The submitted bundle is `outputs/exports/baseline3-fold0-v3`, with bundled-source
SHA-256 `f4976ae171bc3c363e1aba55356a6d9bd1689c37d6ad6ed7d2e184e298e63578`
and no unmet manifest prerequisites. The remote workflow started on 2026-09-11:
private Dataset version 2 is ready and offline notebook version 3 completed.
Its validated output was submitted as `56164711` and scored **0.878** publicly;
the state record is
`outputs/submissions/baseline3-fold0-20260911-v6.json`.

After that submission launched, the local project migrated from the absent
`lightning` distribution to Kaggle's installed
`pytorch-lightning==2.6.5`. Full validation inference again matched all 903
saved rows exactly. The ready `outputs/exports/baseline3-fold0-v5` bundle has
bundled-source SHA-256
`4a270850d0594612ec913504f38dd5e988f349ea6fa7e67fc70cc4c1922e54f0`,
records `pytorch-lightning==2.6.5`, and contains only the three offline DICOM
codec wheels. Its generated notebook completed locally without the
`lightning` distribution.

Training and prediction both select PyTorch's `high` float32 matrix
multiplication policy. This keeps the float32 target-attention calculation
consistent while BF16 autocast remains active for CUDA inference.

Baseline4's ReXNet-200 epoch-7 checkpoint reproduced all 903 saved validation
probabilities exactly at **0.872510 macro AUC**. The ready wheel-free bundle is
`outputs/exports/baseline4-fold0-v1`, with bundled-source SHA-256
`6cedf5a92eed77fe40a6064eb55357d5266cfdf84f364c45a488b5865a7e258b`.
Kaggle notebook version 1 completed its visible three-study run in 54.14
seconds and submission `56167669` scored **0.910** publicly. Kaggle scoring was
observed complete 23 minutes 41 seconds after submission; that interval
includes queue and evaluation time and is accurate only to the runner's
30-second polling interval. The state record is
`outputs/submissions/baseline4-fold0-20260911.json`.

## Fast submission preparation

Routine submission preparation runs **no local prediction**: no full-fold
parity, sample parity, example inference, or VRAM profiling. Export the selected
checkpoint directly, reusing the prepared codec wheel directory:

```bash
rtk proxy uv run --locked python -m scripts.pipeline.export \
  --checkpoint outputs/baseline4/fold_0/best-epoch=07-val_macro_auc=0.87251.ckpt \
  --output outputs/exports/baseline4-fold0-2x-v2 \
  --wheel-dir outputs/dependencies/kaggle-v170 \
  --target-python 3.12 --target-platform manylinux_x86_64 \
  --image-multiplier 2 --inference-batch-size 2
```

Use a fresh output path for each immutable bundle. Set `--image-multiplier 3`
for 3×. If omitted, the multiplier is **2**, independently of the checkpoint's
saved validation configuration. The selected density is stored separately from training settings
in `config.json` and the manifest. Bundle prediction rejects a different density
override; export a new bundle for that comparison. The notebook uses the
exported batch size, which defaults to 2, while precision remains native CUDA
BF16 or FP32 fallback. The wheel directory contains only the three codecs;
Kaggle provides `pytorch-lightning`, so no Lightning wheel is uploaded.

`manifest.json` records `preparation_seconds`, excluding Python process startup.
The actual baseline4 2× and 3× v2 exports took **0.64 s** and **0.56 s**
respectively, with `local_inference: not_requested` in both manifests.
The submit runner records timestamped transitions in `history` and prints them
to its log. These distinguish local preparation, upload, notebook execution and
scoring waits; the last intervals include network latency and polling delay.

The optional utility lives at
[`scripts/tools/verify_inference.py`](../scripts/tools/verify_inference.py).
Only explicit `--quick-verify --codec-reference-bundle PATH` invokes it through
the exporter. It compares about 16 held-out studies using complete original
validation batches and the original validation density, then performs a bounded
FP32 memory check at the requested density and predicts the three example
studies. Codec evidence is reused only when wheel hashes match the supplied
verified bundle. This is an on-demand diagnostic after relevant inference
changes, not a step to repeat for each submission. It does not calculate a
partial-fold AUC or require a full training run.

Launch `scripts.pipeline.submit` in a detached user systemd service so uploads,
Kaggle execution and automatic submission continue after the interactive session
returns. Give concurrent experiments distinct bundle, dataset, notebook and
record paths. This account rejected a third concurrent notebook on 2026-09-12
with `Maximum batch GPU session count of 2 reached.` Keep at most two batch
GPU notebook runs active, and start additional jobs after a notebook completes.
The current runner records this rejection as failed; it does not queue a retry.
Reuse the same immutable bundle, dataset and notebook with a fresh record path
after capacity becomes available. The rejection occurred before competition
submission and consumed no submission slot; evidence is
`outputs/submissions/knee-overnight-05-faster-head-20260912.json`.
Do not repeatedly wait for remote completion in the interactive
session; monitor the record or runner log as needed.

### Justfile recipes

The root [justfile](../justfile) covers the maintained path. Run recipes as
`rtk proxy uv run --locked just RECIPE ...` from the repository; the table uses
the shorter `just` spelling. Arguments are positional. Quote paths and messages
containing spaces. After training, routine submissions need only export and
submit; `validate` and `infer` are optional.

| Operation | Recipe |
|---|---|
| Install the locked environment | `just setup` |
| Download the three pinned codec wheels once | `just prepare-codecs` |
| Prepare/recheck fixed split membership | `just prepare-folds` |
| Train fold 0 | `just train RUN_ID [CONFIG]` |
| Resume into a new run directory | `just resume NEW_RUN_ID CHECKPOINT SAVED_CONFIG` |
| Explicit full-fold prediction | `just validate SAVED_CONFIG CHECKPOINT OUTPUT_CSV` |
| Export with 2× images and batch 2 | `just export CHECKPOINT BUNDLE` |
| Export at 2× or 3× | `just export CHECKPOINT BUNDLE MULTIPLIER [BATCH_SIZE]` |
| Optional local test prediction | `just infer BUNDLE [DATA_ROOT] [OUTPUT_CSV]` |
| Launch automatic submission and return | `just submit BUNDLE DATASET_ID NOTEBOOK_ID RECORD_JSON [MESSAGE]` |
| Inspect saved progress and stage durations | `just submit-status RECORD_JSON` |
| Read recent runner output and return | `just submit-logs RECORD_JSON [LINES]` |
| Explicitly follow live runner output | `just submit-follow RECORD_JSON [LINES]` |
| Query current Kaggle submission statuses/scores | `just submissions` |
| Submit in the foreground for debugging | `just submit-foreground BUNDLE DATASET_ID NOTEBOOK_ID RECORD_JSON [MESSAGE]` |

For example, after selecting an actual checkpoint:

```bash
rtk proxy uv run --locked just export \
  outputs/baseline4/fold_0/best-epoch=07-val_macro_auc=0.87251.ckpt \
  outputs/exports/baseline4-next-2x
rtk proxy uv run --locked just submit \
  outputs/exports/baseline4-next-2x \
  claverru/knee-baseline4-next-2x claverru/knee-baseline4-next-2x-inference \
  outputs/submissions/baseline4-next-2x.json "Baseline4, 2x images"
rtk proxy uv run --locked just submit-status outputs/submissions/baseline4-next-2x.json
```

Export uses `outputs/dependencies/kaggle-v170`, Python 3.12 and Linux x86-64 by
default; later positional arguments override those values. `prepare-codecs`
is a separate one-time download for that runtime and includes no Lightning
wheel. It is never invoked automatically by training or submission. Preparing
five-fold membership does not train five models; development continues on fold
0. Resume requires a new run ID and saved config because training restores
the checkpoint into a fresh directory, preserving existing run artifacts.

`infer` honors the bundle's batch size unless explicitly overridden, resolves
paths before changing directory, and runs from the bundle so both the prediction
entry point and package come from frozen source. Validation prediction is never
a dependency of export or submission.

The [submission job utility](../scripts/tools/submission_job.py) performs cheap
bundle integrity checks and starts the existing submission pipeline with a
user systemd service. Launch requires Linux user systemd, derives a unique unit
from the absolute record path, and rejects existing records/logs. The matching
`RECORD_STEM-runner.log` survives service exit. Status and log inspection also
work for earlier manually launched runners. Saved status reflects the runner's
latest update; use `just submissions` for a fresh Kaggle query. Only
`submit-follow` and `submit-foreground` intentionally wait.

Recipe syntax, lint, catalog consistency, and actual read-only status/log
commands were checked on 2026-09-11. The new launcher was not used to create
an extra competition submission during this change. No training, inference
checks, or tests were run.

## Inference image count and memory

Select **2× image limits with batch size 2**, following the completed public
scores on 2026-09-11. New exports and repository test prediction default to 2×;
validation retains its saved validation multiplier. Each bundle freezes its
exported density. Baseline4's saved validation already used 2×.

| Image multiplier | Inference batch size | Public score | Submission | Submission to observed completion |
|---|---:|---:|---|---|
| 1× | 8 | 0.910 | `56167669` | 23m 41s |
| **2×** | **2** | **0.913** | `56169438` | 34m 17s |
| 3× | 2 | 0.912 | `56169445` | 28m 12s |

All three use baseline4's selected fold-0 ReXNet-200 epoch-7 checkpoint,
`outputs/baseline4/fold_0/best-epoch=07-val_macro_auc=0.87251.ckpt`, with
SHA-256 `002778da78be2bb535618f8cb307ee631b94ca561a47e5f691964d879f0d276d`.
The 2× and 3× submissions share batch size 2; the earlier 1× submission used
batch size 8. These public scores favor 2× for current single-fold iteration.
The intervals include queue and evaluation time plus up to one 30-second
polling interval; they do not isolate model inference or establish that 3× is
computationally faster.

On 2026-09-11, a bounded FP32 forward profile on the local RTX 3090 used two
studies selected for high available slice counts at 3×. No training was run.
Artifact: `outputs/profiling/baseline4-inference-vram.json`, tied to baseline4's
selected epoch-7 checkpoint and its saved config. Values are GiB:

| Image multiplier | Batch 2 allocated peak | Batch 2 reserved peak | Rough batch 8 allocated peak |
|---|---:|---:|---:|
| 1× | 1.56 | 1.91 | 6.07 |
| 2× | 3.07 | 3.75 | 12.13 |
| 3× | 4.24 | 5.18 | 16.80 |

Batch-8 values extrapolate batch-dependent memory linearly after subtracting
model parameters; they are estimates, not measured batch-8 runs. Batch 8 is too
risky at 3× on a 16 GB T4. Keep **batch 2 for inference**. Different GPU
workspaces and hidden-study composition can change the peaks, so these are
planning measurements rather than a worst-case guarantee.

Both comparisons completed on 2026-09-11 after running as detached user services:

| Density | Bundle under `outputs/exports/` | User service | Workflow record under `outputs/submissions/` |
|---|---|---|---|
| 2× | `baseline4-fold0-2x-v2` | `knee-baseline4-2x-submit-v2` | `baseline4-fold0-2x-20260911-v2.json` |
| 3× | `baseline4-fold0-3x-v2` | `knee-baseline4-3x-submit-v2` | `baseline4-fold0-3x-20260911-v2.json` |

The records and matching `-runner.log` files preserve completed stages and scores. Notebook IDs
are `claverru/knee-baseline4-2x-inference` and
`claverru/knee-baseline4-3x-inference`; each runner automatically submitted its
successful saved notebook version and recorded the score. The first 2× v1
workflow was stopped after dataset upload, before any notebook or competition
submission, to correct config-envelope loading. Its record is marked superseded.

## Control the workflow from the local CLI

The normal submission cycle can be orchestrated locally: prepare the bundle,
upload it, launch an offline Kaggle notebook, fetch its outputs, submit a saved
version and poll the score. Notebook execution and hidden-test scoring still
run on Kaggle. This does not provide local access to the hidden test set.

Verified on 2026-09-10 against the installed Kaggle CLI/API 2.0.2, its public
SDK and the official CLI documentation. A read-only
`competitions list --group entered -s rsna-knee-abnormality-detection -v`
request succeeded and returned `userHasEntered=True`: authentication and
competition entry were already in place. That research step performed no
uploads or submissions; the later baseline3 workflow uses the resulting
commands. The client reported 2.2.2 available; the training environment remains
on the pinned 2.0.2 interface.

The commands below follow the installed 2.0.2 interface. Prefix each with
`rtk proxy uv run --locked` when invoking it from this repository.

| Step | Local command or configuration |
|---|---|
| Prepare weights, config, code, wheels and notebook | `python -m scripts.pipeline.export` or `just export`; inspect `manifest.json` and resolve every unmet prerequisite |
| Upload a private bundle | `kaggle datasets create -p BUNDLE_DIR --keep-tabular --dir-mode zip` |
| Upload a revised bundle | `kaggle datasets version -p BUNDLE_DIR -m "Bundle revision" --keep-tabular --dir-mode zip` |
| Wait for bundle processing | `kaggle datasets status OWNER/BUNDLE_SLUG` |
| Attach data and request an offline GPU run | Set notebook metadata, then `kaggle kernels push -p NOTEBOOK_DIR -t 32400 --accelerator NvidiaTeslaT4` |
| Monitor the saved notebook run | `kaggle kernels status OWNER/NOTEBOOK_SLUG` and `kaggle kernels logs OWNER/NOTEBOOK_SLUG --follow` |
| Download ordinary run outputs/logs | `kaggle kernels output OWNER/NOTEBOOK_SLUG -p OUTPUT_DIR` |
| Submit a particular notebook version | `kaggle competitions submit COMPETITION -k OWNER/NOTEBOOK_SLUG -v VERSION -f submission.csv -m "Fold 0"` |
| Poll scoring status and available scores | `kaggle competitions submissions rsna-knee-abnormality-detection -v` |

See the official [dataset commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md),
[notebook commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
and [competition commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md).

Create `dataset-metadata.json` beside the upload files, using `datasets init`
if helpful. Set the owner/slug, title and appropriate license metadata; retain
the bundled components' licenses and provenance. Dataset creation defaults to
private. Directory handling defaults to `skip`, so use the explicit archive
option above for nested source/wheel folders and account for their archive
layout in the notebook. See the
[dataset metadata specification](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md).
For reproducibility, prefer an immutable bundle ID for each selected artifact
and record its version and file hashes. Do not replace the bundle while its
notebook run is being prepared or scored.

An illustrative `NOTEBOOK_DIR/kernel-metadata.json` is:

```json
{
  "id": "YOUR_USERNAME/knee-inference",
  "title": "Knee Inference",
  "code_file": "inference.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "competition_sources": ["rsna-knee-abnormality-detection"],
  "dataset_sources": ["YOUR_USERNAME/knee-fold0-bundle"]
}
```

These are example paths and IDs. The baseline3 artifact section records the
actual bundle, Dataset and notebook versions used later.
The [notebook metadata specification](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
supports data attachments, privacy, Internet access and accelerator selection.
Set `enable_internet` explicitly: installed 2.0.2's implementation defaults it
to `True` when omitted, despite the current documentation saying otherwise.
The accelerator request remains subject to account access, quota and service
availability; compatibility must be checked on the actual allocated runtime.

`kernels push` starts a saved-version run and returns its version number.
Wait for successful completion and verify `submission.csv` before submitting
that exact version. For example, if the returned version is 3:

```bash
rtk proxy uv run --locked kaggle competitions submit rsna-knee-abnormality-detection \
  -k YOUR_USERNAME/knee-inference -v 3 -f submission.csv -m "Fold 0"
```

Here `-f` identifies the notebook's output; `-k` and `-v` select the code Kaggle
will rerun. It is not an upload of a local CSV. This code-submission interface
is documented in the official [CLI tutorial](https://github.com/Kaggle/kaggle-cli/blob/main/docs/tutorials.md).
Record the returned submission ID and poll it in the submission list. Newer
CLI documentation also describes `submit --wait` and singular `competitions
submission`; those options are absent from installed 2.0.2 and are unnecessary
for this workflow.

Remaining website steps and monitoring limits:

- Joining and accepting competition rules require the website according to
  the CLI tutorial; this account has already entered. Any requested account
  verification also needs to be completed through Kaggle's account flow.
- No final-submission selection or team-management operation was found in
  the inspected supported CLI/public competition API. Plan to use the website
  to choose the two final submissions and manage any team changes.
- Notebook status/log/output calls address the latest ordinary session by
  owner/slug. Avoid overlapping pushes to the same notebook when collecting
  evidence. They do not establish access to a hidden scoring rerun's full logs.
  The submission API exposes status, scores when available and an error
  description, but its inspected submission schema has no runtime field.
  Detailed hidden-run debugging and official runtime reporting may still need
  the website. Locally recorded notebook timings measure only its visible run.

These limits came from API and client inspection. The active baseline3 attempt
exercises the supported workflow but does not replace the final successful
scored-run check.

## Notebook assembly and submission workflow

1. Export the selected checkpoint with the fast preparation command above.
   Local prediction checks are optional, requested separately when investigating
   preprocessing, target order, precision or model-loading changes.
2. Attach the competition data and the frozen model/code/dependency bundle to
   a Kaggle notebook. Locate the competition root by the required CSVs and
   `test_series/`, considering both direct and nested competition mount layouts.
3. Preserve the existing CLI's layout with a working-directory adapter if
   desired: `<working-data>/raw` can link to the read-only competition root.
   Pass `<working-data>` as `--data-root`; no DICOM copying is required. Keep
   bundle paths separate from competition paths and outputs writable.
4. Load the offline dependencies/code and run the shared inference command
   with the chosen precision/runtime settings. Infer study membership from
   the mounted inputs, validate the output contract and write the named CSV.
5. Reuse recorded memory/runtime measurements for routine submissions. Profile
   separately when a new model or sampling budget needs an estimate. The
   three-example run and local training throughput do not establish full
   hidden-test runtime.
6. Disable Internet, then push and run a version with the
   [local CLI workflow](#control-the-workflow-from-the-local-cli). After a
   successful run, submit its `submission.csv` with the exact notebook version.
   The website's **Save & Run All** and Output submission controls are an
   alternative. Kaggle reruns the code against the hidden inputs; directly
   uploading a locally generated CSV is not the code-competition workflow.
   [Kaggle submission instructions](https://www.kaggle.com/docs/competitions).
7. Record notebook version, bundle version, submission ID, processing status,
   score and reported runtime. Inspect failed-run logs before retrying and
   keep final selections explicit before the deadline.

Bundle readiness records successful packaging checks. The remote notebook
must still finish and produce valid output before automatic submission. A
successful scored submission confirms the hidden-data execution path for that
artifact. Baseline3 and baseline4's original submissions both completed scoring.

A [submission troubleshooting report](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735854)
identifies P100/PyTorch build incompatibility, missing DICOM codecs and nested
`/kaggle/input/competitions/` or `/kaggle/input/datasets/` mount layouts. Treat
these as environment-specific checks: test an actual GPU operation and
representative compressed DICOM decoding, make dependencies available offline,
and use bounded input discovery. A CUDA availability flag or a successful
uncompressed sample is insufficient. See [environment](environment.md) for the
repository's actual pinned versions.

## Unverified submission results

These are distinct configurations and dates, not a controlled scaling study.
The recorded public leaderboard scores and timings have not been independently
verified.

| Date | Setup | Score | Reported inference/submission time |
|---|---|---|---|
| 2026-08-29 | Two-model, fold-0 ensemble | Not recorded | About 4 hours; one 300M-parameter model took about 3 hours |
| 2026-09-01 | Optimized 15-model ensemble | Not stated with this timing | About 3 hours |
| 2026-09-02 | Five small-model runs | 0.938 | About 40 minutes |
| 2026-09-03 | Small-model five-fold ensemble | 0.944 | About 35 minutes |
| 2026-09-03 | Small single model | 0.938 | About 20 minutes |
| 2026-09-08 | Best single-fold model | 0.944 | About 1 hour |

Architecture identities, checkpoint versions, hardware, and inclusion of
startup/data loading are incomplete. Do not equate the unnamed models across
dates or infer linear ensemble cost from these figures.

## Runtime leads

- Separate loading/decoding cost from model execution. A loading-only notebook
  was suggested to estimate the data-access floor, but no result was reported.
- Metadata-based series/slice selection and lower-level pixel loading were
  suggested as ways to avoid decoding redundant data. Skipping over 60% of
  slices was claimed to be possible, without a supplied selection policy or
  controlled score comparison.
- Reduced resolution and window counts were discussed as speed/score tradeoffs;
  see [data preparation](data-preparation.md#resolution-tradeoffs) and
  [window sampling](modeling.md#window-sampling) for the actual reported settings.
- Ensemble distillation into a small model, fixed-shape compilation, parallel
  model execution, and specialized kernels were proposed. Large projected
  speedups and TPU advantages were not measured.
- Some submissions were recorded as failing after additional checks were
  introduced. Their conditions are unknown, so this does not justify
  removing correctness checks indiscriminately.

## Training throughput observations

Targeted benchmarking reportedly raised throughput on the large model from
1.8 to 3.4 studies/second. A small model improved by about 25% in a subsequent
report. An unspecified fusion change reduced GPU memory use from 19 GB to
12 GB; the exact model and changed operations were not identified.

These observations support investigating iteration cost, but they are not
reproducible optimization recipes. There is no before/after code,
batch sizes, numerical-equivalence checks, or stable hardware conditions.
Reported training durations and backbone choices are in
[modeling](modeling.md#unverified-model-structure).

## Remaining efficiency questions and historical context

The [efficiency leaderboard notebook](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)
combines score and execution time.

Earlier notes did not establish the rules or whether efficiency eligibility
depends on selected submissions. A fast model and ensemble were considered as
a way to test that hypothesis, without a definitive result.
The [official requirements above](#verified-submission-requirements) supersede
that uncertainty for final prize eligibility.

TPU eligibility, hardware normalization, and medal eligibility were not resolved
by the sources reviewed. An unverified claim that another entrant achieved
0.948 in about five minutes lacks a linked run. Verify
these details on the official competition pages before choosing a strategy.
