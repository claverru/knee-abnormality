# Justfile recipes and submission preparation

The root [justfile](../../justfile) covers the maintained path. Run recipes as
`uv run --locked just RECIPE ...` from the repository; the table uses
the shorter `just` spelling. Each recipe is named after the module it runs and passes
its arguments through unchanged, so `just RECIPE --help` lists every option. After
training, routine submissions need only export and submit; `predict` and `infer` are
optional.

## Recipe reference

| Operation | Recipe |
|---|---|
| Install the locked environment | `just setup` |
| Download the three pinned codec wheels once | `just prepare-codecs` |
| Print an effective configuration, with overrides | `just config CONFIG [--FIELD VALUE ...]` |
| Prepare/recheck fixed split membership | `just validation prepare` |
| Score complete out-of-fold predictions | `just validation score --predictions OOF_CSV` |
| Train the YAML's folds | `just train CONFIG --run-id RUN_ID [--FIELD VALUE ...]` |
| Resume into a new run directory | `just train SAVED_CONFIG --run-id NEW_RUN_ID --resume CHECKPOINT` |
| Explicit full-fold prediction | `just predict SAVED_CONFIG --checkpoint CHECKPOINT --output OUTPUT_CSV --split validation` |
| Export with the default 2× images, four studies per GPU and eight loader workers | `just export --checkpoint CHECKPOINT --output BUNDLE` |
| Export at another multiplier or batch size | `just export --checkpoint CHECKPOINT --output BUNDLE --image-multiplier 3 --inference-batch-size 2` |
| Optional local test prediction | `just infer BUNDLE [DATA_ROOT] [OUTPUT_CSV] [--FLAG VALUE ...]` |
| Launch automatic submission and return | `just submission-job start --bundle BUNDLE --dataset DATASET_ID --notebook NOTEBOOK_ID --record RECORD_JSON --label LABEL` |
| Inspect saved progress and stage durations | `just submission-job status --record RECORD_JSON` |
| Read recent runner output and return | `just submission-job logs --record RECORD_JSON [--lines N]` |
| Explicitly follow live runner output | `just submission-job logs --record RECORD_JSON --follow` |
| Query current Kaggle submission statuses/scores | `just submissions` |
| Submit in the foreground for debugging | `just submit --bundle BUNDLE --dataset DATASET_ID --notebook NOTEBOOK_ID --record RECORD_JSON --label LABEL` |
| Run or inspect an autoresearch campaign | `just autoresearch SUBCOMMAND CAMPAIGN`, `just autoresearch-watch SUBCOMMAND CAMPAIGN` |

`infer` fills its positionals first, so extra flags need `DATA_ROOT` and `OUTPUT_CSV`
written out, e.g. `just infer BUNDLE data out.csv --device cpu`.

## Export and submit a checkpoint

Routine submission preparation runs **no local prediction**: no full-fold parity, sample parity,
example inference or VRAM profiling. Export the selected checkpoint directly, reusing the prepared
codec wheel directory, then launch the detached job:

```bash
uv run --locked just export \
  --checkpoint outputs/training/overnight-20260914T220805Z-ozsz4qbu/runs/00-baseline/fold_0/best-epoch=09-val_macro_auc=0.89276.ckpt \
  --output outputs/submission/exports/00-baseline-next-2x
uv run --locked just submission-job start \
  --bundle outputs/submission/exports/00-baseline-next-2x \
  --dataset claverru/knee-00-baseline-next-2x --notebook claverru/knee-00-baseline-next-2x-inference \
  --record outputs/submission/records/00-baseline-next-2x.json --label "00-baseline"
uv run --locked just submission-job status --record outputs/submission/records/00-baseline-next-2x.json
```

Use a fresh output path for each immutable bundle. `--image-multiplier` defaults to **2**
(`TEST_IMAGE_MULTIPLIER`) independently of the checkpoint's saved validation configuration; later
submissions exported ×3 to ×8 (see the [ledger](submissions/ledger.md)). The density is stored
separately from training settings in `config.json` and the manifest, and bundle prediction rejects
a different density override: export a new bundle for that comparison. The notebook uses the exported
per-GPU batch size and total loader workers, which default to 4 and 8 (`INFERENCE_BATCH_SIZE`,
`INFERENCE_NUM_WORKERS` in `config.py`; the 2026-09-15 to 2026-09-17 submissions used 16 workers)
with [one process per visible GPU](independent-gpu-inference-processes.md). Precision `auto`
resolves to native CUDA BF16, otherwise CUDA FP16 with fixed 64-image channels-last backbone
chunks (T4), and FP32 on CPU. The wheel directory contains the three codecs, `pydantic-settings`
and `python-dotenv`; Kaggle provides `pytorch-lightning`, so no Lightning wheel is uploaded.

`manifest.json` records `preparation_seconds`, excluding Python process startup (0.64 s and 0.56 s
for the baseline4 2× and 3× exports). Export runs no local inference and records no verification;
the manifest's `checks` cover checkpoint metadata and the strict model load. The submit runner
records timestamped transitions in `history` and prints them to its log; they distinguish local
preparation, upload, notebook execution and scoring waits, and the last intervals include network
latency and polling delay.

### Optional verification diagnostic

[`scripts/tools/verify_inference.py`](../../scripts/tools/verify_inference.py) is not part of export
or bundled with it. Run it from the repository root with `PYTHONPATH=<bundle>/code`, so it imports
the exported package, and a request JSON with `config` (the saved effective configuration),
`reference_csv` (the selected epoch's saved validation predictions), `samples`, `image_multiplier`,
`batch_size`, `checkpoint_sha256`, `bundled_source_sha256` and `codec_evidence`. It compares about 16
held-out studies using complete original validation batches and the original validation density under
native BF16, then performs a bounded FP32 memory check at the requested density and predicts the three
example studies. It writes `verification.json`, `verification_sample.csv` and
`verification_example.csv` into the bundle directory, which then no longer matches its manifest
inventory; run it on a disposable copy:

```bash
uv run --locked env PYTHONPATH=outputs/submission/verification/BUNDLE-copy/code \
  python -m scripts.tools.verify_inference --bundle outputs/submission/verification/BUNDLE-copy \
  --request-file outputs/submission/verification/BUNDLE-request.json
```

The sampled batches span low and high input counts; an ensemble bundle is checked the same way,
comparing the members' averaged prediction with the reference CSV. It is an on-demand diagnostic after
relevant inference changes, not a step per submission, and it does not calculate a partial-fold AUC.

### Concurrency

`scripts.pipeline.submit` runs in a detached user systemd service so uploads, Kaggle execution and
automatic submission continue after the interactive session returns. Give concurrent experiments
distinct bundle, dataset, notebook and record paths. The account rejected a third concurrent notebook
on 2026-09-12 with `Maximum batch GPU session count of 2 reached.`, so at most two batch GPU notebook
runs can be active. The runner records this rejection as failed and does not queue a retry; rerun the
same immutable bundle, dataset and notebook with a fresh record path after capacity frees up. The
rejection occurred before competition submission and consumed no submission slot (evidence:
`outputs/submission/records/knee-overnight-05-faster-head-20260912.json`). Monitor the record or runner
log instead of waiting on remote completion in the interactive session.

## Runtime and resume behavior

Export uses `outputs/submission/dependencies/kaggle-py313`, Python 3.13 and Linux x86-64 by
default (`knee_abnormality.kaggle`); `--wheel-dir`, `--target-python` and `--target-platform`
override them. `prepare-codecs` is a separate one-time download of the pinned offline wheels (the codecs,
`pydantic-settings` and `python-dotenv`) for that runtime and includes no Lightning wheel. It is never invoked automatically by training or
submission. Preparing five-fold membership does not train five models; development continues on
fold 0. Resume requires a new run ID and saved config because training restores the checkpoint
into a fresh directory, preserving existing run artifacts.

`infer` honors the bundle's batch size unless explicitly overridden, resolves
paths before changing directory, and runs from the bundle so both the prediction
entry point and package come from frozen source. Validation prediction is never
a dependency of export or submission.

## Submission labels

`LABEL` is the only submission text chosen by hand: 2 to 45 characters naming the run and its one
change. Everything Kaggle shows is built from it and the bundle, in the
[unified submission format](submissions/ledger.md#naming), and a submission description cannot be
edited afterwards.

## Detached submission status and logs

The [submission job utility](../../scripts/tools/submission_job.py) performs cheap
bundle integrity checks and starts the existing submission pipeline with a
user systemd service. Launch requires Linux user systemd, derives a unique unit
from the absolute record path, and rejects existing records/logs. The matching
`RECORD_STEM-runner.log` survives service exit. Status and log inspection also
work for earlier manually launched runners. Saved status reflects the runner's
latest update; use `just submissions` for a fresh Kaggle query. Only
`submission-job logs --follow` and `submit` intentionally wait.
