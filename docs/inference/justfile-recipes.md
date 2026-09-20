# Justfile recipes

The root [justfile](../../justfile) covers the maintained path. Run recipes as
`rtk proxy uv run --locked just RECIPE ...` from the repository; the table uses
the shorter `just` spelling. Arguments are positional. Quote paths and labels
containing spaces. After training, routine submissions need only export and
submit; `validate` and `infer` are optional.

## Recipe reference

| Operation | Recipe |
|---|---|
| Install the locked environment | `just setup` |
| Download the three pinned codec wheels once | `just prepare-codecs` |
| Prepare/recheck fixed split membership | `just prepare-folds` |
| Train fold 0 | `just train RUN_ID [CONFIG]` |
| Resume into a new run directory | `just resume NEW_RUN_ID CHECKPOINT SAVED_CONFIG` |
| Explicit full-fold prediction | `just validate SAVED_CONFIG CHECKPOINT OUTPUT_CSV` |
| Export with the default 2× images, four studies per GPU and sixteen loader workers | `just export CHECKPOINT BUNDLE` |
| Export at 2× or 3× | `just export CHECKPOINT BUNDLE MULTIPLIER [BATCH_SIZE]` |
| Optional local test prediction | `just infer BUNDLE [DATA_ROOT] [OUTPUT_CSV]` |
| Launch automatic submission and return | `just submit BUNDLE DATASET_ID NOTEBOOK_ID RECORD_JSON LABEL` |
| Inspect saved progress and stage durations | `just submit-status RECORD_JSON` |
| Read recent runner output and return | `just submit-logs RECORD_JSON [LINES]` |
| Explicitly follow live runner output | `just submit-follow RECORD_JSON [LINES]` |
| Query current Kaggle submission statuses/scores | `just submissions` |
| Submit in the foreground for debugging | `just submit-foreground BUNDLE DATASET_ID NOTEBOOK_ID RECORD_JSON LABEL` |

## Export and submit a checkpoint

For example, after selecting an actual checkpoint:

```bash
rtk proxy uv run --locked just export \
  outputs/training/overnight-20260914T220805Z-ozsz4qbu/runs/00-baseline/fold_0/best-epoch=09-val_macro_auc=0.89276.ckpt \
  outputs/submission/exports/00-baseline-next-2x
rtk proxy uv run --locked just submit \
  outputs/submission/exports/00-baseline-next-2x \
  claverru/knee-00-baseline-next-2x claverru/knee-00-baseline-next-2x-inference \
  outputs/submission/records/00-baseline-next-2x.json "00-baseline"
rtk proxy uv run --locked just submit-status outputs/submission/records/00-baseline-next-2x.json
```

## Runtime and resume behavior

Export uses `outputs/submission/dependencies/kaggle-v170`, Python 3.12 and Linux x86-64 by
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
`submit-follow` and `submit-foreground` intentionally wait.

## Verification record

Recipe syntax, lint, catalog consistency, and actual read-only status/log
commands were checked on 2026-09-11. The new launcher was not used to create
an extra competition submission during this change. No training, inference
checks, or tests were run.
