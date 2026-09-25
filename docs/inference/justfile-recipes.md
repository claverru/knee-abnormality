# Justfile recipes

The root [justfile](../../justfile) covers the maintained path. Run recipes as
`rtk proxy uv run --locked just RECIPE ...` from the repository; the table uses
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
| Export with the default 2× images, four studies per GPU and sixteen loader workers | `just export --checkpoint CHECKPOINT --output BUNDLE` |
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

For example, after selecting an actual checkpoint:

```bash
rtk proxy uv run --locked just export \
  --checkpoint outputs/training/overnight-20260914T220805Z-ozsz4qbu/runs/00-baseline/fold_0/best-epoch=09-val_macro_auc=0.89276.ckpt \
  --output outputs/submission/exports/00-baseline-next-2x
rtk proxy uv run --locked just submission-job start \
  --bundle outputs/submission/exports/00-baseline-next-2x \
  --dataset claverru/knee-00-baseline-next-2x --notebook claverru/knee-00-baseline-next-2x-inference \
  --record outputs/submission/records/00-baseline-next-2x.json --label "00-baseline"
rtk proxy uv run --locked just submission-job status --record outputs/submission/records/00-baseline-next-2x.json
```

## Runtime and resume behavior

Export uses `outputs/submission/dependencies/kaggle-v170`, Python 3.12 and Linux x86-64 by
default (`knee_abnormality.kaggle`); `--wheel-dir`, `--target-python` and `--target-platform`
override them. `prepare-codecs` is a separate one-time download of the pinned codec wheels for
that runtime and includes no Lightning wheel. It is never invoked automatically by training or
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
