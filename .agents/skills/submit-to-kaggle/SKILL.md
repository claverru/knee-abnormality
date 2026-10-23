---
name: submit-to-kaggle
description: Make a new Kaggle submission from a trained checkpoint (export the bundle, launch the detached submission job, watch it, relaunch a queued or slow notebook, handle the known failures, record the score). Use when the user asks to submit a model, a run or an ensemble to Kaggle.
---

# Submit to Kaggle

Commands run with `uv run --locked python -m`. Five submissions a day, reset at
00:00 UTC; at most two GPU notebooks at once.

| Placeholder | Meaning |
|---|---|
| `OUT` | `outputs/submission` of the main checkout, whatever worktree the command runs from |
| `NAME` | Short name of this bundle, such as `rn50-detail-view-f1-x4-tau09` |
| `NN` | Submission number |
| `NOTEBOOK` | `claverru/sNN-NAME` |
| `RECORD` | `OUT/records/NAME.json` |

1. **Check it is new.** Compare the checkpoint's SHA-256 with `checkpoint.source_sha256` of every
   `OUT/exports/*/manifest.json` and with `OUT/kaggle-ledger.json`.
   Never submit the same weights twice; the same weights with other inference settings need the
   user's approval.
2. **Export** from the worktree whose code trained the checkpoint, since the bundle carries that
   code: `scripts.pipeline.export --checkpoint CKPT... --output OUT/exports/NAME
   --image-multiplier M --attention-temperature T`, adding `--reflection-tta` when wanted. Several
   checkpoints make an ensemble; at x6 an ensemble needs `--inference-batch-size 2
   --inference-num-workers 8`. The default `--backbone graph` compiles each member's backbone and
   captures it in a CUDA graph; pass `--backbone eager` for an ensemble whose graph pools do not fit
   the T4 (about 0.31 GiB per member on top of the weights; 20 mixed ConvNeXt, NextViT and
   ResNet-50 members peaked at 12.6 of 14.6 GiB at x4). `manifest.json` must say `ready`.
3. **Launch.** `scripts.tools.submission_job start --bundle OUT/exports/NAME
   --dataset claverru/knee-NAME --notebook NOTEBOOK --record RECORD --label "LABEL"
   --sequence NN`.
   - `NN`: one after the live count (`kaggle competitions submissions
     rsna-knee-abnormality-detection`), plus any job that has not submitted yet. Always pass it.
   - `LABEL`: the run and its one change, 2 to 45 characters, without the `S` number. It is
     permanent on Kaggle.
   - Slugs: 6 to 50 characters.
   - `start` prints the job's unit name, `UNIT` below.
4. **Watch.** `scripts.tools.submission_job status --record RECORD` gives the state and the seconds
   of each stage; `logs` prints the runner log. A bundle without `compile-cache.bin` first gets
   one: the job uploads it, runs a `NOTEBOOK-cache` notebook on the visible test
   (`compile_cache_running`), adds the cache to the local bundle and re-verifies it
   (`compile_cache_built`). It then submits the notebook version itself and ends at
   `submission_complete` with the public score.
5. **Record.** `scripts.tools.update_submission_ledger`, then the score in `docs/ongoing/` and the
   run's result document.

## Relaunch

Relaunch when either holds:

- the notebook is **queued**: `kaggle kernels status NOTEBOOK` still says `QUEUED` a minute after
  the record reached `notebook_running` (or `compile_cache_running`, for the record's
  `compile_cache_notebook`);
- the `notebook_running` stage has lasted **double** that of a similar reference: the same stage in
  the record of an earlier submission with the same backbone, member count, multiplier and TTA.

To relaunch:

1. `systemctl --user stop UNIT`.
2. `uv run --locked kaggle kernels delete -y NOTEBOOK` (or the compile-cache notebook).
   Deleting the notebook ends its run and frees the slug and title. A bundle that already holds
   `compile-cache.bin` skips the cache build on the next start.
3. `start` again with the same arguments and a new record path, such as `OUT/records/NAME-r2.json`.

Repeat until a notebook completes.

If the delete fails or the run survives it, in this order:

1. **New slug and label.** `start` with the notebook slug suffixed `-b` and the label suffixed ` b`.
   It needs nothing from the stalled notebook, which stays behind.
2. **Cancel by session id**, as in
   [cancelling a notebook run](../../../docs/inference/kaggle-submission-workflow.md#cancelling-a-notebook-run).
3. **The website.** Cancel the run from the notebook's page.

One live job per bundle. Stop the old job before starting the next: two jobs on one bundle both
submit.

## Known failures

| Seen | Do |
|---|---|
| `429` on `ListSubmissions` at start | Pass `--sequence`; wait a minute and launch again |
| `already has an active run`, or `SaveKernel` `409` | The slug or title belongs to a stalled notebook: delete it as in the relaunch |
| `SaveKernel` `400` | Title over 50 characters: shorten the label |
| `Maximum batch GPU session count of 2 reached` | Wait for a running notebook to end |
| Notebook `error` | Read its log (`kaggle kernels output NOTEBOOK -p DIR`), fix the bundle, export under a new name |
| `CreateCodeSubmission` `400` | Daily limit reached: submit the finished notebook version after the reset |
| `JSONDecodeError`, `503` or an interruption at or after `output_verified` | The submission may exist: read the live list first. If absent, submit that notebook version by hand, never relaunch |
| Hidden rerun fails (`errorDescription`, no score) | The slot is spent. The visible run has three studies, so memory and time faults show only here: lower batch and workers, submit as a new number |

Submit a finished notebook version by hand with
`KaggleApi().competition_submit_code("submission.csv", DESCRIPTION, "rsna-knee-abnormality-detection", NOTEBOOK, VERSION, False)`,
`DESCRIPTION` and `VERSION` taken from the record (`description`, `notebook_version`).

Avoid:

- `pkill -f` on a record name: it matches the submit job too.
- Replacing a bundle or its dataset while its notebook runs or scores.
- Searching `/kaggle/input` recursively in a hand-built notebook: it lists every DICOM folder.

Details: [workflow](../../../docs/inference/kaggle-submission-workflow.md),
[ledger and naming](../../../docs/inference/submissions/ledger.md).
