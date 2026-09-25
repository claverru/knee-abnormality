# Kaggle submission workflow

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
| Submit a particular notebook version | `kaggle competitions submit COMPETITION -k OWNER/NOTEBOOK_SLUG -v VERSION -f submission.csv -m DESCRIPTION` |
| Poll scoring status and available scores | `kaggle competitions submissions rsna-knee-abnormality-detection -v` |

`-m` is the submission description, the only text Kaggle keeps for a submission row, and it is
write-once: the API exposes submit, status, list and download but no update, and the site offers
no edit control either. `scripts.pipeline.submit` composes it in the
[unified submission format](submissions/ledger.md#naming) from one `--label`, so use the recipe
rather than a hand-written message. The notebook title shown beside a submission can be changed
afterwards: `kernels push` would re-run the notebook, but the
[ledger script](submissions/ledger.md#regenerating-the-ledger) renames it with an unexecuted quick save.

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

Bundle readiness records successful packaging checks. `scripts.pipeline.submit` refuses an
existing record path, an unready bundle and a notebook whose latest run is still active, and
writes each remote response to the record before advancing. The remote notebook
must still finish and produce valid output before automatic submission. A
successful scored submission confirms the hidden-data execution path for that
artifact. Baseline3 and baseline4's original submissions both completed scoring.

A [submission troubleshooting report](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735854)
identifies P100/PyTorch build incompatibility, missing DICOM codecs and nested
`/kaggle/input/competitions/` or `/kaggle/input/datasets/` mount layouts. Treat
these as environment-specific checks: test an actual GPU operation and
representative compressed DICOM decoding, make dependencies available offline,
and use bounded input discovery. A CUDA availability flag or a successful
uncompressed sample is insufficient. See [environment](../architecture/environment.md) for the
repository's actual pinned versions.
