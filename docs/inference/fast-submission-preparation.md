# Fast submission preparation

Routine submission preparation runs **no local prediction**: no full-fold
parity, sample parity, example inference, or VRAM profiling. Export the selected
checkpoint directly, reusing the prepared codec wheel directory:

```bash
rtk proxy uv run --locked python -m scripts.pipeline.export \
  --checkpoint outputs/training/overnight-20260914T220805Z-ozsz4qbu/runs/00-baseline/fold_0/best-epoch=09-val_macro_auc=0.89276.ckpt \
  --output outputs/submission/exports/00-baseline-fold0-2x \
  --wheel-dir outputs/submission/dependencies/kaggle-v170 \
  --target-python 3.12 --target-platform manylinux_x86_64 \
  --image-multiplier 2 --inference-batch-size 4 --inference-num-workers 16
```

Use a fresh output path for each immutable bundle. Set `--image-multiplier 3`
for 3×. If omitted, the multiplier is **2** (`TEST_IMAGE_MULTIPLIER`), independently of the checkpoint's
saved validation configuration. The selected density is stored separately from training settings
in `config.json` and the manifest. Bundle prediction rejects a different density
override; export a new bundle for that comparison. The notebook uses the
exported per-GPU batch size and total loader workers, which default to 4 and 16
(`INFERENCE_BATCH_SIZE`, `INFERENCE_NUM_WORKERS` in `config.py`) with
[one process per visible GPU](independent-gpu-inference-processes.md). Precision `auto`
resolves to native CUDA BF16, otherwise CUDA FP16 with fixed 64-image channels-last backbone
chunks (T4), and FP32 on CPU. The wheel directory contains only the three codecs;
Kaggle provides `pytorch-lightning`, so no Lightning wheel is uploaded.

`manifest.json` records `preparation_seconds`, excluding Python process startup.
The actual baseline4 2× and 3× v2 exports took **0.64 s** and **0.56 s**
respectively. Export runs no local inference and records no verification; the
manifest's `checks` cover checkpoint metadata and the strict model load.
The submit runner records timestamped transitions in `history` and prints them
to its log. These distinguish local preparation, upload, notebook execution and
scoring waits; the last intervals include network latency and polling delay.

The optional diagnostic
[`scripts/tools/verify_inference.py`](../../scripts/tools/verify_inference.py) is no
longer part of export or bundled with it. Run it from the repository root with
`PYTHONPATH=<bundle>/code`, so it imports the exported package, and a request JSON with
`config` (the saved effective configuration), `reference_csv` (the selected epoch's saved
validation predictions), `samples`, `image_multiplier`, `batch_size`, `checkpoint_sha256`,
`bundled_source_sha256` and `codec_evidence`. It compares about 16 held-out studies using
complete original validation batches and the original validation density under native BF16,
then performs a bounded FP32 memory check at the requested density and predicts the three
example studies. It writes `verification.json`, `verification_sample.csv` and
`verification_example.csv` into the bundle directory, which then no longer matches its
manifest inventory; run it on a disposable copy of the bundle:

```bash
rtk proxy uv run --locked env PYTHONPATH=outputs/submission/verification/BUNDLE-copy/code \
  python -m scripts.tools.verify_inference --bundle outputs/submission/verification/BUNDLE-copy \
  --request-file outputs/submission/verification/BUNDLE-request.json
```

The sampled batches span low and high input counts. An ensemble bundle is checked the same way,
comparing the members' averaged prediction with the reference CSV. This is an on-demand diagnostic
after relevant inference changes, not a step to repeat for each submission. It does not
calculate a partial-fold AUC or require a full training run.

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
`outputs/submission/records/knee-overnight-05-faster-head-20260912.json`.
Do not repeatedly wait for remote completion in the interactive
session; monitor the record or runner log as needed.
