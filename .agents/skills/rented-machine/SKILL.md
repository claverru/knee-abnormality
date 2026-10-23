---
name: rented-machine
description: Connect to and work on the rented Vast.ai GPU machine (instance labelled knee-main) - find its address, its layout, sync code, run trainings and exports there, bring results back, check costs, and vet a newly rented host. Use whenever a task runs anything on the rented machine.
---

# Rented machine

The project rents one Vast.ai GPU machine for trainings and large data (Kaggle, fastMRI, OAI).
Renting, stopping and destroying instances are the user's call.

## Connect

- **Find it:** `~/.local/bin/vastai show instances` lists the instances (label `knee-main`);
  `~/.local/bin/vastai ssh-url <id>` gives `ssh://root@HOST:PORT`. The address changes with every
  rental, so look it up instead of reusing an old one.
- **SSH:** `ssh -i ~/.ssh/vast_knee -p PORT root@HOST`. Every command prints a two-line Vast
  banner; filter it with `grep -v -i "welcome\|have fun"`.
- **Copy:** `scp -P PORT` / `rsync -e "ssh -i ~/.ssh/vast_knee -p PORT"`.

## Layout

| Path | Holds |
|---|---|
| `/workspace/data` | `raw/` (Kaggle `train_series`, `test_series`, CSVs; `raw/fastmri`; `raw/oai`), `fastmri/` (the fastMRI data root), `folds`, `metadata`, `labels` |
| `/workspace/outputs` | `training/`, `submission/` (exports and `dependencies/kaggle-py313` wheels), `logs/` |
| `/workspace/logs` | data-build logs and `rebuild.state` |
| `/home/claudio/src/<worktree>` | copy of the local worktree, its own `.venv` (`uv sync --locked`), `data` and `outputs` linked to `/workspace` |

Paths mirror the local ones, so configs and recorded paths work on both sides.

## Work there

- **Sync code:** `rsync -a --exclude .git --exclude .venv --exclude __pycache__ --exclude /data
  --exclude /outputs <worktree>/ root@HOST:<worktree>/`. Never sync hashed sources while a
  training from that tree runs: its final prediction checks them.
- **Run:** start every job with `ulimit -n 1048576`, `PYTHONPATH=src:.` and `HF_HUB_OFFLINE=1`,
  inside `tmux new-session -d -s NAME "bash SCRIPT"`, writing its log under `/workspace/outputs/logs`
  and a one-line state per step to a state file; watch with a background loop that exits on the
  state change, not with a long Monitor.
- **Check the GPU first:** `nvidia-smi` before launching anything.
- **Keep CPU free:** no conversions, downloads or other heavy jobs beside a training or a timing.
- **Export there:** checkpoints trained on the machine record `/workspace` paths, so export them on
  the machine (wheels copied to `/workspace/outputs/submission/dependencies/`) and rsync the bundle
  back to the main checkout's `outputs/submission/exports/`; then submit with `$submit-to-kaggle`.
- **Bring results back:** the run directory without `last.ckpt` (delete it on the machine after
  the run); mind the free space on the local Windows drive before large copies.

## Credentials

Never leave a token or password on the machine. Fetch a signed download link locally (Kaggle,
NDA) and copy only the link; delete link files on both sides once the download ends.

## Costs

`~/.local/bin/vastai show user --raw` gives `credit`; `show instances --raw` gives `dph_total`,
which includes the disk. Storage bills while an instance is stopped; only destroying ends it.

## Vetting a new host

Before keeping a newly rented machine, copy `vet_host.py` from this folder and run it there in a
venv with `torch` and `timm` (about two minutes). It prints bf16 matmul, device and host-to-device
bandwidth, small-kernel launch time, a 60-second sustained ConvNeXt-Tiny load with power, clock and
"SW Power Cap" per sample, and CPU memory-copy scaling.

References from 2026-10-14: a healthy RTX PRO 6000 held about 7,700 img/s at 540 W, a healthy
RTX 5090 about 5,000 img/s at 420 W; a power-capped 5090 fell to 2,500 to 3,200 img/s at about
320 W with "SW Power Cap: Active". Memory copy should keep rising with processes (9 GB/s flat
starved the loader). Destroy a host that fails.
