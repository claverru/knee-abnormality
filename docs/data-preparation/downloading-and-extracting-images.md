# Downloading and extracting the images

Run [`scripts/tools/download_data.py`](../../scripts/tools/download_data.py) from the project
root when ready. It uses the installed uv environment and existing Kaggle
credentials; competition rules must already be accepted. No additional package
or external unzip program is needed.

```bash
rtk proxy uv run --locked python -m scripts.tools.download_data
```

The destination defaults to `data/raw/`; `--data-dir /absolute/path` changes it.
The script downloads 128 MiB ZIP chunks and extracts in archive order, keeping
three completed chunks plus the chunk being fetched. It never stores the whole
ZIP. Python's [ZIP64 support](https://docs.python.org/3.12/library/zipfile.html)
handles the large archive. Loading its directory requires a few GiB of available
RAM, including Python's per-file metadata.

An authenticated HEAD request and the full remote ZIP directory established
this snapshot on 2026-09-09:

| Item | Size/count |
|---|---:|
| Archive | 265,018,885,676 bytes (265.02 GB) |
| Extracted files | 569,763,893,144 bytes (569.76 GB) |
| Files, including the five CSVs | 819,640 |
| ZIP entry compression | DEFLATE |
| Archive ETag | `3d99a46825629f617aec1e9900ba7afc` |

This inventory supersedes an earlier estimate of roughly 550 GB raw. Keeping
both the archive and extracted files would require about
835 GB before filesystem overhead. At inspection, the Windows backing drive
had about 770 GB free. WSL's virtual filesystem reported more; its capacity
does not establish physical host availability.
[Microsoft's WSL disk-space explanation](https://learn.microsoft.com/en-us/windows/wsl/disk-space).

The script checks both filesystems at startup and periodically during extraction.
It budgets remaining raw files, filesystem overhead, temporary state, and a
50 GiB free-space reserve (`--reserve-gib` changes the reserve). On WSL it uses
PowerShell to identify the physical backing drive and stops if that check fails.
Preprocessing caches need a separate budget after download; the reserve is not
an estimate of their size.

For an overnight run, start a persistent terminal, then run the command above:

```bash
rtk proxy tmux new-session -s knee-download
```

Detach with **Ctrl-b, then d**; reconnect with
`rtk proxy tmux attach-session -t knee-download`. Keep Windows awake and WSL
running: tmux survives terminal disconnection, not host sleep or shutdown.

## Progress, recovery and completion

- Logs and state live in `data/raw/.download/`, or `.download/` inside the chosen
  destination. Follow `download.log` for progress. A lock prevents simultaneous
  instances using that destination.
- Rerun the same command after interruption. Partial chunks resume at their
  saved byte offset. Requests must confirm the exact range and archive ETag;
  retries refresh signed URLs. URLs and credentials are not persisted in state.
  The default is 20 attempts per network operation, with bounded backoff.
- New files are published only after extraction reaches EOF and passes the ZIP
  checksum. Existing files, including the audited CSVs, must match; conflicting
  files cause a stop and are left untouched. Completed files use a SQLite
  journal; unchanged size/mtime allows them to be skipped on later runs.
- `--verify-existing` rereads checksums of all existing files. Use it when file
  integrity is in doubt, including after a host crash; it adds a full raw-data
  read. For a reported cache/checksum problem, preserve the affected file and
  diagnose it before removing private cached chunks or replacing any raw data.
- Success produces `COMPLETE` in the log, exit status zero and `complete.json`
  with the snapshot identity, file count and extracted byte count. Until that
  happens, acquisition remains incomplete. Keep `.download/` for efficient
  recovery; its files and raw data are ignored by Git.

## Completed acquisition and local verification

The full download completed on 2026-09-10 at 02:33 Europe/Madrid (00:33 UTC),
after about 2 hours 30 minutes, with no logged warnings or errors.
`data/raw/.download/complete.json`, `inventory.json`, and the SQLite journal
agree on the snapshot and totals above; the journal's integrity check passed.

The local audit at 10:01 Europe/Madrid checked every journaled file on disk:
all 819,640 were regular files with the expected sizes and recorded modification
times, totaling 569,763,893,144 bytes. The 819,078 training DICOMs cover all
4,407 studies and 24,371 metadata series; the 557 example-test DICOMs cover all
3 studies and 15 series. No metadata series were missing or unexpected. The
five CSV SHA-256 fingerprints still match the [original audit](../metadata/INDEX.md).

Post-download CRC checks passed for 135 files: the five CSVs, 64 training and
64 example-test DICOMs selected by reservoir sampling over sorted journal paths
with seed `20260910`, and the two reported malformed slices below. All 128
randomly sampled DICOMs decoded into finite pixel arrays. All 130 inspected
DICOM headers matched their study, series and instance paths; this sample used
Explicit VR Little Endian transfer syntax. The two targeted malformed slices
failed pixel decoding despite matching the archive CRCs.

`data/metadata/download-health.json` records the source identity, sample paths,
counts, checksums, coverage, and exact pixel errors. This verifies acquisition
completeness and sampled pixel health, not every DICOM's decodability. The audit
checked every file's size but did not reread all 570 GB for post-download CRCs;
ZIP member CRCs were checked during extraction. Interruption recovery has not
been exercised. About 183.6 GiB remained free on the Windows backing drive at
the post-download check; recheck before allocating preprocessing caches.
