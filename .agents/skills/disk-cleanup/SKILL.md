---
name: disk-cleanup
description: Free disk space by fixed rules, without asking, across every worktree and the home caches (obsolete checkpoints, export weights of unsubmitted models, regenerable analysis arrays, download chunks, tool caches), and record the cleanup. Use when the user asks to free disk space, clean the disk or prune checkpoints.
---

# Disk cleanup

Run it to the end without asking the user. The rules decide; whatever no rule covers stays and
is reported.

- **Run:** `uv run --locked python -m scripts.tools.free_disk` from the repository
  root. It scans every worktree's `outputs/` and `data/raw/.download`, the Hugging Face cache,
  `~/.codex` and the session scratchpads, deletes, and writes
  `outputs/archive/maintenance/disk-cleanup-<UTC>/manifest.json` with every deleted and kept
  path, its reason and size. `--dry-run` prints the plan and deletes nothing.
- **uv cache:** then run `uv cache prune` on its own, not under `uv run`, which holds the cache
  lock. If another uv process holds the lock, skip it and say so; never pass `--force`.
- **Record:** add one dated paragraph to
  [checkpoint pruning](../../../docs/architecture/configs-and-experiment-artifacts.md#checkpoint-pruning):
  the totals, the manifest link, what went by rule, and what was kept for a decision. Leave it
  uncommitted.
- **Report:** GiB freed per rule, the `Kept, needs a decision` list, any path that failed to
  delete, and that Windows only regains the space once the WSL virtual disk is compacted.
- **Never** delete by hand what the script kept, change a rule to free more, or touch
  `data/raw` series, labels, folds, predictions, metrics, configs, source snapshots, manifests
  and campaign states.

## Rules

Every row also requires that no live process uses the path (working directory, open file, or a
command line naming it). There is no grace period: `docs/ongoing/` and the "Runs to keep" section of
`docs/final-solution.md` are the list of runs still needed, so a run to keep must be named in one of
them before the cleanup runs.

| Deleted | When |
|---|---|
| `last.ckpt` and non-selected checkpoints | The fold has a `selected_checkpoint.json` and its campaign, if any, is `stopped`; a pretraining's `last.ckpt` (run name ending in `-pretrain`) is kept as the start weights of later stages |
| Selected checkpoint | Same, and the run is named in no worktree's `docs/ongoing/` or `docs/final-solution.md`, while its result is recorded: the run is named somewhere in `docs/` or in its campaign's `state.json` |
| Export weights (`model.ckpt`, `models/*.ckpt`, `wheels/`) | The bundle is absent from `kaggle-ledger.json`, and neither it nor its source run is named in `docs/ongoing/` or `docs/final-solution.md`; checkpoints also need the bundle or its source run named somewhere in `docs/` |
| Arrays of 50 MiB or more (`.npy`, `.npz`, `.pt`, `.pkl`) in `outputs/analysis`, `profiling`, `previews` | The investigation folder is named neither in `docs/ongoing/`, `docs/final-solution.md` nor in `scripts/`, `src/` or `configs/` |
| `data/raw/.download/chunks` | `complete.json` exists |
| Cached Hugging Face models | No file in `scripts/`, `src/`, `configs/` or `z_labels/` of any worktree names the model |
| Codex package releases | Not the target of `current` |
| Build folders inside Codex reviews and debates | Cargo target, tagged cache folder, virtual environment or `__pycache__`, and no live process anywhere in that review or debate |
| Session scratchpads under the temporary directory | The session is not one of the running Claude sessions in `~/.claude/sessions` |
| uv cache entries | Whatever `uv cache prune` removes while no other uv process holds the cache |

A checkpoint that a bundle in the ledger was exported from is never deleted, whatever its run.

Kept and listed for a decision: checkpoints of a run without a selection record, selected and
exported checkpoints whose result is recorded nowhere, and checkpoints outside a `fold_<k>`
folder.
