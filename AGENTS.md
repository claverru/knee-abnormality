# Project guidance

Workspace for the [RSNA Knee Abnormality Detection competition on Kaggle](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

- Consult the competition page for rules, data details, evaluation, and submission requirements.
- Use the repository's [$docs-maintainer](.agents/skills/docs-maintainer/SKILL.md) to consult, create, update, or audit repository documentation. Open known paths or sections directly, use `rg` for specific terms, and use the relevant folder index for broad topics; start with `docs/INDEX.md` when the destination is unclear.
- Keep indexes navigation-only, with distinctive "read when" descriptions and section links for common lookups. Update entries when documents are added, moved, removed, change purpose, or their navigation needs repair; a changed result inside a document does not require an index update.
- Use the repository's [$scripts-maintainer](.agents/skills/scripts-maintainer/SKILL.md) to find, create, update, move, or retire scripts. Open known paths directly or start at `scripts/INDEX.md`; keep the script indexes current in the same change.
- Use the repository's [$disk-cleanup](.agents/skills/disk-cleanup/SKILL.md) to free disk space; it deletes by fixed rules without asking and records the cleanup.
- Use the repository's [$submit-to-kaggle](.agents/skills/submit-to-kaggle/SKILL.md) to make a Kaggle submission: export, launch, relaunch a queued or slow notebook, and record the score.
- Use the repository's [$rented-machine](.agents/skills/rented-machine/SKILL.md) to connect to and work on the rented Vast.ai GPU machine: address, layout, syncing, trainings, exports, costs and vetting a new host.
- Prefer Polars for tabular data processing.
- Use Lightning for the training framework.
- Use `pydantic-settings` for CLIs.
- Keep experiments reproducible: record configurations, random seeds, and validation results.
- Keep datasets, model weights, and credentials out of version control.
- Make focused changes.
- No comments, no docstrings.
- Do not write or run tests.
- Treat accepted implementation decisions as fixed behavior, not configurable alternatives. Replace superseded implementations and remove their branches, flags, and configuration fields; update affected callers and documentation. Do not retain legacy behavior, compatibility fallbacks, or aliases for old configs, checkpoints, or APIs. Keep options only for parameters or alternatives we actually intend to vary, never for rejected approaches or speculative future use.
- Check GPU usage before launching anything in GPU.
- 0.001 is important in a 3 decimal competition.

## Work tracking

- The competition has ended; [docs/final-solution.md](docs/final-solution.md) records the solution it ended on.
- Keep active work in [docs/ongoing/](docs/ongoing/INDEX.md), empty when nothing is in progress. Update it in the same change when something starts or finishes.
- Record new ideas that are not started in [docs/backlog/untried-ideas.md](docs/backlog/untried-ideas.md), with the evidence behind each.
- The user launches training, detached from agent sessions.
- Before deleting a finished entry, preserve its durable findings in the relevant topic documentation.
