# Project guidance

Workspace for the [RSNA Knee Abnormality Detection competition on Kaggle](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

- Consult the competition page for rules, data details, evaluation, and submission requirements.
- Use `$docs-maintainer` to consult, create, update, or audit repository documentation. Open known paths or sections directly, use `rg` for specific terms, and use the relevant folder index for broad topics; start with `docs/INDEX.md` when the destination is unclear.
- Keep indexes navigation-only, with distinctive "read when" descriptions and section links for common lookups. Update entries when documents are added, moved, removed, change purpose, or their navigation needs repair; a changed result inside a document does not require an index update.
- Use the repository's [$scripts-maintainer](.agents/skills/scripts-maintainer/SKILL.md) to find, create, update, move, or retire scripts. Open known paths directly or start at `scripts/INDEX.md`; keep the script indexes current in the same change.
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

- Keep a snapshot of the current state and active work in [docs/ongoing/](docs/ongoing/INDEX.md). Update it in the same change when something starts or finishes.
- Classify every new idea into exactly one [backlog](docs/backlog/INDEX.md) document:
  - `docs/backlog/overnight.md`: hyperparameter-only runs on the current recipe, one change per run against a fresh control.
  - `docs/backlog/small-changes.md`: small code additions behind one new option. The user supervises them: implement one only when the user asks, then move its run to `overnight.md`.
  - `docs/backlog/big-bets.md`: investigation-sized plans. Start one only in a session with the user, and move it to `docs/ongoing/` when it starts.
  - `docs/backlog/endgame.md`: final-phase work; it waits for the final phase.
  - `docs/backlog/refactors/`: tooling and code-structure changes that leave training behavior unchanged, one document per refactor.
- The user launches training, detached from agent sessions.
- Before deleting a finished entry, preserve its durable findings in the relevant topic documentation.
