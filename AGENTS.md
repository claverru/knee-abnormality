# Project guidance

Workspace for the [RSNA Knee Abnormality Detection competition on Kaggle](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

- Consult the competition page for rules, data details, evaluation, and submission requirements.
- Use `$docs-maintainer` to consult, create, update, or audit repository documentation; start with `docs/INDEX.md` when present.
- Use the repository's [$scripts-maintainer](.agents/skills/scripts-maintainer/SKILL.md) to find, create, update, move, or retire scripts; start with `scripts/INDEX.md` and keep catalog entries current in the same change.
- Prefer Polars for tabular data processing.
- Use Lightning for the training framework.
- Use `pydantic-settings` for argument parsing and configuration in all new CLIs.
- Keep experiments reproducible: record configurations, random seeds, and validation results.
- Keep datasets, model weights, and credentials out of version control.
- Make focused changes.
- No comments, no docstrings.
- Do not write or run tests.
- Treat accepted implementation decisions as fixed behavior, not configurable alternatives. Replace superseded implementations and remove their branches, flags, and configuration fields; update affected callers and documentation. Do not retain legacy behavior, compatibility fallbacks, or aliases for old configs, checkpoints, or APIs. Keep options only for parameters or alternatives we actually intend to vary, never for rejected approaches or speculative future use.

## Work tracking

- Keep active work and tasks planned for the short term in [docs/ongoing.md](docs/ongoing.md). Delete entries once done, preserving durable findings in the relevant topic documentation first.
- Keep ideas to explore in [docs/future-work.md](docs/future-work.md). When work on an idea starts, move its entry to `docs/ongoing.md` and remove it from `docs/future-work.md` in the same update.
