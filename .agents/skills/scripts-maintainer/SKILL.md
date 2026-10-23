---
name: scripts-maintainer
description: Find, create, change, move and retire scripts in this repository and keep the script indexes current. Use when locating an existing command, adding a runnable script, changing a script's CLI, or promoting or retiring a script.
---

# Scripts maintainer

Apply the repository's `AGENTS.md` throughout. Paths are repository-relative.

## Find

- Start at `scripts/INDEX.md`. The `pipeline/` and `tools/` indexes are generated from each
  script's CLI description; `adhoc/INDEX.md` is written by hand, by topic, with each
  investigation's question and findings doc.
- `uv run --locked python -m scripts.<role>.<module> --help` gives the purpose and
  every option. `rg <file stem> docs` finds procedures and findings.
- Reuse or extend a close match before adding a script.

## Place

- `pipeline/`: the maintained submission path. `tools/`: repeated supporting operations; they
  must run against current code. `adhoc/`: one investigation, frozen once it finishes.
- Logic, option groups and types used by more than one script belong in `src/knee_abnormality`
  ([code organization](../../../docs/architecture/code-organization.md)). Shared option models:
  `cli.py` (`Runtime`, `Bootstrap`, `RunFold`), `kaggle.py` (`NotebookRun`,
  `SubmissionRequest`, `KaggleId`, `COMPETITION`); `config.py` holds `Device`,
  `RuntimePrecision`, `STUDY_HEADERS` and the dotted experiment overrides. Inherit these
  instead of redeclaring fields; add a new shared model when fields repeat together.

## Write a CLI

- Subclass `CLISettings`. `model_config = describe("...")` states the purpose and every field
  has `Field(description=...)`. Booleans are flags.
- Related actions are subcommands: one `BaseModel` per action with a `cli_cmd` method,
  `name: CliSubCommand[Model] = command(Model)` on the root, whose `cli_cmd` calls
  `CliApp.run_subcommand(self)`; `main()` calls `CliApp.run(RootCLI)`.
- No `try`/`except` around `main`: argument errors print usage and exit 2, other errors raise.
- Keep imports free of data processing and run behavior under `if __name__ == "__main__"`.
  Launch as `python -m scripts.<role>.<module>` from the repository root.
- An experiment setting is never a new training flag; use the dotted overrides
  (`--trainer.class-balance-exponent 0.5`).

## Keep the indexes current, in the same change

- After adding, removing, moving or re-describing a pipeline or tools script, run
  `uv run --locked python -m scripts.tools.script_index write`.
- For an adhoc script, add or edit its row in `scripts/adhoc/INDEX.md`.
- Finish with `... script_index check`. It runs every maintained script's `--help`, so a
  failure can also mean a tool no longer imports.
- Add a justfile recipe only for routine commands; recipes pass their arguments through and
  are named after the module.

## Promote or retire

- Promote with `git mv` to `tools/`, add the descriptions, update every reference (`rg` the
  old path and module name) and regenerate the index.
- A finished investigation stays in `adhoc/` as the record behind its findings. Delete a
  script only when no documented finding depends on it; its findings doc then cites the last
  commit that contains it as `commit:path`. Do not keep text copies of deleted code.
- Never alter recorded dataset or artifact hashes to hide a move.
