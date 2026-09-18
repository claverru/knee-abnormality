---
name: scripts-maintainer
description: Find, create, update, organize, and index scripts in this competition repository. Use when locating an existing utility, adding a runnable script, changing script inputs or outputs, or moving, promoting, and retiring scripts. Keeps script catalogs aligned with the implementation.
---

# Scripts Maintainer

Read [scripts/INDEX.md](../../../scripts/INDEX.md) from the repository root for
discovery commands, catalog fields, role routing, and growth conventions. The
Markdown link is relative to this skill; command paths are repository-relative.
Apply the repository's `AGENTS.md` instructions throughout.

## Find and reuse

Start with a focused catalog search for the operation and its inputs or
outputs. Prefer `--role tools --status active` for reusable commands; include
`adhoc` when looking for previous investigations or evidence. Read short search
results, then `show --id` for the strongest candidates. Open only the relevant
source and linked procedure before deciding whether it fits.

An empty search is not proof that code is absent. Try synonyms or fewer terms,
then use targeted `rg --files` and `rg -n` in the likely topic or role. For an
explicitly named path, inspect it directly and look up its entry as needed.
Do not read every catalog or Python file into context for an ordinary lookup.
If an existing script fits, reuse it; extend a close match when that preserves
its purpose and existing callers. Report the path, appropriate invocation, and
relevant assumptions. Finding a command does not require executing it.

## Create or change

Search before introducing a new entry point. Use the role routing in the
index: pipeline for the maintained prepare/train/predict workflow, tools for
repeated supporting operations, adhoc for focused investigations. Keep shared
logic in the location defined by the project architecture when several callers
need it. Give scripts descriptive
action-oriented snake_case names and retain stable catalog IDs across moves.

Implement the actual requested behavior; do not create empty script scaffolds
or placeholder catalog entries. Follow the existing explicit CLI settings
pattern in the shared config module: use `pydantic-settings`, protect runnable
behavior with a main guard, and keep imports free of dataset processing.
Launch from the repository root using `python -m scripts.<role>.<module>` and
follow the architecture's shared-code import conventions.
Keep experiment data and output artifacts in their documented locations.

Add or update the entry in the owning `catalog.yaml` in the same change.
Record a concise purpose, useful search tags, realistic commands, inputs,
outputs, and assumptions that affect execution or reuse. Link canonical
procedures and findings rather than copying them into the catalog. When the
interface changes, update commands and affected callers too.

## Move, promote, or retire

Promote an investigation when it becomes a repeated supporting operation or
part of the final workflow. Move the useful implementation rather than keeping
parallel copies. Extract shared logic when required by multiple callers.
Move the catalog entry to the destination's catalog and preserve its ID.
Search for old file paths and module names in callers and documentation;
check `__file__`-relative paths and imports after changing directory depth.

Keep useful past work as `historical` with its assumptions and a link to the
findings. For a retained script with a replacement, use `superseded` and the
active replacement's ID. Before deleting a script, inspect its callers and
preserve unique findings in topic documentation. Remove deleted entries and
repair replacement references; do not leave catalog records pointing at absent
files. Never alter recorded dataset or artifact hashes to disguise a code move.

## Finish efficiently

Run `rtk proxy uv run --locked python -m scripts.tools.catalog check` after
catalog or script changes and resolve missing, duplicate, or misplaced entries.
Perform the narrow static or actual development checks appropriate to the
change, within project rules; do not write or run tests. Catalog checks never
execute indexed scripts, and passing them does not prove runtime correctness.
Review the diff and affected links. Keep work tracking in `docs/ongoing/`
and durable findings in the relevant topic docs.

For a requested collection audit, first check catalog consistency, then review
one relevant role or topic at a time for misleading metadata, overlapping
behavior, and obsolete scripts. An ordinary lookup or localized change does
not require a full collection audit. Split catalogs into topic groups only
when the existing collection becomes cumbersome to navigate or maintain.
