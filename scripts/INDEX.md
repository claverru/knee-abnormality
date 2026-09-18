# Script index

Start here to find an existing command or decide where a new script belongs.
The catalogs hold script metadata; the [maintenance skill](../.agents/skills/scripts-maintainer/SKILL.md)
defines how agents search, create, move, and promote scripts. Component
boundaries live in the [project architecture](../docs/architecture/INDEX.md).

## Find a script

Run from the repository root:

```bash
rtk proxy uv run --locked python -m scripts.tools.catalog search --query "fold auc"
rtk proxy uv run --locked python -m scripts.tools.catalog search --query labels --role adhoc
rtk proxy uv run --locked python -m scripts.tools.catalog show --id validation
rtk proxy uv run --locked python -m scripts.tools.catalog check
```

Search returns up to ten short JSON entries and a total count. Use `--limit`
and `--offset` for more results; `next_offset` identifies the next page.
`--role pipeline|tools|adhoc` and `--status active|historical|superseded` narrow
the results. Without a status filter, historical entries remain discoverable.
Every query term must occur in the ID, path, summary, tags, inputs, outputs,
or notes; matching ignores case. Try synonyms or fewer terms before concluding
that no script exists. An empty query lists entries in ID order.

`show --id` returns the complete entry and its owning catalog, including
commands, inputs, outputs, limitations, and linked procedures. Read the selected
source before changing or executing it. Catalog commands are usage examples;
discovery never imports or executes indexed scripts.

## Choose a location

| Need | Location | Catalog |
|---|---|---|
| Prepare data, train the solution, or predict a submission | `pipeline/` | [pipeline/catalog.yaml](pipeline/catalog.yaml) |
| Reuse a supporting command across experiments | `tools/` | [tools/catalog.yaml](tools/catalog.yaml) |
| Investigate a question, profile behavior, or reproduce an audit | `adhoc/` | [adhoc/catalog.yaml](adhoc/catalog.yaml) |

The catalog covers Python scripts under `scripts/`, excluding `__init__.py`.
Non-Python assets belong to the script that reads them and appear in its
`inputs`, such as the bundle notebook template `pipeline/inference.ipynb`.
Shared implementation modules and the config inspector remain described
in the [architecture](../docs/architecture/code-organization.md#where-code-belongs). Proposed
scripts belong in [ongoing work](../docs/ongoing/INDEX.md) until implemented.

## Catalog format

Each `catalog.yaml` has `version: 1` and a `scripts` list. Keep one entry per
script, in a catalog in its own directory or a parent directory. All script
and documentation paths are relative to the repository root.

| Field | Contents |
|---|---|
| `id` | Unique stable name using lowercase letters, digits, and hyphens; retain it when moving a script |
| `path` | Current Python file under one of the three script roles |
| `summary` | One sentence describing when to use the script |
| `tags` | Search vocabulary, including domain terms and useful synonyms |
| `status` | `active` by default; `historical` for retained reference work; `superseded` when replaced |
| `commands` | Complete usage examples; active scripts require at least one |
| `inputs`, `outputs` | Files, artifacts, or resources consumed and produced; empty lists are allowed |
| `notes` | Optional material assumptions, side effects, limitations, and required resources |
| `docs` | Optional links to canonical procedures or findings, using repository-relative paths and optional anchors |
| `superseded_by` | Required only for superseded entries; ID of an active replacement |

The schema is implemented in [tools/catalog.py](tools/catalog.py). Describe
searchable behavior in metadata, with detailed reasoning in the topic docs.
Keep comments and docstrings out of code and YAML according to project guidance.

## Maintenance and growth

Update code, its catalog entry, and affected references in the same change.
After creating, moving, deleting, or changing a script's interface, run the
catalog `check` command. It validates metadata, unique IDs and paths, catalog
ownership, coverage of Python scripts, documentation file existence, and
replacement references. It performs static checks only; it does not validate
command behavior, documentation anchors, data availability, or semantic duplicates.
Exit codes are 0 for success, 1 for consistency issues, and 2 for invalid
metadata or arguments. Inspect candidate scripts to judge whether they overlap.

Keep this entry point small as the collection grows. When a category becomes
difficult to scan or edit, group related scripts into topic subdirectories,
such as `tools/labels/`, and give each group its own `catalog.yaml`. Move entries
with their scripts and retain IDs. The search command discovers catalogs
recursively; no generated global listing needs synchronization. Do not create
empty topic directories in advance. Add a topic README only for shared
instructions beyond individual catalog entries, and link it through their
`docs` fields.

The repository skill lives under `.agents/skills/`, following the
[official local skill discovery layout](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills).
The root `AGENTS.md` also links it directly so the maintenance instructions can
be read even when a session has not refreshed its skill list.

## Spatial-token diagnosis and retired work

Use `diagnose-encoding-usage` for the manually invoked post-training spatial-token
interventions; see [the procedure](../docs/validation/manual-encoding-usage-diagnosis.md).
Removed regional coverage, usage and profiling commands are preserved as
[historical source snapshots](../docs/history/regional-sources/README.md), outside
the executable catalog. Reproduce them only with their original analysis/run
source bundles and configurations.

The [historical sharing sources](../docs/history/sharing-sources/README.md)
preserve the 1C compartment diagnostic, CPU generalization/case-mixture audits,
blinded disagreement audit and eleven-run overnight launcher. They are text
records outside the executable catalog and require their original runtimes.
The current sixteen-run launcher does not reproduce that historical queue.

The crop-selection scripts `analyze_crop_diversity.py`, `evaluate_crop_placement.py` and
`probe_glimpses.py` were removed with the crop-selection code on 2026-09-16; the
[historical crop-selection sources](../docs/history/crop-selection-sources/README.md) locate
their archived code. The token graph's edge diagnostics were removed from
`diagnose-encoding-usage` with the graph; the
[historical token-graph sources](../docs/history/token-graph-sources/README.md) keep them.

`analyze_knee_side_embedding.py` and `analyze_meniscus_coupling.py` were removed on
2026-09-16 with their candidate runs' checkpoints; the
[historical knee-side sources](../docs/history/knee-side-sources/README.md) and
[historical meniscus-coupling sources](../docs/history/meniscus-coupling-sources/README.md)
keep their code.
