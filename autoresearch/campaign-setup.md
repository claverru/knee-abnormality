# Adding a campaign

Add a directory under `autoresearch/campaigns/<id>/` with a contract, configuration,
initial proposals and empty runtime projections. **No runner changes are needed
for a new topic using the existing local training and evaluation protocol.**
The [near-misses campaign](campaigns/near-misses/INDEX.md) is the current example;
the historical [shared-loss campaign](campaigns/losses/INDEX.md) shows different
permissions using the same runtime.

## Files and startup

1. Write `contract.md`: objective, exclusions, fixed recipe, permitted mechanisms
   and evidence requirements. Link the
   [shared evaluation and adoption rule](protocol.md#shared-evaluation-and-adoption)
   instead of restating it. State the granularity of every fallback rule (per study
   or per token); knee-geometry's "for that study" was read both ways, and children
   were repeatedly sent back to repair behavior inherited from the incumbent. Record prior evidence,
   general-backlog ownership and blocked directions in `prior-work.md`, with a precedent
   table of measured runs outside autoresearch in the campaign topic (mechanism, setting,
   run, macro delta, source). The researcher checks proposals against that table instead
   of searching `docs/` on every refill.
2. Write `campaign.yaml` with a unique `id`, recipe, saved control, evidence runs,
   `backlog` settings (`refill_below`, `refill_to` of at most five, `seed_order`) and
   `policy`. The recipe's single `validation.folds` entry is the campaign fold, and the
   control must be trained on it. Artifacts go to `outputs/autoresearch/<id>`.
   The control must match the starting effective recipe apart from name, seed,
   output path and `reference_equivalences`. An equivalence names a recipe key the
   control's config lacks and the exact value that reproduces the control's
   behavior; validation rejects it if the control records the key or the recipe
   value differs. Use `{}` when there are none. A reference trained on source that differed
   only by now-removed, inactive code may instead be declared with
   `reference_config_exceptions` and `reference_identity_exceptions`: dotted paths (`[]` for
   every item of a list) that exist in the reference's recorded config or input identity and are
   removed from both sides before the start, sealing and comparison checks. Each needs a user
   authorization and a recorded proof that the removed code never ran. It supplies the fixed epochs, optimizer steps and 115% runtime
   ceiling; the baseline CPU replay (two epochs, extrapolated to the schedule) supplies realized triplet counts.
3. Put actionable proposals in `seed-ideas.yaml` using the
   [research-report schema](../src/knee_abnormality/autoresearch/campaign/reports.py). A report
   holds at most five ideas; put more in further `seed-ideas-*.yaml` files, which
   startup imports after `seed-ideas.yaml` as separate proposal events. Keep
   `ideas.yaml` as `revision: 0` plus `ideas: []`, and `backlog.yaml` as
   `revision: 0` plus `queue: []`. Create `findings.md` with an explicit no-results
   status; the runner later rewrites it as a decision summary and keeps the full
   orchestrator log in `outputs/autoresearch/<id>/findings.md`.
   With the default `backlog.seed_order: ranked`, startup submits seeds for
   orchestrator ranking; they are not preapproved. With `seed_order: forced`,
   startup queues every seed directly in file order (files as above, ideas in file
   order), outside the `refill_to` target. The orchestrator cannot reorder or park
   them except by blocking a seed at its own review, and the researcher does not
   run until every seed is completed or blocked; normal refills resume after that.
   Forced seeds cannot declare dependencies: order a follow-up after its parents
   instead.
4. List each file in the campaign's `INDEX.md` and add the directory to the
   [campaign index](campaigns/INDEX.md). Keep campaign work tracking here.
5. Validate without a provider, model call, GPU work, or training:

   ```bash
   rtk proxy uv run --locked python -m scripts.tools.autoresearch validate CAMPAIGN
   ```

   Replace `CAMPAIGN` with the new ID. This checks the definition, policy, contract
   presence, seed schema/IDs, editable files and control/recipe compatibility. It
   does not establish scientific correctness or execute candidate code.
6. The user starts the campaign:

   ```bash
   rtk proxy uv run --locked just autoresearch start CAMPAIGN --provider claude
   ```

   Either configured subscription provider works. Every command, including status,
   stop, resume, usage and import-ideas, takes the campaign ID; there is no default.

## Declarative policy

The strict [policy schema](../src/knee_abnormality/autoresearch/campaign/definition.py) rejects
unknown fields. All fields below are explicit in the configuration.

| Field | Effect |
|---|---|
| `editable_files` | Exact repository-relative files the candidate may change |
| `editable_directories` | Repository-relative helper directories whose descendants may change |
| `config_keys` | Exact dotted effective-config keys allowed to differ from the sealed recipe |
| `config_prefixes` | Config subtrees allowed to differ, matching the prefix itself or its dotted descendants |
| `list_item_fields` | Map of a config list key to mutable fields within each item; length, order and other item fields stay fixed |
| `input_identity_keys` | Dotted input-identity differences permitted during comparison; an empty list permits none |
| `require_source_change` | Whether a configuration-only candidate is rejected before review |
| `developer_instructions` | Topic-specific implementation guidance, delivered to the developer |
| `researcher_instructions` | Topic-specific search guidance, delivered to the researcher |

File paths are normalized and relative. Editable Python belongs under
`src/knee_abnormality/`, with the harness, evaluator and identity implementation
protected. `configs/training.yaml` is the only editable non-Python file. Scripts,
documentation, dependencies and campaign instructions remain outside candidate
scope. Choose narrow paths; permission to edit a shared file does not authorize
changing unrelated functions in it.

Name, seed and output path are always logistical differences. Near-misses gives each
idea family its own config prefix (`model.meniscus_readout`, `trainer.half_knee` and
so on) and permits `preprocessing_identity` differences. List-item permissions suit
sampling's `data.series_types: [max_images]`, without allowing plane or contrast
definitions to change. Losses permitted only `trainer.loss` and the class-balance
switch, with no input-identity exceptions. Do not add broad config prefixes or
identity exceptions merely to bypass a failed gate.

The runner seals the full policy in runtime state at initialization and includes
its operative permissions in role assignments. A later YAML/state mismatch is an
error. A different scope needs a new campaign; campaign agents cannot amend a seal,
and only an explicit user decision, recorded under the campaign's
`outputs/autoresearch/<id>/decisions/`, can. A candidate cannot edit its policy,
contract, role briefs or acceptance machinery. All topic decisions live in these declarations; adding an
ID does not require a Python dispatch branch.

## Shared invariants and limits

Every campaign still uses the existing training and saved-prediction comparison
scripts, the [shared evaluation and adoption rule](protocol.md#shared-evaluation-and-adoption),
the sealed epoch and optimizer-step budget, original runtime ceiling, and fixed
cohort/visit/triplet accounting. GPU leases, frozen sources,
review, seed handling and detached scheduling remain shared runtime behavior.
Those invariants are not policy switches. A fundamentally different training or
evaluation protocol requires separate harness work, not a permissive topic policy.

Mechanism correctness, shared-versus-per-label semantics, leakage and changes inside
an allowed shared file still need substantive orchestrator review. Static policy
validation is not proof that a candidate respects its scientific contract.
