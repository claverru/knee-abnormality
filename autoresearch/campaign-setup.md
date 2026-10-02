# Adding a campaign

Add a directory under `autoresearch/campaigns/<id>/` with a contract, configuration,
initial proposals and empty runtime projections. **No runner changes are needed
for a new topic using the existing local training and evaluation protocol.** Under the
[topic loop](topic-rotation.md), drop the topic into the
[topic queue](topics/INDEX.md#dropping-a-topic) instead: the loop writes the recipe,
reference and chain and creates the campaign directory.
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
   `backlog` settings (`refill_below`, `refill_to` of at most five, `seed_order`, and the
   required `rotate_after`, at least 1: the researcher-run streak without an adoption after
   which the [topic loop](topic-rotation.md#trigger) rotates the campaign; drafted topics
   get 6), `budget` ([training budget](#training-budget)) and
   `policy`. The recipe's single `validation.folds` entry is the campaign fold, and the
   control must be trained on it. Artifacts go to `outputs/autoresearch/<id>`.
   The control must match the starting effective recipe apart from name, seed,
   output path and `reference_equivalences`. A topic the loop chains on an incumbent
   instead takes a `chain` block (predecessor, incumbent, source, run, prediction hash):
   its recipe is that run's `effective_config.json` and its reference is the run itself,
   with no equivalences or exceptions, and it inherits the predecessor's runtime ceiling
   and sealed counts ([chaining](topic-rotation.md#chaining)). An equivalence names a recipe key the
   control's config lacks and the exact value that reproduces the control's
   behavior; validation rejects it if the control records the key or the recipe
   value differs. Use `{}` when there are none. A reference trained on source that differed
   only by now-removed, inactive code may instead be declared with
   `reference_config_exceptions` and `reference_identity_exceptions`: dotted paths (`[]` for
   every item of a list) that exist in the reference's recorded config or input identity and are
   removed from both sides before the start, sealing and comparison checks. Each needs a user
   authorization and a recorded proof that the removed code never ran. The reference supplies the
   [training budget](#training-budget); the baseline CPU replay (two epochs) supplies realized
   triplet and visit counts.
3. Put actionable proposals in `seed-ideas.yaml` using the
   [idea-report schema](../src/knee_abnormality/autoresearch/campaign/reports.py) (`IdeaReport`; researcher refills add `source_change` and `targeted_curve`). A report
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
   run until every seed has launched, completed or been blocked; the first refill then runs
   beside the last seed's training.
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
   stop, resume, usage, import-ideas, move-idea, remove-idea and adopt, takes the campaign ID;
   there is no default.
   To let autoresearch continue past this topic on its own, start it as the first topic of
   the loop instead: `just autoresearch loop-start --campaign CAMPAIGN --provider claude`
   ([runtime](runtime.md#topic-loop)).

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
| `input_identity_keys` | Dotted input-identity differences permitted during comparison; an empty list permits none. The supervisor also removes each key from both identities before the comparison, so a nested key such as `preprocessing_identity.sources_sha256` works with a campaign's frozen comparison script |
| `require_source_change` | Whether a configuration-only candidate is rejected before review |
| `configuration_only_ideas` | Allows configuration-only ideas (requires `require_source_change: false`) while each nonempty research refill must carry at least one idea with `source_change: true`; such an idea must then change source. `false` in every existing campaign |
| `developer_instructions` | Topic-specific implementation guidance, delivered to the developer |
| `researcher_instructions` | Topic-specific search guidance, delivered to the researcher |

File paths are normalized and relative. Editable Python belongs under
`src/knee_abnormality/`; the protected list (`PROTECTED_SOURCES` and `PROTECTED_CONFIG`
in the policy schema) rejects the autoresearch harness, the evaluator, the input-identity
implementation (`identity.py`, `training/checkpoints.py`), the split and evaluation targets
(`data/splits.py`), and config paths for the name, seed, paths, labels,
validation, batch size, gradient accumulation and validation multiplier, including
any prefix that contains one. `trainer.max_epochs` is protected unless the campaign's budget is a
wall-clock range. `input_identity_keys` cannot name an identity key of the validation frame, split
or label source (`validation_rows_sha256`, `membership_sha256`, `split_protocol_sha256`,
`label_manifest_sha256`, `label_source`): the fold's validation frame, which checkpoint selection
reads, is assembled in the editable `data/labels.py`, so the comparison must always reject a changed
one. Training-target construction (`data/labels.py`) is editable;
the label source (`labels`) is not. `configs/training.yaml` is the only editable non-Python file.
Scripts, documentation, dependencies and campaign instructions remain outside candidate
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
error. A different scope needs a new campaign, which the topic loop creates itself on a
rotation; campaign agents cannot amend a seal,
and only an explicit user decision, recorded under the campaign's
`outputs/autoresearch/<id>/decisions/`, can. A candidate cannot edit its policy,
contract, role briefs or acceptance machinery. All topic decisions live in these declarations; adding an
ID does not require a Python dispatch branch.

## Training budget

`budget` in `campaign.yaml` chooses one of two sealed budgets; both are in use, and the
mechanics are in [sealing](runtime.md#initial-sealing-and-fixed-budget).

- **`{mode: sealed}`** (every existing campaign): the reference's epochs and optimizer steps are
  fixed, the realized training and validation triplets and study visits are sealed as totals, and
  each experiment gets 115% of the reference's recorded setup, training and prediction time.
  `trainer.max_epochs` is protected.
- **`{mode: range, min_fraction, max_fraction}`**: a candidate's measured training time must lie
  between the two fractions of the reference's `training_seconds` (`min_fraction` below
  `max_fraction`; neither has to bracket 1, so a campaign of cheaper models can sit wholly below the
  reference, as RadImageNet does at 0.284–0.851 of its ReXNet reference). The range is the only
  budget rule: epochs, batch size, gradient accumulation, optimizer steps and the input per study
  (`data.study_triplet_total`, `max_images` in `data.series_types`, slices per input) may change
  within the policy's config permissions; list `trainer.max_epochs`, `trainer.batch_size`,
  `trainer.accumulate_grad_batches` and dependent schedule keys there to let them vary. The
  training and validation cohorts and deterministic ×1 validation stay fixed. Training that runs
  past the upper bound is stopped as `invalid`, and a completion outside the range is `invalid`.
  From the second completed epoch on, the supervisor also projects the total from the mean epoch
  time after the first epoch and the planned `trainer.max_epochs`; a projection more than 2% over
  the upper bound stops the run as `invalid` at once.

A topic the loop chains keeps its predecessor's budget, with the predecessor's absolute bounds.

## Shared invariants and limits

Every campaign still uses the existing training and saved-prediction comparison
scripts, the [shared evaluation and adoption rule](protocol.md#shared-evaluation-and-adoption),
its sealed [training budget](#training-budget), fixed training and validation cohorts, and
per-epoch study-visit accounting. GPU leases, frozen sources,
review, seed handling and detached scheduling remain shared runtime behavior.
Those invariants are not policy switches. A fundamentally different training or
evaluation protocol requires separate harness work, not a permissive topic policy.

Mechanism correctness, shared-versus-per-label semantics, leakage and changes inside
an allowed shared file still need substantive orchestrator review. Static policy
validation is not proof that a candidate respects its scientific contract.
