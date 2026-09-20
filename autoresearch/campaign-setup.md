# Adding a campaign

Add a directory under `autoresearch/campaigns/<id>/` with a contract, configuration,
initial proposals and empty runtime projections. **No runner changes are needed
for a new topic using the existing local training and evaluation protocol.**
The [shared-loss campaign](campaigns/losses/INDEX.md) is a complete example; the
[sampling campaign](campaigns/triplet-sampling/INDEX.md) shows different permissions
using the same runtime.

## Files and startup

1. Write `contract.md`: objective, exclusions, fixed recipe, permitted mechanisms,
   evidence requirements, and the existing acceptance rule. Record prior evidence,
   general-backlog ownership and blocked directions in `prior-work.md`.
2. Write `campaign.yaml` with a unique `id`, recipe, saved control, evidence runs,
   refill thresholds, `artifacts_root: outputs/autoresearch/<id>`, and `policy`.
   The control must match the starting effective recipe apart from name, seed,
   and output path. It supplies the fixed epochs, optimizer steps and 115% runtime
   ceiling; the baseline CPU replay supplies realized triplet counts.
3. Put actionable proposals in `seed-ideas.yaml` using the
   [research-report schema](../src/knee_abnormality/autoresearch/models.py). Keep
   `ideas.yaml` as `revision: 0` plus `ideas: []`, and `backlog.yaml` as
   `revision: 0` plus `queue: []`. Create `findings.md` with an explicit no-results
   status. Startup submits seeds for orchestrator ranking; they are not preapproved.
4. List each file in the campaign's `INDEX.md` and add the directory to the
   [campaign index](campaigns/INDEX.md). Keep campaign work tracking here.
5. Validate without a provider, model call, GPU work, or training:

   ```bash
   rtk proxy uv run --locked python -m scripts.tools.autoresearch validate losses
   ```

   Replace `losses` with the new ID. This checks the definition, policy, contract
   presence, seed schema/IDs, editable files and control/recipe compatibility. It
   does not establish scientific correctness or execute candidate code.
6. The user starts the campaign:

   ```bash
   rtk proxy uv run --locked just autoresearch-start claude losses
   ```

   Either configured subscription provider works. Use the campaign ID with status,
   stop, resume, usage and import-ideas too. Omitting it selects `triplet-sampling`.

## Declarative policy

The strict [policy schema](../src/knee_abnormality/autoresearch/campaigns.py) rejects
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

Name, seed and output path are always logistical differences. List-item permissions
are useful for sampling's `data.series_types: [max_images]`, without allowing plane
or contrast definitions to change. Losses instead permits `trainer.loss` and the
existing class-balance switch, with no input-identity exceptions. Do not add broad
config prefixes or identity exceptions merely to bypass a failed gate.

The runner seals the full policy in runtime state at initialization and includes
its operative permissions in role assignments. A later YAML/state mismatch is an
error. Campaign agents must use a new campaign for a different scope; they cannot
amend a seal. The user explicitly authorized the stopped sampling amendment
recorded below. A candidate cannot edit its policy, contract, role briefs or
acceptance machinery. All topic decisions live in these declarations; adding an
ID does not require a Python dispatch branch.

## Shared invariants and limits

Every campaign still uses the existing training and saved-prediction comparison
scripts, fold 0 of `all_5fold_v3`, deterministic x1 validation, the sealed epoch and
optimizer-step budget, original runtime ceiling, fixed cohort/visit/triplet
accounting, and the same paired-bootstrap verdict. GPU leases, frozen sources,
review, seed handling and detached scheduling remain shared runtime behavior.
Those invariants are not policy switches. A fundamentally different training or
evaluation protocol requires separate harness work, not a permissive topic policy.

Mechanism correctness, shared-versus-per-label semantics, leakage and changes inside
an allowed shared file still need substantive orchestrator review. Static policy
validation is not proof that a candidate respects its scientific contract.

The existing stopped sampling campaign was explicitly migrated at revision 178;
its original state and equivalent policy are preserved in the
[migration record](../outputs/autoresearch/triplet-sampling/decisions/campaign-policy-migration-20260920T100729Z-22dfb2d4/decision.json).
Its incumbent, queue, experiment history and sealed numerical limits did not change.
The runtime does not retain a legacy hardcoded-policy fallback.

At revision 180, the user explicitly authorized a narrow stopped-campaign
amendment for `physical-span-standardized-triplets`: permit `preparation.py`
changes for nonconsecutive-triplet validation. The
[amendment record](../outputs/autoresearch/triplet-sampling/decisions/user-authorize-triplet-preparation-20260920T192746Z-e03bfe70/decision.json)
preserves the previous state, definition and projections. It updates the live
definition and saved policy together, retaining all numerical limits, incumbent,
experiments and blocked follow-ups. This is a recorded user instruction; it grants
no policy-editing authority to campaign agents and adds no automatic migration.
