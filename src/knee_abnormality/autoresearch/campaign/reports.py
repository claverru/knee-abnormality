from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Idea(Report):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,79}$")
    mechanism_id: str = Field(max_length=120)
    parent_idea_id: str | None
    title: str = Field(description="Aim for at most 240 characters.")
    opportunity: str = Field(description="Aim for at most 1,200 characters.")
    mechanism: str = Field(description="Aim for at most 1,600 characters.")
    precedents: list[str] = Field(
        description="Prefer up to eight concise references, about 600 characters each; retain necessary evidence."
    )
    novelty: str = Field(description="Aim for at most 800 characters.")
    coverage: str = Field(description="Aim for at most 800 characters.")
    exploitation: str = Field(description="Aim for at most 800 characters.")
    budget: str = Field(description="Aim for at most 1,000 characters.")
    implementation: str = Field(
        description="Aim for at most 1,600 characters. State whether the method adds to or replaces existing behavior, which components must be retained or removed, and how to handle a newly accepted incumbent."
    )
    open_choices: str = Field(description="Aim for at most 800 characters.")
    dependencies: list[str] = Field(
        description="Active incumbent idea IDs this mechanism requires; historical methods belong in precedents."
    )
    priority_rationale: str = Field(description="Aim for at most 600 characters.")


class IdeaReport(Report):
    ideas: list[Idea] = Field(max_length=5)
    exhausted_directions: str = Field(description="Aim for at most 1,600 characters.")


class FamilyVerdict(Report):
    mechanism_id: str = Field(max_length=120)
    verdict: Literal["extend", "retry", "close"] = Field(
        description="extend: build on a result that helped; retry: a named change could fix a diagnosed or unactivated failure; close: drop the family."
    )
    reason: str = Field(
        min_length=1,
        description="Aim for at most 600 characters; cite the result and diagnosis behind the verdict.",
    )
    next_step: str | None = Field(
        description="The concrete follow-up for extend or retry, aim for at most 400 characters; null for close."
    )

    @model_validator(mode="after")
    def open_line_step(self):
        if (self.verdict == "close") != (self.next_step is None):
            raise ValueError("extend and retry need a next step; close has none")
        return self


class ResearchIdea(Idea):
    source_change: bool = Field(
        description="True when the idea needs a Python source change; false when it only sets values in configs/campaign.yaml."
    )
    targeted_curve: str = Field(
        min_length=1,
        description="Aim for at most 400 characters; the curve behaviour in run_curves the idea targets (which run, epochs, metric or labels, and what it should change).",
    )


class ResearchReport(IdeaReport):
    ideas: list[ResearchIdea] = Field(max_length=5)
    family_verdicts: list[FamilyVerdict] = Field(
        description="One verdict for every family in new_results; other families only when new evidence changes their verdict."
    )


class IntegrationPlan(Report):
    retained_components: list[Annotated[str, Field(min_length=1)]]
    added_components: list[Annotated[str, Field(min_length=1)]]
    removed_components: list[Annotated[str, Field(min_length=1)]] = Field(
        description="Components present in the supplied implementation parent source but absent from the final candidate, including ablations and cleanup of an earlier failed attempt. Designs borrowed from historical or rejected implementations that never existed in the parent are not removals."
    )
    removed_idea_ids: list[str] = Field(
        description="Accepted ideas intentionally superseded from active_incumbent_ideas; their components must appear in removed_components. Baseline behavior has no idea ID. Unlisted accepted ideas must stay implemented."
    )
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def coherent_changes(self):
        groups = (
            self.retained_components,
            self.added_components,
            self.removed_components,
        )
        names = [name.strip() for group in groups for name in group]
        if not all(names) or len(names) != len(set(names)):
            raise ValueError("Integration components must be nonempty and distinct")
        if len(self.removed_idea_ids) != len(set(self.removed_idea_ids)):
            raise ValueError("Removed idea IDs must be unique")
        if self.removed_idea_ids and not self.removed_components:
            raise ValueError("Removed idea IDs need their removed components")
        if not self.rationale.strip():
            raise ValueError("Integration needs a rationale")
        return self


class ActivationSignature(Report):
    model_config = ConfigDict(extra="forbid", strict=True)

    metric: str = Field(
        pattern=r"^\S+$",
        description="Exact metrics.csv column the candidate logs through Lightning's CSV logger, e.g. token_dropout_masked_tokens; an epoch-level aggregate is always written.",
    )
    comparison: Literal[">", ">="]
    threshold: float = Field(allow_inf_nan=False)
    epoch: int = Field(
        ge=0,
        description="Epoch after which the check runs; 0 unless the mechanism is gated by a warm-up.",
    )
    rationale: str = Field(
        min_length=1,
        description="Aim for at most 400 characters; why this value proves the claimed mechanism ran.",
    )

    @model_validator(mode="after")
    def mechanism_metric(self):
        if self.metric in {"epoch", "step"}:
            raise ValueError("The activation metric must measure the mechanism")
        if not self.rationale.strip():
            raise ValueError("The activation signature needs a rationale")
        return self


class DeveloperReport(Report):
    status: Literal["prepared", "blocked"]
    summary: str = Field(description="Aim for at most 1,800 characters.")
    constraint_evidence: str = Field(description="Aim for at most 2,400 characters.")
    blocker: str | None = Field(
        description="Aim for at most 1,200 characters; preserve the complete blocker."
    )
    integration: IntegrationPlan | None = Field(
        description="Required when prepared, null when blocked. Relative to the supplied implementation parent, name topic-relevant components retained, added and removed, using behavior and source/config references. A candidate replaces parent behavior exactly when removed_components is nonempty."
    )
    activation: ActivationSignature | None = Field(
        description="Required when prepared, null when blocked. Passes when the largest finite value of the column over all rows logged through the named epoch satisfies the comparison; a column never logged by then fails, stops the run as invalid and returns it as a repair."
    )

    @model_validator(mode="after")
    def prepared_integration(self):
        if (self.status == "prepared") != (self.integration is not None):
            raise ValueError("Only a prepared report must contain an integration plan")
        if (self.status == "prepared") != (self.activation is not None):
            raise ValueError(
                "Only a prepared report must contain an activation signature"
            )
        return self


class CompletionReport(Report):
    status: Literal["reported", "escalate"]
    summary: str = Field(
        min_length=1, description="Aim for 100–180 words, around 1,400 characters."
    )
    concerns: list[Annotated[str, Field(min_length=1)]] = Field(
        description="Prefer up to six concise concerns, about 300 characters each; never omit an unresolved issue to fit a target."
    )
    evidence_refs: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1,
        description="Prefer up to six concise references, about 600 characters each; retain necessary evidence.",
    )


class OrchestratorReport(Report):
    accepted_proposal_ids: list[str] = Field(
        description="Proposals to queue now; only those whose dependencies proposal_admission reports satisfied."
    )
    ranked_ids: list[str] | None = Field(
        description="Null preserves the live queue order; supply a full order only when changing it."
    )
    park_ids: list[str]
    review: Literal["none", "approve", "repair", "block"]
    integration_review: str | None = Field(
        description="For approve, explain how the inspected diff implements the declared retained/added/removed components against the bound parent and reviewed evidence. Null for every other review value. A clean merge is not proof of semantic integration."
    )
    activation_review: str | None = Field(
        description="For approve, explain how the inspected logging code makes the bound activation signature measure the declared mechanism, and why its threshold and epoch separate a run where it acted from one where it did not. Null for every other review value."
    )
    rationale: str = Field(
        description="Aim for at most 2,400 characters; explain only changed decisions."
    )
    finding: str = Field(
        description="Aim for at most 1,800 characters; retain new findings and their limits."
    )
    concerns: list[str] = Field(
        description="Prefer up to six concise concerns, about 300 characters each; never omit an unresolved issue to fit a target."
    )
    evidence_refs: list[str] = Field(
        description="Prefer up to eight concise references, about 600 characters each; retain necessary evidence."
    )

    @model_validator(mode="after")
    def approved_integration(self):
        if self.review == "approve":
            if not self.integration_review or not self.integration_review.strip():
                raise ValueError("Approval requires an explicit integration review")
            if not self.activation_review or not self.activation_review.strip():
                raise ValueError("Approval requires an explicit activation review")
        elif self.integration_review is not None or self.activation_review is not None:
            raise ValueError(
                "Only approval can contain an integration or activation review"
            )
        return self


class TopicSeed(Idea):
    complexity: Literal["low", "medium", "high"] = Field(
        description="Implementation and conceptual complexity tier."
    )
    structural: bool = Field(
        description="True when the seed adds or replaces a model, loss, target or data-flow component instead of changing values of existing ones."
    )
    source_paths: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1,
        description="Python files under src/knee_abnormality/ the seed changes or adds; each must be inside the topic's editable scope. A seed without a source change is configuration-only and rejected.",
    )
    config_prefixes: list[Annotated[str, Field(min_length=1)]] = Field(
        description="Dotted configuration subtrees the seed sets; each must be inside the topic's config keys or prefixes."
    )
    closest_precedent: str = Field(
        min_length=1,
        description="Exactly one of: an idea ID from the packet's family ledger, a repository-relative path that exists (an optional #anchor is allowed), or an http(s) URL.",
    )
    difference: str = Field(
        min_length=1,
        description="Aim for at most 400 characters; the concrete change from the closest precedent.",
    )


class CitedValue(Report):
    pointer: str = Field(
        pattern=r"^/",
        description="JSON pointer into the topic packet, e.g. /incumbent/per_label/PF OA; keys are separated by /.",
    )
    value: float = Field(
        allow_inf_nan=False,
        description="The value at the pointer, copied with at least three decimals; integers exactly.",
    )
    reading: str = Field(
        min_length=1,
        description="Aim for at most 300 characters; the weakness this value shows.",
    )


class ListItemFields(Report):
    key: str = Field(min_length=1)
    fields: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)


class TopicScope(Report):
    editable_files: list[str]
    editable_directories: list[str]
    config_keys: list[str]
    config_prefixes: list[str]
    list_item_fields: list[ListItemFields]
    input_identity_keys: list[str]
    developer_instructions: str = Field(min_length=1)
    researcher_instructions: str = Field(min_length=1)


class TopicDraft(Report):
    id: str = Field(
        pattern=r"^[a-z][a-z0-9-]{2,59}$",
        description="New campaign ID, unused under autoresearch/campaigns and outputs/autoresearch.",
    )
    title: str = Field(min_length=1, description="Aim for at most 160 characters.")
    weaknesses: list[CitedValue] = Field(
        min_length=1,
        description="Weaknesses of the incumbent cited by value from the topic packet; code resolves each pointer.",
    )
    literature: list[Annotated[str, Field(pattern=r"^https?://\S+$")]] = Field(
        min_length=2,
        description="Primary-literature links on the cited weakness, at least two distinct.",
    )
    mechanism_families: list[Annotated[str, Field(min_length=1, max_length=120)]] = (
        Field(
            min_length=1,
            description="Mechanism families the topic explores; every seed's mechanism_id is one of them.",
        )
    )
    scope: TopicScope
    contract: str = Field(
        min_length=1,
        description="Markdown campaign contract: objective, exclusions, permitted mechanisms and evidence requirements. Code appends the cited weaknesses, literature and starting point.",
    )
    prior_work: str = Field(
        min_length=1,
        description="Markdown prior work: precedent table of measured runs in the topic (mechanism, setting, run, macro delta, source) and closed directions.",
    )
    seeds: list[TopicSeed] = Field(
        min_length=5,
        max_length=10,
        description="At least five seed ideas across the three complexity tiers, at least two structural, including every untaken seed-pool entry inside the scope.",
    )


class TopicDraftReport(Report):
    drafts: list[TopicDraft] = Field(min_length=2, max_length=3)


class TopicPick(Report):
    draft_id: str = Field(min_length=1)
    rationale: str = Field(
        min_length=1,
        description="Aim for at most 1,600 characters; why this topic has the best expected yield for the incumbent.",
    )
    concerns: list[str] = Field(
        description="Prefer up to six concise concerns, about 300 characters each."
    )


class WatchdogReport(Report):
    status: Literal["ready", "blocked"]
    summary: str
    changed_files: list[str]


REPORTS = {
    "researcher": ResearchReport,
    "developer": DeveloperReport,
    "orchestrator": OrchestratorReport,
    "completion_reporter": CompletionReport,
}
TOPIC_REPORTS = {"researcher": TopicDraftReport, "orchestrator": TopicPick}
