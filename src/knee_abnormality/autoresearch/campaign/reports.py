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


class ResearchReport(Report):
    ideas: list[Idea] = Field(max_length=5)
    exhausted_directions: str = Field(description="Aim for at most 1,600 characters.")


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
        if not self.added_components and not self.removed_components:
            raise ValueError("Integration must add or remove a component")
        if len(self.removed_idea_ids) != len(set(self.removed_idea_ids)):
            raise ValueError("Removed idea IDs must be unique")
        if self.removed_idea_ids and not self.removed_components:
            raise ValueError("Removed idea IDs need their removed components")
        if not self.rationale.strip():
            raise ValueError("Integration needs a rationale")
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

    @model_validator(mode="after")
    def prepared_integration(self):
        if (self.status == "prepared") != (self.integration is not None):
            raise ValueError("Only a prepared report must contain an integration plan")
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
        elif self.integration_review is not None:
            raise ValueError("Only approval can contain an integration review")
        return self


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
