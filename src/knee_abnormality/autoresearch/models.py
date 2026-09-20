from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


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
    implementation: str = Field(description="Aim for at most 1,600 characters.")
    open_choices: str = Field(description="Aim for at most 800 characters.")
    dependencies: list[str] = Field(
        description="Blocking prerequisite idea IDs; each must have a recorded provisional_accept before this proposal can enter the queue."
    )
    priority_rationale: str = Field(description="Aim for at most 600 characters.")


class ResearchReport(Report):
    ideas: list[Idea] = Field(max_length=5)
    exhausted_directions: str = Field(description="Aim for at most 1,600 characters.")


class DeveloperReport(Report):
    status: Literal["prepared", "blocked"]
    summary: str = Field(description="Aim for at most 1,800 characters.")
    constraint_evidence: str = Field(description="Aim for at most 2,400 characters.")
    blocker: str | None = Field(
        description="Aim for at most 1,200 characters; preserve the complete blocker."
    )


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
        description="Proposals to queue now. Every dependency must already have decision=provisional_accept. Omit unresolved proposals from this list and ranked_ids; they remain proposed."
    )
    ranked_ids: list[str] | None = Field(
        description="Null preserves the live queue order; supply a full order only when changing it."
    )
    park_ids: list[str]
    review: Literal["none", "approve", "repair", "block"]
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


REPORTS = {
    "researcher": ResearchReport,
    "developer": DeveloperReport,
    "orchestrator": OrchestratorReport,
    "completion_reporter": CompletionReport,
}
