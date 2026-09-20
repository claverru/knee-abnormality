from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Idea(Report):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,79}$")
    mechanism_id: str = Field(max_length=120)
    parent_idea_id: str | None
    title: str = Field(max_length=240)
    opportunity: str = Field(max_length=1200)
    mechanism: str = Field(max_length=1600)
    precedents: list[Annotated[str, Field(max_length=600)]] = Field(max_length=8)
    novelty: str = Field(max_length=800)
    coverage: str = Field(max_length=800)
    exploitation: str = Field(max_length=800)
    budget: str = Field(max_length=1000)
    implementation: str = Field(max_length=1600)
    open_choices: str = Field(max_length=800)
    dependencies: list[str]
    priority_rationale: str = Field(max_length=600)


class ResearchReport(Report):
    ideas: list[Idea] = Field(max_length=5)
    exhausted_directions: str = Field(max_length=1600)


class DeveloperReport(Report):
    status: Literal["prepared", "blocked"]
    summary: str = Field(max_length=1800)
    constraint_evidence: str = Field(max_length=2400)
    blocker: str | None = Field(max_length=1200)


class CompletionReport(Report):
    status: Literal["reported", "escalate"]
    summary: str = Field(min_length=1, max_length=1400)
    concerns: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(
        max_length=6
    )
    evidence_refs: list[Annotated[str, Field(min_length=1, max_length=600)]] = Field(
        min_length=1, max_length=6
    )


class OrchestratorReport(Report):
    accepted_proposal_ids: list[str]
    ranked_ids: list[str] | None = Field(
        description="Null preserves the live queue order; supply a full order only when changing it."
    )
    park_ids: list[str]
    review: Literal["none", "approve", "repair", "block"]
    rationale: str = Field(max_length=2400)
    finding: str = Field(max_length=1800)
    concerns: list[Annotated[str, Field(max_length=300)]] = Field(max_length=6)
    evidence_refs: list[Annotated[str, Field(max_length=600)]] = Field(max_length=8)


REPORTS = {
    "researcher": ResearchReport,
    "developer": DeveloperReport,
    "orchestrator": OrchestratorReport,
    "completion_reporter": CompletionReport,
}
