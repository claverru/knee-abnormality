import re
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from knee_abnormality.autoresearch.campaign.definition import (
    CAMPAIGN_ID,
    BacklogPolicy,
    CampaignPolicy,
    chain_recipe,
)
from knee_abnormality.autoresearch.campaign.reports import (
    Idea,
    IdeaReport,
    TopicDraft,
    TopicSeed,
)
from knee_abnormality.autoresearch.runtime.storage import atomic_write
from knee_abnormality.autoresearch.topics.history import CAMPAIGNS, OUTPUTS

TOPICS = Path("autoresearch/topics")
QUEUE = TOPICS / "queue"
REJECTED = TOPICS / "rejected"
POOL = TOPICS / "seed-pool.yaml"
DRAFT_BACKLOG = BacklogPolicy(
    refill_below=3, refill_to=5, seed_order="ranked", rotate_after=6
)
SEEDS_PER_FILE = 5
TOPIC_FILES = ("topic.yaml", "contract.md", "prior-work.md")


class TopicDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    backlog: BacklogPolicy
    additional_evidence_runs: list[str]
    policy: CampaignPolicy


class PoolSeed(TopicSeed):
    evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def independent(self):
        if self.dependencies:
            raise ValueError("Seed-pool ideas cannot declare dependencies")
        return self


class SeedPool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seeds: list[PoolSeed]

    @model_validator(mode="after")
    def unique(self):
        ids = [seed.id for seed in self.seeds]
        if len(ids) != len(set(ids)):
            raise ValueError("Seed-pool IDs must be unique")
        return self


def load_pool(root: Path):
    return SeedPool.model_validate(yaml.safe_load((root / POOL).read_text()))


def load_topic(directory: Path):
    missing = [name for name in TOPIC_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Topic directory lacks {missing}")
    return TopicDefinition.model_validate(
        yaml.safe_load((directory / "topic.yaml").read_text())
    )


def queued_topics(root: Path):
    queue = root / QUEUE
    return (
        sorted(path for path in queue.iterdir() if path.is_dir())
        if queue.is_dir()
        else []
    )


def campaign_free(root: Path, campaign: str):
    return (
        not (root / CAMPAIGNS / campaign).exists()
        and not (root / OUTPUTS / campaign).exists()
    )


def free_id(root: Path, name: str):
    if not re.match(CAMPAIGN_ID, name):
        raise ValueError(f"Topic directory name is not a campaign ID: {name}")
    candidate, index = name, 1
    while not campaign_free(root, candidate):
        index += 1
        candidate = f"{name}-{index}"
    return candidate


def draft_policy(draft: TopicDraft):
    scope = draft.scope
    return CampaignPolicy(
        editable_files=scope.editable_files,
        editable_directories=scope.editable_directories,
        config_keys=scope.config_keys,
        config_prefixes=scope.config_prefixes,
        list_item_fields={item.key: item.fields for item in scope.list_item_fields},
        input_identity_keys=scope.input_identity_keys,
        require_source_change=True,
        configuration_only_ideas=False,
        developer_instructions=scope.developer_instructions,
        researcher_instructions=scope.researcher_instructions,
    )


def write_seeds(directory: Path, seeds, exhausted: str):
    ideas = [
        Idea.model_validate(seed.model_dump(include=set(Idea.model_fields)))
        for seed in seeds
    ]
    for index in range(0, len(ideas), SEEDS_PER_FILE):
        name = (
            "seed-ideas.yaml"
            if index == 0
            else f"seed-ideas-{index // SEEDS_PER_FILE + 1}.yaml"
        )
        report = IdeaReport(
            ideas=ideas[index : index + SEEDS_PER_FILE], exhausted_directions=exhausted
        )
        atomic_write(
            directory / name,
            yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False, width=100),
        )


def write_draft(draft: TopicDraft, directory: Path):
    definition = TopicDefinition(
        title=draft.title,
        backlog=DRAFT_BACKLOG,
        additional_evidence_runs=[],
        policy=draft_policy(draft),
    )
    directory.mkdir(parents=True)
    atomic_write(
        directory / "topic.yaml",
        yaml.safe_dump(definition.model_dump(mode="json"), sort_keys=False, width=100),
    )
    weaknesses = "\n".join(
        f"- `{item.pointer}` = {item.value!r}: {item.reading}"
        for item in draft.weaknesses
    )
    literature = "\n".join(f"- <{link}>" for link in draft.literature)
    atomic_write(
        directory / "contract.md",
        f"{draft.contract.rstrip()}\n\n## Cited weaknesses\n\nValues from the topic packet, "
        f"checked by code:\n\n{weaknesses}\n\n## Literature\n\n{literature}\n",
    )
    precedents = "\n".join(
        f"| {seed.id} | {seed.complexity} | {'yes' if seed.structural else 'no'} | "
        f"{seed.closest_precedent} | {seed.difference} |"
        for seed in draft.seeds
    )
    atomic_write(
        directory / "prior-work.md",
        f"{draft.prior_work.rstrip()}\n\n## Seed precedents\n\n"
        "| Seed | Complexity | Structural | Closest precedent | Difference |\n"
        f"|---|---|---|---|---|\n{precedents}\n",
    )
    write_seeds(
        directory,
        draft.seeds,
        "Topic seeds drafted by the researcher from the topic packet; research refills follow them.",
    )
    atomic_write(
        directory / "draft.yaml",
        yaml.safe_dump(draft.model_dump(mode="json"), sort_keys=False, width=100),
    )


def starting_section(start: dict):
    chain = start.get("chain")
    lines = [
        "## Starting point",
        "",
        f"- Reference run: `{start['reference_run']}`, recipe `{start['recipe']}`.",
    ]
    if chain:
        lines.append(
            f"- Chained on `{chain['incumbent']}`, the incumbent of "
            f"[{chain['predecessor']}](../{chain['predecessor']}/INDEX.md), with its frozen "
            f"source `{chain['source']}`, seed and fold."
        )
    lines.append(
        "- Evaluation and adoption follow the "
        "[shared rule](../../protocol.md#shared-evaluation-and-adoption)."
    )
    return "\n".join(lines) + "\n"


def campaign_index(campaign: str, title: str, origin: str, files):
    reads = {
        "campaign.yaml": "Resolving the reference, chain, backlog settings and editable scope",
        "topic.yaml": f"Checking the topic definition as taken from its {origin} source",
        "draft.yaml": "Checking the researcher's complete draft: cited weaknesses, literature, families and seed metadata",
        "contract.md": "Checking the objective, exclusions, permitted mechanisms, cited weaknesses and starting point",
        "prior-work.md": "Checking precedents, closed directions and each seed's closest precedent",
        "ideas.yaml": "Recording ideas, their ancestry and completed outcomes",
        "backlog.yaml": "Checking the live queue",
        "findings.md": "Reviewing campaign decisions",
    }
    rows = "\n".join(
        f"| {reads.get(name, 'Checking seed ideas imported at startup')} | [{name}]({name}) |"
        for name in files
    )
    return f"# {title}\n\n{origin.capitalize()} topic `{campaign}`.\n\n| Read when | Document |\n|---|---|\n{rows}\n"


def build_campaign(topic: Path, target: Path, campaign: str, start: dict, origin: str):
    definition = load_topic(topic)
    shutil.copytree(topic, target)
    contract = target / "contract.md"
    atomic_write(
        contract, f"{contract.read_text().rstrip()}\n\n{starting_section(start)}"
    )
    chain = start.get("chain")
    value = {
        "id": campaign,
        "recipe": start["recipe"],
        "reference_run": start["reference_run"],
        "reference_equivalences": {},
        "additional_evidence_runs": definition.additional_evidence_runs,
        "backlog": definition.backlog.model_dump(mode="json"),
        "budget": start["budget"],
        "policy": definition.policy.model_dump(mode="json"),
        **({"chain": chain} if chain else {}),
    }
    if chain and value["recipe"] != chain_recipe(chain["run"]):
        raise ValueError(
            "Chained recipe must be the incumbent's effective configuration"
        )
    atomic_write(
        target / "campaign.yaml", yaml.safe_dump(value, sort_keys=False, width=100)
    )
    atomic_write(target / "ideas.yaml", yaml.safe_dump({"revision": 0, "ideas": []}))
    atomic_write(target / "backlog.yaml", yaml.safe_dump({"revision": 0, "queue": []}))
    atomic_write(target / "findings.md", "# Campaign findings\n\nNo results yet.\n")
    files = sorted(path.name for path in target.iterdir() if path.name != "INDEX.md")
    atomic_write(
        target / "INDEX.md", campaign_index(campaign, definition.title, origin, files)
    )
    return definition


def register_campaign(
    root: Path, campaign: str, title: str, origin: str, start: dict, superseded=False
):
    index = root / CAMPAIGNS / "INDEX.md"
    link = f"]({campaign}/INDEX.md)"
    lines = [line for line in index.read_text().splitlines() if link not in line]
    chain = start.get("chain")
    basis = (
        f"chained on {chain['incumbent']} from {chain['predecessor']}"
        if chain
        else f"on {start['reference_run']}"
    )
    state = (
        "Recovering the superseded topic rotation attempt"
        if superseded
        else "Following the topic"
    )
    row = f"| {state} {campaign} from a {origin} source ({title}; {basis}) | [{title}{link} |"
    position = next(i for i, line in enumerate(lines) if line.startswith("|---")) + 1
    lines.insert(position, row)
    atomic_write(index, "\n".join(lines) + "\n")
