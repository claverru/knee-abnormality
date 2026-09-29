import re
from decimal import Decimal
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from knee_abnormality.autoresearch.campaign.definition import PACKAGE
from knee_abnormality.autoresearch.campaign.reports import TopicDraft
from knee_abnormality.autoresearch.supervisor import Supervisor
from knee_abnormality.autoresearch.topics.files import (
    build_campaign,
    campaign_free,
    draft_policy,
)
from knee_abnormality.autoresearch.topics.packet import resolve, scope

URL = re.compile(r"^https?://[^\s/]+\.[^\s/]+\S*$")
TIERS = {"low", "medium", "high"}
STRUCTURAL = 2


def under(item: str, other: str):
    return item == other or item.startswith(f"{other}.")


def source_within(path: str, allowed):
    return path in allowed["editable_files"] or any(
        PurePosixPath(path).is_relative_to(directory)
        for directory in allowed["editable_directories"]
    )


def config_within(item: str, allowed):
    return any(
        under(item, other)
        for other in (
            *allowed["config_keys"],
            *allowed["config_prefixes"],
            *allowed["list_item_fields"],
        )
    )


def scope_within(inner, outer):
    return (
        all(source_within(path, outer) for path in inner["editable_files"])
        and all(
            any(
                PurePosixPath(directory).is_relative_to(other)
                for other in outer["editable_directories"]
            )
            for directory in inner["editable_directories"]
        )
        and all(
            config_within(item, outer)
            for item in (
                *inner["config_keys"],
                *inner["config_prefixes"],
                *inner["list_item_fields"],
            )
        )
    )


def seed_within(seed, allowed):
    return all(source_within(path, allowed) for path in seed["source_paths"]) and all(
        config_within(item, allowed) for item in seed["config_prefixes"]
    )


def matches(cited: float, recorded):
    if isinstance(recorded, bool) or not isinstance(recorded, (int, float)):
        return False
    if isinstance(recorded, int):
        return cited == recorded
    decimals = -Decimal(repr(cited)).normalize().as_tuple().exponent
    return decimals >= 3 and abs(recorded - cited) <= 0.5 * 10.0**-decimals + 1e-12


def grounded(draft: TopicDraft, packet):
    reasons = []
    for item in draft.weaknesses:
        try:
            recorded = resolve(packet, item.pointer)
        except KeyError:
            reasons.append(f"Cited pointer is absent from the packet: {item.pointer}")
            continue
        if not matches(item.value, recorded):
            reasons.append(
                f"Cited value {item.value!r} does not match {item.pointer} = {recorded!r} "
                "(cite at least three decimals)"
            )
    return reasons


def new(draft: TopicDraft, packet, allowed):
    reasons = []
    families = set(draft.mechanism_families)
    seeded = {seed.mechanism_id for seed in draft.seeds}
    if seeded - families:
        reasons.append(f"Seed families are not declared: {sorted(seeded - families)}")
    if families - seeded:
        reasons.append(f"Declared families have no seed: {sorted(families - seeded)}")
    closed = families & set(packet["closed_families"])
    if closed:
        reasons.append(f"Closed families stay closed: {sorted(closed)}")
    for topic in packet["topics"]:
        if topic["id"] == draft.id:
            reasons.append(f"Campaign ID {draft.id} is already a topic")
        if topic["scope"] and scope_within(allowed, topic["scope"]):
            reasons.append(
                f"Scope duplicates {topic['id']}: every declared file and config path is inside it"
            )
        if topic["families"] and families <= set(topic["families"]):
            reasons.append(
                f"Mechanism families duplicate {topic['id']}: all are among its families"
            )
    return reasons


def known_ideas(packet):
    return {
        attempt["id"]
        for campaign in packet["campaigns"]
        for family in campaign["families"]
        for attempt in family["attempts"]
    }


def precedent(value: str, packet, root: Path):
    return (
        value in known_ideas(packet)
        or bool(URL.match(value))
        or (
            not PurePosixPath(value.split("#")[0]).is_absolute()
            and ".." not in PurePosixPath(value.split("#")[0]).parts
            and (root / value.split("#")[0]).exists()
        )
    )


def substantive(draft: TopicDraft, packet, allowed, root: Path):
    reasons = []
    seeds = [seed.model_dump(mode="json") for seed in draft.seeds]
    ids = [seed["id"] for seed in seeds]
    if len(ids) != len(set(ids)):
        reasons.append("Seed IDs must be unique")
    tiers = {seed["complexity"] for seed in seeds}
    if tiers != TIERS:
        reasons.append(
            f"Seeds must span the complexity tiers {sorted(TIERS)}; got {sorted(tiers)}"
        )
    if sum(seed["structural"] for seed in seeds) < STRUCTURAL:
        reasons.append(f"At least {STRUCTURAL} seeds must be structural")
    for seed in seeds:
        name = seed["id"]
        if seed["dependencies"]:
            reasons.append(f"{name}: a new topic has no accepted ideas to depend on")
        for path in seed["source_paths"]:
            if not path.endswith(".py") or not PurePosixPath(path).is_relative_to(
                PACKAGE
            ):
                reasons.append(
                    f"{name}: source path must be Python under {PACKAGE}: {path}"
                )
            elif not source_within(path, allowed):
                reasons.append(
                    f"{name}: source path is outside the topic scope: {path}"
                )
        for item in seed["config_prefixes"]:
            if not config_within(item, allowed):
                reasons.append(
                    f"{name}: config path is outside the topic scope: {item}"
                )
        if not precedent(seed["closest_precedent"].strip(), packet, root):
            reasons.append(
                f"{name}: closest precedent must be a packet idea ID, an existing repository path or a URL"
            )
        if not seed["difference"].strip():
            reasons.append(
                f"{name}: name a concrete difference from the closest precedent"
            )
    missing = [
        seed["id"]
        for seed in packet["seed_pool"]
        if seed_within(seed, allowed) and seed["id"] not in ids
    ]
    if missing:
        reasons.append(f"Seed-pool ideas inside the scope must be taken: {missing}")
    return reasons


def researched(draft: TopicDraft):
    links = {link.rstrip("/") for link in draft.literature}
    reasons = [f"Not a literature URL: {link}" for link in links if not URL.match(link)]
    if len(links) < 2:
        reasons.append("Cite at least two distinct primary-literature links")
    return reasons


def check_draft(draft: TopicDraft, packet, root: Path):
    try:
        policy = draft_policy(draft)
    except (ValidationError, ValueError) as error:
        return [f"Scope is invalid or protected: {error}"]
    allowed = scope(policy.model_dump(mode="json"))
    reasons = [
        *grounded(draft, packet),
        *new(draft, packet, allowed),
        *substantive(draft, packet, allowed, root),
        *researched(draft),
    ]
    if not campaign_free(root, draft.id):
        reasons.append(f"Campaign ID is taken: {draft.id}")
    return reasons


def stage(
    root: Path, topic: Path, target: Path, campaign: str, start: dict, origin: str
):
    definition = build_campaign(topic, target, campaign, start, origin)
    return definition, Supervisor(root, campaign, docs=target).validate()
