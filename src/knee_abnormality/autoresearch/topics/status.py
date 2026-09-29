import time

from knee_abnormality.autoresearch.topics.files import load_pool, queued_topics
from knee_abnormality.autoresearch.topics.history import (
    campaign_state,
    outcome,
    seeded_ideas,
)
from knee_abnormality.autoresearch.topics.loop import TopicLoop


def clock(value):
    return time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(value)) if value else "-"


def loop_status(loop: TopicLoop):
    state = loop.state
    rotation = state["rotation"] if state else None
    return {
        "status": state["status"] if state else "not_started",
        "loop_alive": loop.alive(),
        "provider": state["provider"] if state else None,
        "current": state["current"] if state else None,
        "rotation": rotation and rotation["phase"],
        "successor": rotation and (rotation.get("successor") or {}).get("campaign"),
        "attention": state["attention"] if state else None,
        "artifacts": str(loop.output),
    }


def loop_summary(loop: TopicLoop):
    current = loop_status(loop)
    alive = "loop alive" if current["loop_alive"] else "loop not running"
    lines = [
        f"topic loop · {current['provider'] or 'no provider'} · "
        + f"{current['status'].upper()} · {alive}"
    ]
    state = loop.state
    if state is None:
        return "\n".join(lines)
    warnings = []
    if state.get("error"):
        warnings.append(
            f"Error at {clock(state['error']['at'])}: {state['error']['message']}"
        )
    if state.get("reason"):
        warnings.append(f"Reason: {state['reason']}")
    if state["attention"]:
        warnings.append(f"Attention: {state['attention']}")
    if (loop.output / "stop.json").exists() and state["status"] != "stopped":
        warnings.append(
            "Stop requested; waiting for the topics and agent jobs to stop."
        )
    if warnings:
        lines.extend(["", *warnings])
    root = loop.root
    campaign = state["current"]
    if campaign:
        campaign_record = campaign_state(root, campaign)
        record = loop.topic(campaign)
        facts = (
            outcome(
                campaign_record, seeded_ideas(root / "outputs/autoresearch" / campaign)
            )
            if campaign_record
            else None
        )
        lines.extend(
            [
                "",
                (
                    f"Current topic  {campaign} ({record['origin']}) · campaign "
                    f"{(campaign_record or {}).get('status', 'not_started').upper()}"
                ),
            ]
        )
        if facts:
            lines.append(
                f"  researcher streak {facts['streak']}/{campaign_record['rotate_after']} · "
                f"{facts['decided']} decided, "
                f"{facts['adoptions']} adopted, {facts['invalid']} invalid"
            )
    else:
        lines.extend(["", f"Current topic  none; first topic on {state['reference']}"])
    rotation = state["rotation"]
    if rotation:
        successor = (rotation.get("successor") or {}).get("campaign") or "-"
        lines.extend(
            [
                "",
                (
                    f"Rotation  {rotation['phase']} since {clock(rotation['started_at'])} · "
                    f"{rotation['predecessor'] or 'start'} → {successor} · "
                    f"{len(rotation['attempts'])} draft attempt(s), "
                    f"{rotation['rejections']} consecutive failure(s)"
                ),
            ]
        )
        if rotation["previous_rejection"]:
            lines.append(f"  last failure: {rotation['previous_rejection'][:600]}")
        for item in rotation["queue_rejections"]:
            lines.append(
                f"  queue topic {item['topic']} rejected: {item['reason'][:300]}"
            )
    lines.extend(["", "Topics"])
    width = max([len(item["campaign"]) for item in state["topics"]] + [8])
    lines.append(
        f"  {'campaign':<{width}}  {'origin':<10}  {'status':<10}  decided  adopted  streak at rotation"
    )
    for item in state["topics"]:
        facts = item["outcome"] or {}
        lines.append(
            f"  {item['campaign']:<{width}}  {item['origin']:<10}  {item['status']:<10}  "
            f"{facts.get('decided', '-')!s:>7}  {facts.get('adoptions', '-')!s:>7}  "
            f"{item['streak_at_rotation'] if item['streak_at_rotation'] is not None else '-'}"
        )
    queue = [path.name for path in queued_topics(root)]
    lines.extend(
        [
            "",
            f"Queue  {len(queue)} topic(s) waiting"
            + (f": {', '.join(queue)}" if queue else ""),
        ]
    )
    pool = [seed.id for seed in load_pool(root).seeds]
    untaken = [seed for seed in pool if seed not in state["pool_taken"]]
    lines.append(
        f"Seed pool  {len(untaken)} untaken of {len(pool)}"
        + (f": {', '.join(untaken)}" if untaken else "")
    )
    lines.extend(["", f"Artifacts: {loop.output}"])
    return "\n".join(lines)
