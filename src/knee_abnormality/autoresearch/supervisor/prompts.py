import json

from knee_abnormality.autoresearch.campaign.gates import acceptance_text


def role_prompt(guidance, brief, contract, role, context):
    evidence_guidance = (
        "Use only the supplied completion facts and their original artifacts. "
        "Do not research, review code, change priorities or infer scientific conclusions. "
        "Return a factual report or escalate uncertainty; no decision fields are permitted. "
        if role == "completion_reporter"
        else "Full history and evidence remain available through history_index and artifact paths. "
        "Retrieve relevant originals; absent facts are unknown. "
    )
    return (
        f"{guidance}\n\n{brief}\n\n{contract}\n\n"
        "User-directed seed policy supersedes fresh-seed instructions in archived contracts and findings: "
        "every new candidate uses the current incumbent's recorded training seed, enforced by the supervisor. "
        "Historical runs retain their actual seeds; do not relabel them as matched. "
        "The live state's incumbent and recorded user reference transition supersede starting-reference instructions in archived contracts. "
        "The shared evaluation and adoption rule supersede any restatement in archived contracts and findings: "
        f"{acceptance_text()}. "
        "The 95% and per-label intervals remain diagnostic; do not add other adoption thresholds. "
        "Do not keep pending implementation branches or schedule automatic confirmation runs. "
        "Campaign authorization overrides ordinary user-only training guidance, but ONLY the supervisor launches training. "
        "Do not launch training, use GPUs, spawn subagents, write/run tests, or poll jobs. "
        "NO SUBMISSIONS OR PUBLICATION: only local training and local validation are authorized. "
        "Never submit to Kaggle, upload models/predictions, publish datasets/notebooks, or launch remote jobs. "
        "Reading public research/documentation is allowed. "
        "All canonical docs are read-only to this invocation: return the requested structured report; "
        "the supervisor serializes authorized decisions and restricted factual completion records. "
        "Never change limits, evaluator, labels, folds, harness, or unrelated recipe. "
        "Shell commands must start with rtk. Return only the requested structured report and new evidence. "
        "Narrative length and evidence/concern counts are concision targets, not validation limits. Aim comfortably below the targets; link detailed analysis. "
        "Preserve complete concerns and evidence rather than truncating them or changing decisions to fit a target. "
        "Do not restate standing constraints, unchanged rankings, scheduling, or historical conclusions. "
        f"{evidence_guidance}"
        "The supervisor owns scheduling, reservations, numerical verdicts, and mechanical transitions. "
        "For reviews, inspect the actual diff and critical source directly; concise reporting must not reduce substantive review. "
        "Use the supplied absolute python_executable for static checks; do not install dependencies or create an environment in the workspace. "
        "Workspaces are source copies, not Git branches. Compare with parent_source; never switch or commit the host checkout. "
        "No API fallback. Exit when the assignment is complete.\n\n"
        + json.dumps(context, indent=2)
    )


ORCHESTRATOR_INSTRUCTIONS = (
    "Review evidence and interpret only this event. Use ranked_ids=null when the live queue order and membership stay unchanged. "
    "When evidence requires a queue change, return a complete ranking of the LIVE queue plus accepted proposals minus explicitly parked IDs. "
    "Keep any reserved head first. Target at most five. accepted_proposal_ids may contain IDs in this proposals event or existing proposed registry entries. "
    "Consult proposal_admission: every dependency must be accepted and still present in active_incumbent_ideas. Historical acceptance of replaced behavior is insufficient. "
    "Leave unresolved proposals out of accepted_proposal_ids and ranked_ids; they remain proposed automatically. Ranking them last does not make them eligible. "
    "Retain surplus proposals without queueing weak ideas. Use review only for review events; otherwise none. "
    "For completion interpret the fixed eligible_decision; the supervisor records it directly. Do not recompute it or return a verdict. "
    "For review inspect changes.diff and source, including validation independence from training seed, coverage/exploitation, "
    "leakage, actual mask accounting, preprocessing, and any added preparation cost. Failed gates cannot be approved. "
    "Review the integration plan: preserve unrelated active ideas, verify every declared removal, and explain the result in integration_review for approval. "
    "One concrete repair is allowed; block requires adding the idea to park_ids. "
    "Do not park a developing/prepared idea for unrelated reranking. Empty arrays are valid where inapplicable. "
    "Return new findings, unresolved concerns and evidence references. No scheduling instructions, repeated standing conditions, or unchanged queue explanations."
)


def researcher_instructions(policy_instructions):
    return (
        "Use the compact mechanism registry, then retrieve relevant prior work and original reports through history_index before proposing. "
        f"{policy_instructions} "
        "Reference historic/general-backlog ownership and identify overlaps with ongoing work. Use unique stable lowercase-hyphen IDs. "
        "Fewer ideas is fine; this refill will not immediately retrigger itself. Dependencies must name accepted ideas still active in the incumbent. Historical or replaced methods belong in precedents. "
        "State what the proposal retains, adds and replaces, including how to handle an incumbent change during development. "
        "Investigate primary sources as useful. Keep each proposal concise and put supporting evidence in referenced artifacts."
    )


IMPORT_INSTRUCTIONS = "These are existing repository ideas imported at the user's request. Review and rank eligible ideas before researching more. Preserve provenance and campaign ownership; do not queue blocked or previously rejected work. Deduplicate subsequent proposals against these mechanisms."


COMPLETION_RATIONALE = "Code recorded the fixed numerical result after restricted completion reporting. The incumbent and queue are unchanged; no research direction was closed."


FORCED_SEED_INSTRUCTIONS = (
    " User-ordered seed ideas are queued in their given order, ahead of every other idea, and do not "
    "count toward the five-idea target. Keep their order and positions in any ranking and never park "
    "them, except by blocking a seed at its own implementation review. Admit other proposals only "
    "behind the remaining seeds."
)
