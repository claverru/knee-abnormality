import json

from knee_abnormality.autoresearch.campaign.rules import acceptance_text


def shared_rules():
    return f"""# Rules for every autoresearch agent

The repository guidance above is written for interactive sessions; where it conflicts with
these rules, these rules win.

- **Authority.** Return one structured report. The supervisor applies it and owns
  scheduling, reservations, launches, numerical verdicts, records and documentation.
  Campaign state, the idea registry, the backlog and all documentation are read-only to you.
- **Forbidden.** Launching training or using GPUs, spawning agents, writing or running
  tests, polling jobs, submitting to Kaggle, uploading models or predictions, publishing
  anything, and launching remote jobs. Reading public research is allowed.
- **Fixed.** Limits, the evaluator, labels, folds, the harness, and recipe fields outside
  the campaign policy.
- **Evaluation.** The adoption rule: {acceptance_text()}. The 95% and per-label intervals are exploratory:
  unadjusted, blind to training-seed variation, unable to change the macro verdict, and
  no basis for per-label objectives. Add no other threshold, pending branch or
  confirmation run.
- **Seeds and references.** Every candidate trains at the current incumbent's recorded
  seed, set by the supervisor; historical runs keep their actual seeds. The assignment's
  incumbent and seed replace the starting run and seed named in the campaign contract.
  The live incumbent and these rules supersede anything contrary in archived contracts
  and findings.
- **Evidence.** Facts absent from the assignment and the artifacts it links are unknown;
  retrieve originals when a decision depends on them. Separate measurements from
  hypotheses.
- **Reporting.** Schema field descriptions set concision targets, not limits; never drop a
  material concern or evidence to meet them. Link detailed analysis instead of copying
  it, and report only new findings and changed decisions.
- **Environment.** Shell commands start with rtk. Use the supplied `python_executable` and
  install nothing. Workspaces are source copies, not Git branches: compare with
  `parent_source` and never switch or commit the host checkout. Use subscription
  authentication only, with no API fallback. Exit when the assignment is complete."""


def role_prompt(guidance, brief, contract, context):
    return (
        f"{guidance}\n\n{shared_rules()}\n\n{brief}\n\n{contract}\n\n"
        f"# Assignment\n\n{json.dumps(context, indent=2)}"
    )


IMPORT_INSTRUCTIONS = "These are existing repository ideas imported at the user's request. Review and rank eligible ideas before researching more. Preserve provenance and campaign ownership; do not queue blocked or previously rejected work. Deduplicate subsequent proposals against these mechanisms."


COMPLETION_RATIONALE = "Code recorded the fixed numerical result after restricted completion reporting. The incumbent and queue are unchanged; no research direction was closed."


FORCED_SEED_INSTRUCTIONS = (
    "User-ordered seed ideas are queued in their given order, ahead of every other idea, and do not "
    "count toward the five-idea target. Keep their order and positions in any ranking and never park "
    "them, except by blocking a seed at its own implementation review. Admit other proposals only "
    "behind the remaining seeds."
)
