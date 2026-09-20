# Autoresearch guidance

- Read the [protocol](protocol.md) and campaign contract before campaign work.
  Open known paths directly, search for specific terms, and use [INDEX.md](INDEX.md)
  when the destination is unclear.
- No submissions or publication. Only local training and local validation are
  authorized. Never submit to Kaggle, upload models or predictions, publish
  datasets or notebooks, or launch remote evaluation jobs.
- Keep all research documentation and work tracking inside `autoresearch/`, as the
  user requested. Existing repository docs outside this directory are read-only inputs.
- During a campaign, the orchestrator owns research and backlog decisions. Agents
  return structured reports; the supervisor writes canonical records. The restricted
  completion reporter supplies factual summaries or escalates, with no decision authority.
- The developer takes the first ranked idea when assigned. Reserve that entry until
  its training launches; rerank only the unclaimed entries.
- Removing an idea from the backlog never removes its history or experiment evidence.
- Follow the role boundaries and fixed campaign constraints. Campaign agents cannot
  edit these instructions, the contract, or acceptance machinery to relax a limit.
- Use the detached runner described in [runtime operation](runtime.md). Creating or
  editing campaign files does not start an autonomous campaign.
- Keep development one candidate ahead in an isolated workspace. Training uses
  frozen source. Reconcile against a newly accepted incumbent before launch;
  preserve the old attempt and obtain a new review after reconciliation.
