# Autoresearch index

| Read when | Document |
|---|---|
| Starting here: what autoresearch is and which campaigns are active or retired | [Overview](README.md) |
| Applying the directory's agent and documentation ownership rules | [Agent guidance](AGENTS.md) |
| Checking authority, scope, ownership, backlog ranking policy and what agents receive | [Protocol](protocol.md) |
| Checking the metric, bootstrap and 0.001-plus-80%-interval adoption rule every campaign shares | [Shared evaluation and adoption](protocol.md#shared-evaluation-and-adoption) |
| Checking which seed a candidate trains with, what an adoption makes the incumbent, or where the campaign fold comes from | [Training seed policy](protocol.md#training-seed-policy) |
| Checking which completions the cheap reporter handles and when it escalates | [Restricted completion reporting](protocol.md#restricted-completion-reporting) |
| Following one idea from claim through development, gates, review, reconciliation, launch, comparison and completion | [Idea lifecycle](lifecycle.md) |
| Starting, inspecting, stopping or resuming a campaign; reading results and usage; importing ideas, forced ideas or a manual adoption into a running campaign | [Runtime operation](runtime.md) |
| Moving a queued idea to another place in a running campaign's backlog, or removing it, and what a move or removal is refused for | [Moving and removing ideas](runtime.md#importing-moving-and-removing-ideas-and-manual-adoption) |
| Understanding which failures recover automatically and which pause | [Automatic recovery](runtime.md#running-pausing-and-resuming) |
| Checking how the per-experiment triplet and time budgets are sealed | [Sealing and budget](runtime.md#initial-sealing-and-fixed-budget) |
| Looking up idea, experiment, review, completion or usage record fields | [Records](records.md) |
| Checking the researcher's per-family verdicts (extend, retry, close), open-line follow-ups and the two-miss closure | [Family verdicts](records.md#family-verdicts) |
| Running the training and comparison scripts the supervisor invokes | [Script integrations](scripts.md) |
| Enabling the failure-triggered watchdog for the topic loop or a campaign, or checking what its repair agent may change | [Watchdog](watchdog.md) |
| Adding and validating a campaign through YAML policy, contract and seeds | [Campaign setup](campaign-setup.md) |
| Choosing a campaign's sealed or wall-clock range training budget (free epoch count and input per study) | [Training budget](campaign-setup.md#training-budget) |
| Checking the per-epoch curves and configuration changes a researcher refill receives | [Run curves](records.md#run-curves) |
| Checking measured agent cost, completion-routing data and what to measure next | [Agent usage and efficiency](efficiency.md) |
| Understanding the topic loop: per-campaign streak trigger (`rotate_after`), rotation phases and cancellation, chained campaigns, protected scope, topic packet and naive-topic checks | [Topic rotation](topic-rotation.md) |
| Starting, inspecting, stopping or resuming the topic loop | [Topic loop commands](runtime.md#topic-loop) |
| Dropping a topic into the loop's queue or adding a seed-pool idea | [Topics](topics/INDEX.md) |
| Choosing runner mechanisms from the literature (streak stop, activation signature, bootstrap calibration) and which were rejected for our regime | [Literature mechanisms](literature-mechanisms.md) |
| Checking which ScientistTwo ideas we measured, adopted or ruled out (early-epoch screening, campaign stop rule) | [ScientistTwo lessons](scientisttwo-lessons.md) |
| Continuing the runner implementation or checking remaining work | [Implementation status](status.md) |
| Selecting the provider, model and effort for each role | [Agent assignments](agents.yaml) |
| Reading the brief a role is invoked with | [Agent briefs](roles/INDEX.md) |
| Selecting a campaign or recovering a stopped one | [Campaigns](campaigns/INDEX.md) |
