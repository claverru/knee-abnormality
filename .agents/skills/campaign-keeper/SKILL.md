---
name: campaign-keeper
description: Run an autoresearch campaign and keep it alive until the user stops it, repairing and resuming it, with the other provider as fallback. Use when the user asks to run, keep running or watch a campaign.
---

# Campaign keeper

Run the campaign with the chosen provider (Codex by default) and keep it running until the user
stops it. You own its health: stop, patch, requeue and resume on your own, without asking the
user or handing them commands. Tell the user only about decisions, repairs and provider
switches.

**Never block the campaign.** The user leaves while it runs. Ask no questions and end no turn
waiting for a reply: take the reversible action the contract points to and report it afterwards.
Anything else is fine as long as the campaign keeps working.

- **Start:** `uv run --locked just autoresearch start CAMPAIGN --provider PROVIDER` from
  the campaign's root, or `resume` if it already exists. Leave the built-in watchdog off. Save a
  project memory with the campaign, root and provider.
- **Monitor page:** on every start or resume, run
  `bash .agents/skills/campaign-keeper/monitor.sh CAMPAIGN ROOT` and check that port 8765
  answers. It registers the campaign with the one monitor site, the systemd user unit
  `campaign-plots` served from the main checkout and reachable only inside the tailnet, and
  starts the site if it is down. The front page lists every live campaign whatever its status,
  stopped ones too, and each has its page under `/CAMPAIGN/`; a campaign registered later
  appears without a restart. A campaign leaves the list when its root is removed. Leave the
  site running when the campaign stops. `monitor_guard.sh`, run as the user unit
  `campaign-plots-guard`, restarts the site from the main checkout when it is down, serves from
  another checkout or stops answering.
- **Messages:** the user writes to you in the message box of your campaign's page, and the watch
  prints each new text once as `MESSAGE from the user`. Each campaign has its own box and file,
  so you receive only your campaign's messages. These are light indications while the user is away,
  such as dropping, adding or reordering a seed: follow them with your own judgment.
- **Watch:** Monitor `bash .agents/skills/campaign-keeper/watch.sh CAMPAIGN ROOT` at the maximum
  timeout, and re-arm it on every expiry.
- **Repair:** fix the cause, then `resume`, in each of these cases:
  - a pause, a dead supervisor or a stall: read the pause reason, `supervisor.log` and the
    failing job's logs;
  - a developer looping on the same failure when you can see the obvious fix: patch the
    candidate yourself instead of waiting;
  - a clearly defective run, such as one much slower than its change justifies: stop it, patch
    the candidate and requeue it.

  A patch stays inside the idea's stated change and the contract. Stop the campaign before
  editing `state.json`. Never edit hashed sources while a run trains. Never relax the contract,
  the rules or the limits. Requeue an idea blocked by a harness failure rather than by a result.
- **Add seeds:** when results or code show a clear gap or fit for the campaign (a defect a run
  exposed, the obvious next step of an accepted idea, a missing cell in a comparison), write
  the seed in a new ideas file in the campaign's directory, list it in the campaign's index and
  force it: `uv run --locked python -m scripts.tools.autoresearch import-ideas CAMPAIGN
  --ideas-file FILE --forced`. Stay inside the campaign's scope and rules.
- **Fallback:** the supervisor waits out usage limits on its own. When the login check fails or
  every job has been blocked for over 30 minutes, stop the campaign, set `provider` in
  `state.json` to the other provider (Codex ↔ Claude) and resume. Switch back once the primary
  works again.
- **Adoption by the contract:** when a campaign's contract sets its own adoption (a forced
  seed, a lineage, flags from a run report), compute what it asks for after each run and adopt
  by hand with `python -m scripts.tools.autoresearch adopt CAMPAIGN --experiment ID`, citing
  the contract. Arm `bash .agents/skills/campaign-keeper/stop_before_end.sh CAMPAIGN ROOT` so
  the next training cannot launch before the adoption.
- **Never** end the campaign (only the user does), and never submit or upload anything.
