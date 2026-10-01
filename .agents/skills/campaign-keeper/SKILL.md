---
name: campaign-keeper
description: Run an autoresearch campaign and keep it alive until the user stops it, repairing and resuming it, with the other provider as fallback. Use when the user asks to run, keep running or watch a campaign.
---

# Campaign keeper

Run the campaign with the chosen provider (Codex by default) and keep it running until the user
stops it. If it breaks, fix it and resume it. If the provider is unreachable, switch to the other
one (Codex ↔ Claude) as the last resort.

- **Start:** `rtk proxy uv run --locked just autoresearch start CAMPAIGN --provider PROVIDER` from
  the campaign's root, or `resume` if it already exists. Leave the built-in watchdog off. Save a
  project memory with the campaign, root and provider.
- **Monitor page:** on start or resume, serve the campaign's page on port 8765, always, as the
  systemd user unit `campaign-plots`; one page at a time, so it replaces the previous campaign's.
  Stop the unit and any other process holding the port (`ss -ltnp | grep :8765`), then run
  `systemd-run --user --unit=campaign-plots --working-directory=ROOT --property=Restart=on-failure
  --property=RestartSec=10 /home/claudio/.local/bin/uv run --locked python -m
  scripts.tools.serve_campaign_monitor --campaign CAMPAIGN --directory
  outputs/analysis/campaign-monitor/CAMPAIGN --hosts
  '["100.64.101.0","fd7a:115c:a1e0::f801:6507","127.0.0.1"]' --port 8765` and check that it
  answers. Tailscale Funnel publishes 8765 at https://claudio.tail14383c.ts.net. Leave it running
  when the campaign stops.
- **Watch:** Monitor `bash .agents/skills/campaign-keeper/watch.sh CAMPAIGN ROOT` at the maximum
  timeout. Re-arm it on every expiry. Tell the user only about decisions, repairs and provider
  switches.
- **Repair:** on a pause, a dead supervisor or a stall, read the pause reason, `supervisor.log`
  and the failing job's logs. Fix the cause, then `resume`. Stop the campaign before editing
  `state.json`. Never edit hashed sources while a run trains. Never relax the contract, the
  rules or the limits. Requeue an idea blocked by a harness failure rather than by a result.
- **Ownership:** you own the campaign's health. Stop, patch, requeue and resume on your own,
  without asking the user or handing them commands.
- **Fallback:** the supervisor waits out usage limits on its own. When the login check fails or
  every job has been blocked for over 30 minutes, stop the campaign, set `provider` in
  `state.json` to the other provider, and resume. Switch back once the primary works again.
- **Never** end the campaign (only the user does), and never submit or upload anything.
