#!/usr/bin/env bash
campaign="$1"
root="$(readlink -f "$2")"
shared="$(dirname "$(readlink -f "$root/outputs")")"
registry="$shared/outputs/autoresearch/monitor/campaigns"
mkdir -p "$registry"
printf '{"campaign": "%s", "root": "%s"}\n' "$campaign" "$root" > "$registry/$campaign.json"
systemctl --user is-active --quiet campaign-plots && exit 0
systemctl --user reset-failed campaign-plots 2>/dev/null
systemd-run --user --unit=campaign-plots --working-directory="$shared" \
  --property=Restart=on-failure --property=RestartSec=10 \
  /home/claudio/.local/bin/uv run --locked python -m scripts.tools.serve_campaign_monitor \
  --hosts '["100.64.101.0","fd7a:115c:a1e0::f801:6507","127.0.0.1"]' --port 8765
