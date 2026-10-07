#!/usr/bin/env bash
campaign="$1"
root="$2"
systemctl --user stop campaign-plots 2>/dev/null
systemctl --user reset-failed campaign-plots 2>/dev/null
ss -ltnpH 'sport = :8765' | grep -o 'pid=[0-9]*' | cut -d= -f2 | xargs -r kill
systemd-run --user --unit=campaign-plots --working-directory="$root" \
  --property=Restart=on-failure --property=RestartSec=10 \
  /home/claudio/.local/bin/uv run --locked python -m scripts.tools.serve_campaign_monitor \
  --campaign "$campaign" --directory "outputs/analysis/campaign-monitor/$campaign" \
  --hosts '["100.64.101.0","fd7a:115c:a1e0::f801:6507","127.0.0.1"]' --port 8765
