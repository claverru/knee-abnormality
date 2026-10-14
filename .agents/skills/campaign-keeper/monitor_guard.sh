#!/usr/bin/env bash
code="$(cd "$(dirname "$0")/../../.." && pwd)"
front() {
  curl -s -m 20 -o /dev/null -w '%{http_code}' http://127.0.0.1:8765/
}
while true; do
  healthy=1
  systemctl --user is-active --quiet campaign-plots || healthy=0
  systemctl --user show campaign-plots -p WorkingDirectory | grep -q "=$code$" || healthy=0
  if [ $healthy -eq 1 ] && [ "$(front)" != 200 ]; then
    sleep 20
    [ "$(front)" = 200 ] || healthy=0
  fi
  if [ $healthy -eq 0 ]; then
    echo "$(date '+%F %T') restarting the campaign monitor"
    systemctl --user stop campaign-plots 2>/dev/null
    systemctl --user reset-failed campaign-plots 2>/dev/null
    ss -ltnpH 'sport = :8765' | grep -o 'pid=[0-9]*' | cut -d= -f2 | xargs -r kill
    systemd-run --user --unit=campaign-plots --working-directory="$code" \
      --property=Restart=on-failure --property=RestartSec=10 \
      "$code/.venv/bin/python" -m scripts.tools.serve_campaign_monitor \
      --hosts '["100.64.101.0","fd7a:115c:a1e0::f801:6507","127.0.0.1"]' --port 8765
    sleep 30
  fi
  sleep 30
done
