#!/usr/bin/env bash
campaign="$1"
root="$2"
cd "$root" || exit 1
family=$(cat "autoresearch/campaigns/$campaign/family-curve.txt" 2>/dev/null)
messages="outputs/autoresearch/$campaign/messages.jsonl"
seen="outputs/autoresearch/$campaign/messages.seen"
previous=""
same=0
while true; do
  status=$(rtk proxy uv run --locked python -m scripts.tools.autoresearch status "$campaign" 2>&1)
  header=$(echo "$status" | grep -m1 "^$campaign · ")
  warnings=$(echo "$status" | grep -iE "pause|error|provider wait|dead|stopped|blocked|invalid" | grep -v "research waits" | head -5)
  now=$(echo "$status" | sed -n '/^Now/,/^$/p' | sed -n '2,4p' | sed -E 's/[0-9]+m [0-9]+s//g; s/[0-9]+h [0-9]+m//g' | tr -s ' ')
  if [ -n "$family" ]; then
    now=$(echo "$now" | awk -v curve="$family" '
      BEGIN { split(curve, offset, " ") }
      match($0, /epoch [0-9]+\//) {
        epoch = substr($0, RSTART + 6, RLENGTH - 7) + 0
        if (match($0, /at this epoch, [+-][0-9.]+/) && epoch in offset) {
          delta = substr($0, RSTART + 15, RLENGTH - 15) + 0
          sub(/\)$/, sprintf("; %+.4f against the family)", delta - offset[epoch] / 10000))
        }
      }
      { print }')
  fi
  results=$(echo "$status" | grep -E "^Results|Incumbent:")
  current="$header | $now | $results | $warnings"
  if [ "$current" != "$previous" ]; then
    echo "$(date +%H:%M) $header"
    echo "$now" | sed 's/^/    /'
    echo "    $results"
    [ -n "$warnings" ] && echo "    WARN: $warnings"
    previous="$current"
    same=0
  else
    same=$((same + 1))
    if [ $((same % 45)) -eq 0 ]; then
      echo "$(date +%H:%M) unchanged for $((same * 2)) min: $header | $(echo "$now" | head -1)"
    fi
  fi
  if ! echo "$header" | grep -q "supervisor alive"; then
    echo "$(date +%H:%M) ALERT supervisor not alive: $header"
  fi
  curl -s -m 60 -o /dev/null "http://127.0.0.1:8765/$campaign/"
  page="outputs/analysis/campaign-monitor/$campaign/index.html"
  if [ ! -f "$page" ] || [ $(( $(date +%s) - $(stat -c %Y "$page") )) -gt 900 ]; then
    echo "$(date +%H:%M) ALERT monitor page of $campaign is missing or older than 15 minutes"
  fi
  for _ in $(seq 12); do
    total=0
    [ -f "$messages" ] && total=$(wc -l < "$messages")
    delivered=$(cat "$seen" 2>/dev/null || echo 0)
    if [ "$total" -gt "$delivered" ]; then
      echo "$(date +%H:%M) MESSAGE from the user on the monitor page:"
      tail -n +$((delivered + 1)) "$messages" | sed 's/^/    /'
      echo "$total" > "$seen"
    fi
    sleep 10
  done
done
