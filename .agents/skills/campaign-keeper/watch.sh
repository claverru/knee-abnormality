#!/usr/bin/env bash
campaign="$1"
root="$2"
cd "$root" || exit 1
previous=""
same=0
while true; do
  status=$(rtk proxy uv run --locked python -m scripts.tools.autoresearch status "$campaign" 2>&1)
  header=$(echo "$status" | grep -m1 "^$campaign · ")
  warnings=$(echo "$status" | grep -iE "pause|error|provider wait|dead|stopped|blocked|invalid" | grep -v "research waits" | head -5)
  now=$(echo "$status" | sed -n '/^Now/,/^$/p' | sed -n '2,4p' | sed -E 's/[0-9]+m [0-9]+s//g; s/[0-9]+h [0-9]+m//g')
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
  sleep 120
done
