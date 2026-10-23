#!/bin/bash
path="$3"
[ -f "$path" ] || exit 0
relative="${path#/workspace/downloads/oai/}"
target="/workspace/data/raw/oai/$(dirname "$relative")/$(basename "$relative" .tar.gz)"
mkdir -p "$target"
if tar -xzf "$path" -C "$target" 2>>/workspace/logs/oai_extract_errors.log; then
  rm -f "$path"
  echo "$relative" >> /workspace/logs/oai_extracted.txt
else
  echo "FAILED $relative" >> /workspace/logs/oai_extract_errors.log
fi
