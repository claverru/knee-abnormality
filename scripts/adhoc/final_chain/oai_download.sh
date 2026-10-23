#!/bin/bash
L=/workspace/logs
mkdir -p /workspace/downloads/oai /workspace/data/raw/oai
chmod +x /root/oai_extract.sh
echo "$(date -u +%H:%M:%S) oai download started, $(grep -c '^http' /root/oai_aria2_input.txt) links" >> $L/rebuild.state
aria2c -i /root/oai_aria2_input.txt -d /workspace/downloads/oai -j 32 -x 4 -s 4 --file-allocation=none --auto-file-renaming=false --allow-overwrite=true --max-tries=10 --retry-wait=10 --summary-interval=120 --on-download-complete=/root/oai_extract.sh > $L/oai_download.log 2>&1
rc=$?
sleep 30
echo "$(date -u +%H:%M:%S) oai download EXIT $rc, extracted $(wc -l < $L/oai_extracted.txt), extract errors $(grep -c FAILED $L/oai_extract_errors.log 2>/dev/null || echo 0)" >> $L/rebuild.state
rm -f /root/oai_aria2_input.txt
