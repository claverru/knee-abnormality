#!/bin/bash
L=/workspace/logs
part=$1
echo "$(date -u +%H:%M:%S) oai fetch $part started" >> $L/rebuild.state
/root/layoutenv/bin/python /root/oai_fetch.py /root/oai_$part.txt 32 >> $L/oai_fetch.log 2>&1
echo "$(date -u +%H:%M:%S) oai fetch $part EXIT $? $(tail -n 1 $L/oai_fetch.log)" >> $L/rebuild.state
rm -f /root/oai_$part.txt
