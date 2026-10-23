#!/bin/bash
set -e
ulimit -n 1048576
P=/home/claudio/.cache/knee-segmenters/venv/bin/python
R=/home/claudio/src/knee-abnormality-sync-free
M=$R/scripts/adhoc/meniscus_masks
N=oai-dess
S=/workspace/outputs/logs/oai_dess_full.state
step() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
rm -rf /workspace/data/labels/prepared/soft-tissue-nnunet-knee-oai1887
step "segment, 4 shards x 8 workers"
for shard in 0 1 2 3; do
  $P /root/oai_segment_shard.py $N $shard 4 8 > /workspace/outputs/logs/oai_dess_shard_$shard.log 2>&1 &
done
( while pgrep -f oai_segment_shard > /dev/null; do echo "$(date -u +%H:%M:%S) $(free -g | awk '/Mem:/ {print $7}') GB available, $(ls /root/.cache/knee-segmenters/$N/nnunet_knee/*.nii.gz 2>/dev/null | wc -l) done" >> /workspace/outputs/logs/oai_dess_memory.log; sleep 30; done ) &
wait
step "segmented $(ls /root/.cache/knee-segmenters/$N/nnunet_knee/*.nii.gz | wc -l) volumes"
cd $R
step "export"
$P $M/export_dess.py $N soft-tissue-nnunet-knee-oai1887 30 >> /workspace/outputs/logs/oai_dess_full.log 2>&1
step "DONE"
