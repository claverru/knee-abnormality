#!/bin/bash
set -e
ulimit -n 1048576
P=/home/claudio/.cache/knee-segmenters/venv/bin/python
R=/home/claudio/src/knee-abnormality-sync-free
M=$R/scripts/adhoc/meniscus_masks
N=oai-dess
L=/workspace/outputs/logs/oai_dess_full.log
S=/workspace/outputs/logs/oai_dess_full.state
step() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
step "convert"
$P /root/oai_convert_dess.py $N 60 >> $L 2>&1
step "segment"
for shard in 0 1 2 3; do
  $P /root/oai_segment_shard.py $N $shard 4 8 > /workspace/outputs/logs/oai_dess_shard_$shard.log 2>&1 &
done
wait
step "segmented $(ls /root/.cache/knee-segmenters/$N/nnunet_knee/*.nii.gz | wc -l) volumes"
cd $R
step "export"
$P $M/export_dess.py $N soft-tissue-nnunet-knee-oai1887 30 >> $L 2>&1
step "DONE"
