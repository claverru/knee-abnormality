#!/bin/bash
D=/workspace/dispatch; LOG=/workspace/outputs/logs/night6/chain.state
PY=/home/claudio/src/knee-abnormality-sync-free/.venv/bin/python; TREE=/home/claudio/src/knee-abnormality-nextvit
ok() { grep -q "^[a-z0-9]* 0 " $D/done/$1 2>/dev/null; }
all_ok() { for j in "$@"; do ok $j || return 1; done; }
mark() { echo "machine2 0 $(date -u +%Y-%m-%dT%H:%M:%S)" > $D/done/$1; echo "$(date -u +%H:%M:%S) $2" >> $LOG; }
fail_once() { [ -e /root/v3_chain.fail.$1 ] || { touch /root/v3_chain.fail.$1; echo "$(date -u +%H:%M:%S) V3 CHAIN FAILED: $1 ($2)" >> $LOG; }; }
G1=$(for f in convnext resnet50 nextvit; do for k in 0 1 2 3 4; do echo $f-v3-gen1-fold-$k; done; done)
G2=$(for f in convnext resnet50 nextvit; do for k in 0 1 2 3 4; do echo $f-v3-gen2-fold-$k; done; done)
while true; do
  if [ ! -e $D/done/v3-ext-table ] && all_ok $(for r in $G1; do echo ext-$r; done); then
    O=/workspace/data/labels/prepared/rebuild-v3-gen1-x4-tau085-tta-oai-soft-tissue
    if (cd $TREE && $PY /root/build_external_table_rebuild.py /root/teachers-v3.list $O > /root/v3_ext_table.log 2>&1); then mark v3-ext-table "v3 external table built ($O); v3 pretrainings released"; else fail_once ext "see /root/v3_ext_table.log"; fi
  fi
  if [ ! -e $D/done/v3-t2-table ] && all_ok $(for r in $G1; do echo rsna-$r; done); then
    O=/workspace/data/labels/prepared/rebuild-v3-t2-x4-tau085-tta
    if (cd $TREE && $PY /root/build_kaggle_table.py /root/spec-t2-v3.json $O > /root/v3_t2_table.log 2>&1); then mark v3-t2-table "v3 gen-1 teacher table built ($O); share check due"; else fail_once t2 "see /root/v3_t2_table.log"; fi
  fi
  if [ ! -e $D/done/v3-mixed-table ] && all_ok $(for r in $G2; do echo rsna-$r; done); then
    O=/workspace/data/labels/prepared/rebuild-v3-mixed-x4-tau085-tta
    if (cd $TREE && $PY /root/build_kaggle_table.py /root/spec-mixed-v3.json $O > /root/v3_mixed_table.log 2>&1); then mark v3-mixed-table "v3 mixed table built ($O); gen-3 share check due"; else fail_once mixed "see /root/v3_mixed_table.log"; fi
  fi
  sleep 60
done
