#!/bin/bash
export PYTHONPATH=src:. HF_HUB_OFFLINE=1
unset KNEE_AUTORESEARCH_LIMITS
L=/workspace/outputs/logs/night6
S=$L/chain.state
PY=/home/claudio/src/knee-abnormality-sync-free/.venv/bin/python
O=/workspace/predictions/pretrain-all
step() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
step "PRETRAIN PREDICTIONS started: six pretrainings on all 4407 Kaggle studies, x4 tau 0.85 mirror TTA"
for item in "convnext sync-free" "resnet50 sync-free" "mambavision sync-free" "focalnet teachers2" "efficientvit effvit" "nextvit nextvit"; do
  read -r family tree <<< "$item"
  [ -f $O/$family.parquet ] && continue
  (cd /home/claudio/src/knee-abnormality-$tree && nice -n 10 $PY /root/predict_pretrain_all.py /workspace/outputs/training/$family-teachers2-pretrain $O/$family.parquet > $L/predict-pretrain-$family.log 2>&1)
  step "pretrain prediction $family EXIT $?"
done
step "PRETRAIN PREDICTIONS DONE"
