#!/bin/bash
ulimit -n 1048576
export PYTHONPATH=src:. HF_HUB_OFFLINE=1
unset KNEE_AUTORESEARCH_LIMITS
L=/workspace/outputs/logs/night6
T=/workspace/outputs/training
S=$L/chain.state
PY=/home/claudio/src/knee-abnormality-sync-free/.venv/bin/python
step() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
trainings() { ps -eo args | grep -c -E "^[^ ]*python[0-9.]* (-m [s]cripts.pipeline.train|-P [s]cripts/tools/preflight_training.py)"; }
slot() { while [ "$(trainings)" -ge 3 ] || [ "$(free -g | awk '/Mem:/ {print $7}')" -lt 60 ]; do sleep 30; done; }
tree() {
  case $1 in
    focalnet) echo /home/claudio/src/knee-abnormality-teachers2 ;;
    efficientvit) echo /home/claudio/src/knee-abnormality-effvit ;;
    *) echo /home/claudio/src/knee-abnormality-sync-free ;;
  esac
}

train() {
  family=$1 variant=$2 fold=$3
  config=$family-gen3$variant-finetune.yaml
  run=$family-gen3$variant-finetune-fold-$fold
  if [ -f $T/$run/fold_$fold/selected_checkpoint.json ] || ps -eo args | grep -v grep | grep -q -- "--run-id $run --"; then step "$run already finished or running, skipped"; return 0; fi
  if [ -e $T/$run ] && ! ls $T/$run/fold_$fold/best-*.ckpt > /dev/null 2>&1; then rm -rf $T/$run; fi
  args=(-m scripts.pipeline.train configs/$config --run-id $run --validation.folds "[$fold]" --initial-weights $T/$family-teachers2-pretrain/fold_0/last.ckpt --no-trainer.persistent-workers)
  step "$run training started"
  if [ "$family" = efficientvit ]; then
    mkdir -p $L/preflight-$run
    command=$(printf '%s\n' /usr/bin/env -u KNEE_AUTORESEARCH_LIMITS $PY "${args[@]}" | $PY -c 'import json, sys; print(json.dumps(sys.stdin.read().split("\n")[:-1]))')
    (cd $(tree $family) && KNEE_AUTORESEARCH_LIMITS=/root/effvit-limits.json $PY -P scripts/tools/preflight_training.py --config configs/$config --fold $fold --output $L/preflight-$run --memory-cap-gib 40 --command "$command" > $L/$run.log 2>&1)
  else
    (cd $(tree $family) && $PY "${args[@]}" > $L/$run.log 2>&1)
  fi
  code=$?
  step "$run training EXIT $code"
  [ $code -eq 0 ] && rm -f $T/$run/fold_$fold/last.ckpt
}

step "GEN3 VARIANTS started: share 0.75, teachers = five families' share children (all5) or half children half gen-0 T2 (mixed); order convnext all5, convnext mixed, then resnet50 mambavision efficientvit focalnet mixed, then the same all5"
pending=()
for fold in 0 1 2 3 4; do pending+=("convnext all5 $fold"); done
for fold in 0 1 2 3 4; do pending+=("convnext mixed $fold"); done
for variant in mixed all5; do
  for family in resnet50 mambavision efficientvit focalnet; do
    for fold in 0 1 2 3 4; do pending+=("$family $variant $fold"); done
  done
done
for item in "${pending[@]}"; do
  read -r family variant fold <<< "$item"
  slot
  train $family $variant $fold &
  sleep 90
done
wait
step "GEN3 VARIANTS DONE"
