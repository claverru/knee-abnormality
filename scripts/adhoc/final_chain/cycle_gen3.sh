#!/bin/bash
RECIPE=$1
[ "$RECIPE" = finetune ] || [ "$RECIPE" = joint ] || { echo "usage: cycle.sh finetune|joint"; exit 2; }
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
ready() { [ -f $T/$1-teachers2-pretrain/fold_0/last.ckpt ] && grep -q "$1-teachers2-pretrain training EXIT 0" $S /workspace/outputs/logs/night5/chain.state; }

train() {
  family=$1 fold=$2
  run=$family-gen3-$RECIPE-fold-$fold
  if [ -f $T/$run/fold_$fold/selected_checkpoint.json ] || ps -eo args | grep -v grep | grep -q -- "--run-id $run --"; then step "$run already finished or running, skipped"; return 0; fi
  if [ -e $T/$run ] && ! ls $T/$run/fold_$fold/best-*.ckpt > /dev/null 2>&1; then rm -rf $T/$run; fi
  args=(-m scripts.pipeline.train configs/$family-gen3-$RECIPE.yaml --run-id $run --validation.folds "[$fold]" --initial-weights $T/$family-teachers2-pretrain/fold_0/last.ckpt --no-trainer.persistent-workers)
  step "$run training started"
  if [ "$family" = efficientvit ]; then
    mkdir -p $L/preflight-$run
    command=$(printf '%s\n' /usr/bin/env -u KNEE_AUTORESEARCH_LIMITS $PY "${args[@]}" | $PY -c 'import json, sys; print(json.dumps(sys.stdin.read().split("\n")[:-1]))')
    (cd $(tree $family) && KNEE_AUTORESEARCH_LIMITS=/root/effvit-limits.json $PY -P scripts/tools/preflight_training.py --config configs/$family-gen3-$RECIPE.yaml --fold $fold --output $L/preflight-$run --memory-cap-gib 40 --command "$command" > $L/$run.log 2>&1)
  else
    (cd $(tree $family) && $PY "${args[@]}" > $L/$run.log 2>&1)
  fi
  code=$?
  step "$run training EXIT $code"
  [ $code -eq 0 ] && rm -f $T/$run/fold_$fold/last.ckpt
}

step "GEN3 cycle started: recipe $RECIPE + teacher table gen3-a (M+R share children) at share 0.80, 5 families x 5 folds from the teachers2 pretrainings, order resnet50 mambavision efficientvit convnext focalnet"
pending=()
for family in resnet50 mambavision efficientvit convnext focalnet; do for fold in 0 1 2 3 4; do pending+=("$family $fold"); done; done
while [ ${#pending[@]} -gt 0 ]; do
  launched=0
  for index in "${!pending[@]}"; do
    read -r family fold <<< "${pending[$index]}"
    if ready $family; then
      slot
      train $family $fold &
      unset 'pending[index]'
      pending=("${pending[@]}")
      launched=1
      sleep 90
      break
    fi
  done
  if [ $launched -eq 0 ]; then
    remaining=()
    for item in "${pending[@]}"; do
      read -r family fold <<< "$item"
      if grep -h "$family-teachers2-pretrain training" $S | tail -1 | grep -q "EXIT [1-9]"; then
        step "pretraining of $family failed; $family fold $fold skipped"
      else
        remaining+=("$item")
      fi
    done
    pending=("${remaining[@]}")
    sleep 60
  fi
done
wait
step "GEN3 CYCLE DONE"
