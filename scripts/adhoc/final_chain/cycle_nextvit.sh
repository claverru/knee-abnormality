#!/bin/bash
ulimit -n 1048576
export PYTHONPATH=src:. HF_HUB_OFFLINE=1
unset KNEE_AUTORESEARCH_LIMITS
L=/workspace/outputs/logs/night6
T=/workspace/outputs/training
S=$L/chain.state
PY=/home/claudio/src/knee-abnormality-sync-free/.venv/bin/python
TREE=/home/claudio/src/knee-abnormality-nextvit
PRE=$T/nextvit-teachers2-pretrain/fold_0/last.ckpt
step() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
trainings() { ps -eo args | grep -c -E "^[^ ]*python[0-9.]* (-m [s]cripts.pipeline.train|-P [s]cripts/tools/preflight_training.py)"; }
slot() { while [ "$(trainings)" -ge 3 ] || [ "$(free -g | awk '/Mem:/ {print $7}')" -lt 60 ]; do sleep 2; done; }

run() {
  run=$1 config=$2 fold=$3
  shift 3
  args=(-m scripts.pipeline.train configs/$config --run-id $run --validation.folds "[$fold]" "$@" --no-trainer.persistent-workers)
  mkdir -p $L/preflight-$run
  command=$(printf '%s\n' /usr/bin/env -u KNEE_AUTORESEARCH_LIMITS $PY "${args[@]}" | $PY -c 'import json, sys; print(json.dumps(sys.stdin.read().split("\n")[:-1]))')
  step "$run training started"
  (cd $TREE && KNEE_AUTORESEARCH_LIMITS=/root/nextvit-limits.json $PY -P scripts/tools/preflight_training.py --config configs/$config --fold $fold --output $L/preflight-$run --memory-cap-gib 40 --command "$command" > $L/$run.log 2>&1)
  code=$?
  step "$run training EXIT $code"
  return $code
}

finetune() {
  fold=$1 seed=$2
  r=nextvit-gen3mixed-r6-finetune-fold-$fold
  if [ -f $T/$r/fold_$fold/selected_checkpoint.json ]; then step "$r already finished, skipped"; return 0; fi
  [ -e $T/$r ] && rm -rf $T/$r
  run $r nextvit-gen3mixed-finetune.yaml $fold --validation.split all_5fold_scanner_v2_r6 --seed $seed --initial-weights $PRE && rm -f $T/$r/fold_$fold/last.ckpt
}

step "NEXTVIT started: pretraining seed 4213577004 (old split fold 0, external only, 10 epochs), then gen3 mixed fine-tunes on all_5fold_scanner_v2_r6 (split seed 1624582548), seeds 4081590439 3631603500 763392539 14540636 952220979"
if [ ! -f $PRE ]; then
  slot
  run nextvit-teachers2-pretrain nextvit-teachers2-pretrain.yaml 0 --seed 4213577004 || { step "NEXTVIT STOPPED: pretraining failed"; exit 1; }
fi
for item in "0 4081590439" "1 3631603500" "2 763392539" "3 14540636" "4 952220979"; do
  read -r fold seed <<< "$item"
  slot
  finetune $fold $seed &
  sleep 90
done
wait
step "NEXTVIT DONE"
