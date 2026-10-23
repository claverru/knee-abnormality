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
  family=$1 split=$2 seed=$3 fold=$4
  config=$family-gen3mixed-finetune.yaml
  run=$family-gen3mixed-$split-finetune-fold-$fold
  if [ -f $T/$run/fold_$fold/selected_checkpoint.json ] || ps -eo args | grep -v grep | grep -q -- "--run-id $run --"; then step "$run already finished or running, skipped"; return 0; fi
  if [ -e $T/$run ] && ! ls $T/$run/fold_$fold/best-*.ckpt > /dev/null 2>&1; then rm -rf $T/$run; fi
  args=(-m scripts.pipeline.train configs/$config --run-id $run --validation.split all_5fold_scanner_v2_$split --seed $seed --validation.folds "[$fold]" --initial-weights $T/$family-teachers2-pretrain/fold_0/last.ckpt --no-trainer.persistent-workers)
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

step "GEN3 RESPLIT v3 started: five families, one split each, one seed per run; resnet50 r1 1203997460 643699413 3046000151 2761326745 3154334900; convnext r5 3311724702 765003022 4108126569 206399518 2150820031; mambavision r2 2424155815 558766553 3500031755 1497311460 3863367298; efficientvit r3 1404199329 3988543562 2583390982 3384567468 3942274526; focalnet r4 662050637 246703576 2685161398 982647884 3735886651"
pending=(
"resnet50 r1 1203997460 0" "resnet50 r1 643699413 1" "resnet50 r1 3046000151 2" "resnet50 r1 2761326745 3" "resnet50 r1 3154334900 4"
"convnext r5 3311724702 0" "convnext r5 765003022 1" "convnext r5 4108126569 2" "convnext r5 206399518 3" "convnext r5 2150820031 4"
"mambavision r2 2424155815 0" "mambavision r2 558766553 1" "mambavision r2 3500031755 2" "mambavision r2 1497311460 3" "mambavision r2 3863367298 4"
"efficientvit r3 1404199329 0" "efficientvit r3 3988543562 1" "efficientvit r3 2583390982 2" "efficientvit r3 3384567468 3" "efficientvit r3 3942274526 4"
"focalnet r4 662050637 0" "focalnet r4 246703576 1" "focalnet r4 2685161398 2" "focalnet r4 982647884 3" "focalnet r4 3735886651 4"
)
for item in "${pending[@]}"; do
  read -r family split seed fold <<< "$item"
  slot
  train $family $split $seed $fold &
  sleep 90
done
wait
step "GEN3 RESPLIT DONE"
