#!/bin/bash
ulimit -n 1048576
export PYTHONPATH=src:. HF_HUB_OFFLINE=1
unset KNEE_AUTORESEARCH_LIMITS
L=/workspace/outputs/logs/rebuild
T=/workspace/outputs/training
P=/workspace/predictions/rebuild
S=/workspace/outputs/logs/night6/chain.state
PY=/home/claudio/src/knee-abnormality-sync-free/.venv/bin/python
TREE=/home/claudio/src/knee-abnormality-nextvit
step() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
trainings() { ps -eo args | grep -c -E "^[^ ]*python[0-9.]* (-m [s]cripts.pipeline.train|-P [s]cripts/tools/preflight_training.py)"; }
blocked() { now=$(date -u +%Y%m%d%H%M); [ "$now" -ge 202610182045 ] && [ "$now" -lt 202610182305 ]; }
slot() { while [ "$(trainings)" -ge 3 ] || [ "$(free -g | awk '/Mem:/ {print $7}')" -lt 60 ] || blocked; do sleep 20; done; }
runjob() {
  id=$1 fam=$2 cfg=$3 fold=$4 seed=$5 kind=$6 extra=$8
  args=(-m scripts.pipeline.train $cfg --run-id $id --validation.folds "[$fold]" --seed $seed $extra --no-trainer.persistent-workers)
  mkdir -p $L/preflight-$id
  command=$(printf '%s\n' /usr/bin/env -u KNEE_AUTORESEARCH_LIMITS $PY "${args[@]}" | $PY -c 'import json, sys; print(json.dumps(sys.stdin.read().split("\n")[:-1]))')
  [ -e $T/$id ] && ! ls $T/$id/fold_$fold/best-*.ckpt > /dev/null 2>&1 && rm -rf $T/$id
  step "$id started (remote, seed $seed)"
  (cd $TREE && KNEE_AUTORESEARCH_LIMITS=/root/nextvit-limits.json $PY -P scripts/tools/preflight_training.py --config $cfg --fold $fold --output $L/preflight-$id --memory-cap-gib 40 --command "$command" > $L/$id.log 2>&1)
  code=$?
  if [ $code -eq 0 ] && [ "$kind" != pretrain ]; then
    rm -f $T/$id/fold_$fold/last.ckpt
    if [ "$kind" = teacher ] || [ "$kind" = finetune ]; then
      (cd $TREE && nice -n 10 $PY /root/predict_teacher_x4.py $T/$id $fold $P/$id.parquet > $L/predict-$id.log 2>&1) || code=97
    fi
  fi
  /root/finish_job.sh $id remote $code
}
step "REBUILD DISPATCH remote started"
while true; do
  slot
  job=$(/root/claim_job.sh remote)
  [ -z "$job" ] && break
  IFS=$'\t' read -r id fam cfg fold seed kind where extra <<< "$job"
  runjob "$id" "$fam" "$cfg" "$fold" "$seed" "$kind" "$where" "$extra" &
  sleep 60
done
wait
step "REBUILD DISPATCH remote: no unclaimed jobs left"
