#!/bin/bash
ulimit -n 1048576
export PYTHONPATH=src:. HF_HUB_OFFLINE=1
unset KNEE_AUTORESEARCH_LIMITS
L=/workspace/outputs/logs/rebuild
T=/workspace/outputs/training
P=/workspace/predictions/rebuild
E=/workspace/predictions/rebuild-external
S=/workspace/outputs/logs/night6/chain.state
PY=/home/claudio/src/knee-abnormality-sync-free/.venv/bin/python
TREE=/home/claudio/src/knee-abnormality-nextvit
mkdir -p $L $P $E
step() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
blocked() { [ -f /workspace/dispatch/maintenance-done ] && return 1; now=$(date -u +%Y%m%d%H%M); [ "$now" -ge 202610182045 ] && [ "$now" -lt 202610182305 ]; }
slot() { while [ "$(/root/running_jobs.sh)" -ge 3 ] || [ "$(free -g | awk '/Mem:/ {print $7}')" -lt 60 ] || blocked; do sleep 20; done; }
runjob() {
  id=$1 fam=$2 cfg=$3 fold=$4 seed=$5 kind=$6 extra=$8
  extra=${extra//@T@/$T}
  step "$id started (machine)"
  case $kind in
    teacher|finetune|pretrain)
      args=(-m scripts.pipeline.train $cfg --run-id $id --validation.folds "[$fold]" --seed $seed $extra --no-trainer.persistent-workers)
      mkdir -p $L/preflight-$id
      command=$(printf '%s\n' /usr/bin/env -u KNEE_AUTORESEARCH_LIMITS $PY "${args[@]}" | $PY -c 'import json, sys; print(json.dumps(sys.stdin.read().split("\n")[:-1]))')
      [ -e $T/$id ] && ! ls $T/$id/fold_$fold/best-*.ckpt > /dev/null 2>&1 && rm -rf $T/$id
      (cd $TREE && KNEE_AUTORESEARCH_LIMITS=/root/nextvit-limits.json $PY -P scripts/tools/preflight_training.py --config $cfg --fold $fold --output $L/preflight-$id --memory-cap-gib 40 --command "$command" > $L/$id.log 2>&1)
      code=$?
      [ $code -eq 0 ] && [ "$kind" != pretrain ] && rm -f $T/$id/fold_$fold/last.ckpt ;;
    rsna)
      if [ -f $P/$cfg.parquet ]; then code=0; else (cd $TREE && nice -n 10 $PY /root/predict_teacher_x4.py $T/$cfg $fold $P/$cfg.parquet > $L/predict-$cfg.log 2>&1); code=$?; fi ;;
    external)
      ck=$(ls $T/$cfg/fold_$fold/best-*.ckpt | head -1)
      (cd $TREE && nice -n 10 $PY /root/external_predict_rebuild.py "$ck" $E/$cfg 8 > $L/external-$cfg.log 2>&1); code=$? ;;
    *) code=99 ;;
  esac
  /root/finish_job.sh $id machine2 $code
}
step "REBUILD DISPATCH machine2 started"
while true; do
  slot
  job=$(/root/claim_job.sh machine2)
  if [ -z "$job" ]; then sleep 60; continue; fi
  id=$(cut -f1 <<< "$job"); fam=$(cut -f2 <<< "$job"); cfg=$(cut -f3 <<< "$job"); fold=$(cut -f4 <<< "$job"); seed=$(cut -f5 <<< "$job"); kind=$(cut -f6 <<< "$job"); where=$(cut -f7 <<< "$job"); extra=$(cut -f8 <<< "$job"); after=$(cut -f9 <<< "$job")
  runjob "$id" "$fam" "$cfg" "$fold" "$seed" "$kind" "$where" "$extra" &
  sleep 45
done
