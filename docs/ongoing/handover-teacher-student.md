# Handover: teacher and student generations (2026-10-17 evening)

Written for the agent that takes over the keeper role of this line from session `knee-abnormality-4c`.
Times are CEST (UTC+2); the competition ends 2026-10-22 23:59 UTC; five submissions a day, reset at
02:00 CEST.

## What we are doing

Each generation of models learns its RSNA targets from the previous generation's out-of-fold
predictions mixed with the silver labels at a teacher share, then becomes the teacher of the next.

| Generation | Teachers | Kaggle targets | Start weights | Public |
|---|---|---|---|---|
| Gen 0 silver teachers | none | silver | ImageNet / RadImageNet | S158 0.957 (ConvNeXt fold 0) |
| Share children (`*-share075-finetune-fold-k`) | gen-0 ResNet-50 + ConvNeXt, table T2 (`teachers2-kaggle-x4-tau085-tta`) | 0.25 silver + 0.75 T2 | round-2 pretrainings (`*-teachers2-pretrain`) | **S170 0.965, ConvNeXt × 5, best** |
| Next generation, training overnight | the five families' share children | 0.25 silver + 0.75 table (two variants, below) | same round-2 pretrainings | S174 pending |

Every child is a Kaggle-only fine-tune (no external data) of its family's round-2 pretraining. The
joint stage with fastMRI/OAI adds nothing once the share is in (S168 sorted below S167).

The two variants of the next generation, both at share 0.75:

- **all5**, table `data/labels/prepared/gen3-all5-kaggle-x4-tau085-tta`: mean of the five families' share
  children, each predicting its own validation fold at ×4, τ 0.85, mirror TTA.
- **mixed**, table `data/labels/prepared/gen3-mixed-kaggle-x4-tau085-tta`: half that mean, half the gen-0
  table T2 (probabilities averaged), the literature's remedy against students converging (share
  children agree at 0.99 logit correlation).

Runs are named `<family>-gen3all5-finetune-fold-k` and `<family>-gen3mixed-finetune-fold-k`.

## Read first

1. Memory: `silver15-external-plan.md` (the running plan, every order the user gave, the lessons),
   `never-decide-submissions.md`, `s126-best-public.md` (score history), `vast-rented-machine.md`.
2. [fastMRI joint training](fastmri-joint-training.md), from "Third round" to the end: the share
   cycle, "Next generation (planned 2026-10-17)", S171 to S174 and the selection error.
3. The lineage page the user asked for: <https://claude.ai/artifact/WYJk8JAUoMBZS8VRNDQsf3>.

## Rules the user set

- **Never decide a submission.** Propose with evidence; send to the submission session only what the
  user approved explicitly. The submission session is the Claude Desktop session named **Submission**
  (find it with ListAgents; it was "Kaggle submission handler"). It exports, hash-checks, launches and
  reports; you never launch yourself.
- **Share students are judged on the board only.** Their gold out-of-fold readings leak (their
  teachers saw each fold's gold labels) and the leak differs by family: S171 (MambaVision + ResNet-50,
  gold +0.0031 over ConvNeXt alone) scored 0.964 below S170. New ensembles start from ConvNeXt × 5 and
  add a family only when a submission shows it helps. Gen-0 gold readings (58 studies, 5 folds) are
  clean.
- Machine: at most 3 trainings at once; check the GPU before launching; delete `last.ckpt` after a
  finished run but keep the pretrainings' `last.ckpt` (every generation starts from them); copy each
  finished run without `last.ckpt` to `/home/claudio/src/knee-abnormality/outputs/training/`; no
  credentials left on the machine; destroying the instance is the user's call.
- The local 3090 runs the user's campaigns; production stays on the machine.
- Keep answers brief; report the place among equals for tied public scores.

## The machine

`ssh -i /home/claudio/.ssh/vast_knee -p 22038 root@136.55.192.172` (filter the banner with
`grep -v -i "welcome\|have fun"`). RTX PRO 6000 98 GB.

- Trees: `/home/claudio/src/knee-abnormality-sync-free` (ConvNeXt, MambaVision, ResNet-50; its venv is
  used for all families), `/home/claudio/src/knee-abnormality-teachers2` (FocalNet),
  `/home/claudio/src/knee-abnormality-effvit` (EfficientViT; behind `scripts/tools/preflight_training.py`
  with `KNEE_AUTORESEARCH_LIMITS=/root/effvit-limits.json` set for the preflight only).
- tmux `gen3v`: `/root/cycle_gen3_variants.sh`. Order: ConvNeXt all5 (done), ConvNeXt mixed (done),
  then ResNet-50 (done), MambaVision, EfficientViT, FocalNet mixed, then ResNet-50, MambaVision,
  EfficientViT, FocalNet all5. Expected end about 09:30 CEST Sunday. tmux `ram_guard` stops the newest
  training below 8 GB free RAM.
- Every event goes to `/workspace/outputs/logs/night6/chain.state`; logs per run in the same folder.
- Re-predictions of the share children: `/workspace/predictions/gen3-teachers/` (script
  `/root/predict_teacher_x4.py`), local copy `outputs/predictions/gen3-teachers/`.
- Disk: about 110 GB free on 2026-10-17 evening; the monitor alerts under 50 GB.
- **Maintenance: Vast takes the instance offline Monday 00:00 to 01:00 CEST.** Launch nothing that
  spans it; tmux sessions do not survive a restart, so check and restart afterwards.

## Monitor to place

Write this script to your scratchpad, seed `watch.seen` with the current line count of
`chain.state` so old events are not replayed, and run it with Monitor (30-minute expiry, re-arm each
time). It prints new `chain.state` lines and alerts on GPU memory over 93 GB, RAM under 20 GB, disk
under 50 GB, and new error logs.

```bash
#!/bin/bash
SSH="ssh -i /home/claudio/.ssh/vast_knee -p 22038 -o ConnectTimeout=20 -o ServerAliveInterval=15 -o BatchMode=yes root@136.55.192.172"
DIR=<your scratchpad>
SEEN=$DIR/watch.seen
ALERTED=$DIR/watch.alerted
[ -f $SEEN ] || echo 0 > $SEEN
touch $ALERTED
failures=0
while true; do
  out=$($SSH 'S=/workspace/outputs/logs/night6/chain.state; wc -l < $S; cat $S; echo "@@"; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -g | awk "/Mem:/ {print \$7}"; df -BG /workspace | awk "NR==2 {print \$4}" | tr -d G; grep -l -E "Traceback|CUDA out of memory|Killed" /workspace/outputs/logs/night6/*.log 2>/dev/null | xargs -r -n1 basename' 2>/dev/null | grep -v -i "welcome\|have fun")
  if [ -z "$out" ]; then
    failures=$((failures + 1))
    [ $failures -eq 5 ] && echo "remote: ssh failing for 5 minutes"
    sleep 60; continue
  fi
  failures=0
  total=$(echo "$out" | head -1)
  seen=$(cat $SEEN)
  if [ "$total" -gt "$seen" ]; then
    echo "$out" | sed -n "2,/^@@$/p" | grep -v "^@@$" | tail -n $((total - seen)) | sed "s/^/remote: /"
    echo $total > $SEEN
  fi
  rest=$(echo "$out" | sed -n "/^@@$/,\$p" | tail -n +2)
  gpu=$(echo "$rest" | sed -n 1p); ram=$(echo "$rest" | sed -n 2p); disk=$(echo "$rest" | sed -n 3p)
  errors=$(echo "$rest" | tail -n +4)
  [ -n "$gpu" ] && [ "$gpu" -gt 93000 ] && echo "remote ALERT: GPU memory ${gpu} MiB"
  [ -n "$ram" ] && [ "$ram" -lt 20 ] && echo "remote ALERT: RAM available ${ram} GB"
  [ -n "$disk" ] && [ "$disk" -lt 50 ] && echo "remote ALERT: disk free ${disk} GB"
  for e in $errors; do grep -q -x "$e" $ALERTED || { echo "remote ALERT: error in log $e"; echo "$e" >> $ALERTED; }; done
  sleep 60
done
```

On each `training EXIT 0`, copy the run:
`rsync -a -e "ssh -i /home/claudio/.ssh/vast_knee -p 22038" --exclude last.ckpt root@136.55.192.172:/workspace/outputs/training/<run> /home/claudio/src/knee-abnormality/outputs/training/`.
On an `EXIT` other than 0, read the run's log and fix before relaunching. Known traps: launching a
training with `KNEE_AUTORESEARCH_LIMITS` set makes it crash on `KeyError: 'input_slices'` (only the
preflight may see it); `pgrep -f`/`pkill -f` patterns also match your own ssh command line.

## Waiting on the board

| Submission | Members | Launch (timers in the Submission session) | Reference |
|---|---|---|---|
| S173 | ConvNeXt + ResNet-50 share children, 10 equal | 02:01 | S170 0.965 |
| S174 | ConvNeXt mixed × 5 | 02:03 | S170 0.965 |

Scores so far: S170 0.965 (best); 0.964 sorted S169 > S167 > S163 > S154 > S165 > S168 > S156 > S161 >
S172 > S171 > S155; S166 0.963.

## Open decisions for the user (morning of 2026-10-18)

- Whether to pivot, given S173 and S174.
- Which variant (all5 or mixed) and which families to build ensembles from; ConvNeXt all5 × 5 is
  trained and unsubmitted.
- The full-data ConvNeXt child (S167's recipe on all five folds): the user decides after S173/S174.
- What the machine runs after the queue ends (about 09:30).
