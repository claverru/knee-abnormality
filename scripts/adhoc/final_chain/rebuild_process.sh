#!/bin/bash
D=/workspace/downloads
L=/workspace/logs
S=$L/rebuild.state
E=/root/layoutenv/bin/python
step() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
wait_for() { until grep -q "$1 download EXIT" $S; do sleep 30; done; grep -q "$1 download EXIT 0" $S; }

/root/.local/bin/uv venv -q /root/layoutenv --python 3.12 && /root/.local/bin/uv pip install -q --python $E polars pydicom numpy pillow > $L/layoutenv.log 2>&1
step "layout environment EXIT $?"

kaggle() {
  wait_for kaggle.zip || { step "kaggle download failed"; return 1; }
  mkdir -p /workspace/data/raw
  $E /root/parallel_unzip.py $D/kaggle.zip /workspace/data/raw 60 > $L/unzip_kaggle.log 2>&1; rc=$?
  step "kaggle unzip EXIT $rc"
  [ $rc -eq 0 ] && rm -f $D/kaggle.zip $D/kaggle.zip.aria2 && step "kaggle archive deleted"
}

fastmri() {
  name=knee_DICOMs_batch$1
  wait_for $name.tar.xz || { step "$name download failed"; return 1; }
  xz --robot -lv $D/$name.tar.xz | grep -E "^totals" > $L/xz_$name.txt
  mkdir -p /workspace/data/raw/fastmri/batch$1
  xz -T60 -dc $D/$name.tar.xz | tar -x -C /workspace/data/raw/fastmri/batch$1; rc=$?
  step "$name extract EXIT $rc"
  [ $rc -eq 0 ] && rm -f $D/$name.tar.xz $D/$name.tar.xz.aria2 && step "$name archive deleted"
}

kaggle &
fastmri 1 &
fastmri 2 &
wait
grep -q "batch1 extract EXIT 0" $S && grep -q "batch2 extract EXIT 0" $S || { step "fastMRI extraction incomplete, stopping"; exit 1; }
$E /root/fastmri_layout.py /workspace/data/fastmri /workspace/data/raw/fastmri/batch1 /workspace/data/raw/fastmri/batch2 > $L/fastmri_layout.log 2>&1
step "fastmri layout EXIT $?"
$E /root/uncompress_fastmri.py sample > $L/uncompress_sample.log 2>&1
step "fastmri sample EXIT $?"
$E /root/uncompress_fastmri.py convert > $L/uncompress_convert.log 2>&1
step "fastmri convert EXIT $?"
$E /root/uncompress_fastmri.py verify > $L/uncompress_verify.log 2>&1
step "fastmri verify EXIT $?"
step "PROCESSING DONE"
