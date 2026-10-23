#!/bin/bash
D=/workspace/downloads
L=/workspace/logs
mkdir -p $D $L /workspace/data/raw/fastmri
S=$L/rebuild.state
step() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq aria2 xz-utils unzip tmux rsync >/dev/null
step "tools installed"
download() {
  name=$1; url=$2
  aria2c -x16 -s16 -k64M --file-allocation=none --summary-interval=60 -d $D -o $name "$url" > $L/download_$name.log 2>&1
  step "$name download EXIT $?"
}
download kaggle.zip "$(cat /root/kaggle_url.txt)" &
download knee_DICOMs_batch1.tar.xz "$(sed -n 1p /root/fastmri_urls.txt)" &
download knee_DICOMs_batch2.tar.xz "$(sed -n 2p /root/fastmri_urls.txt)" &
wait
rm -f /root/kaggle_url.txt
step "DOWNLOADS DONE"
