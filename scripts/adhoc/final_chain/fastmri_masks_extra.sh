#!/bin/bash
set -e
ulimit -n 1048576
C=/home/claudio/.cache/knee-segmenters
UV=/root/.local/bin/uv
N=fastmri-extra
R=/home/claudio/src/knee-abnormality-sync-free
B=$R/scripts/adhoc/side_routed_session
S=/workspace/outputs/logs/fastmri_masks_extra.state
step() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
step "headers"
/root/layoutenv/bin/python /root/fastmri_study_headers.py > /workspace/outputs/logs/fastmri_study_headers.log 2>&1
step "environment"
cd $C
[ -x venv/bin/python ] || $UV venv -q --python 3.12 venv
$UV pip install -q --python venv/bin/python nnunetv2==2.8.1 nibabel==5.4.2 simpleitk==2.5.6 scipy==1.18.1 numpy==2.5.3 torch==2.14.1 acvl-utils==0.2.6 batchgenerators==0.25.3 batchgeneratorsv2==0.3.5 dynamic-network-architectures==0.4.4 polars > /workspace/outputs/logs/segmenter_env.log 2>&1
P=$C/venv/bin/python
step "choose"
/root/layoutenv/bin/python /root/fastmri_choose_extra.py $N 0.2 > /workspace/outputs/logs/fastmri_choose_extra.log 2>&1
step "convert"
$P /root/fastmri_convert.py $N 30 > /workspace/outputs/logs/fastmri_convert_extra.log 2>&1
cd $R
step "segment"
for attempt in 1 2 3 4 5 6; do
  STEP=1.0 $P $B/segmenter_mrannotator.py $N mrannotator >> /workspace/outputs/logs/fastmri_mrannotator_extra.log 2>&1 && break
done
step "segmented $(ls $C/$N/mrannotator/*.nii.gz | wc -l) volumes"
step "export bones"
$P $B/segmenter_export_targets.py $N anatomy-mrannotator-$N > /workspace/outputs/logs/fastmri_export_extra.log 2>&1
step "export joints"
$P $B/segmenter_export_joint_targets.py $N anatomy-mrannotator-$N anatomy-mrannotator-$N-joints >> /workspace/outputs/logs/fastmri_export_extra.log 2>&1
step "measure"
$P $B/mask_measurements.py $N anatomy-mrannotator-$N-joints >> /workspace/outputs/logs/fastmri_export_extra.log 2>&1
step "DONE"
