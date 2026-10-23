#!/bin/bash
ulimit -n 1048576
cd /home/claudio/src/knee-abnormality-fastmri-pretraining
export HF_HUB_OFFLINE=1 PYTHONPATH=src:.
O=/workspace/screen/oai
mkdir -p $O
.venv/bin/python /root/screen_loader.py $O /workspace/data/oai > /workspace/logs/oai_screen_loader.log 2>&1
echo "loader EXIT $?" >> /workspace/logs/oai_screen_loader.log
.venv/bin/python /root/screen_geometry_any.py /workspace/data/oai $O > /workspace/logs/oai_screen_geometry.log 2>&1
echo "geometry EXIT $?" >> /workspace/logs/oai_screen_geometry.log
echo SCREENS DONE >> /workspace/logs/oai_screen_geometry.log
