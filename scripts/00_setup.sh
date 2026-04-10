#!/usr/bin/env bash
# RunPod 环境初始化 (只需跑一次)
set -euo pipefail
cd /workspace
# git clone <your-repo-url> Codes
cd Codes
pip install -r requirements.txt
pip install scikit-image fvcore
mkdir -p results
echo "Setup done."
