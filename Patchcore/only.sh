#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# venv
source /home/yamamao/patchcore_git/corevenv/bin/activate
export PYTHONPATH=src

MODEL_DIR="/home/yamamao/patchcore_git/patchcore-inspection/results/pose_results/safe_pose_train_24/models/mvtec_safe_pose"
DATA_PATH="/home/yamamao/patchcore_git/datasets/pose_dataset"

python run_patchcore_infer_only.py \
  --model_dir "$MODEL_DIR" \
  --data_path "$DATA_PATH" \
  --subdataset safe_pose \
  --gpu 0 \
  --batch_size 1 \
  --num_workers 4 \
  --resize 384 \
  --imagesize 384 \
  --repeat 105 \
  --warmup 5




cd /home/yamamao/patchcore_git/patchcore-inspection
source /home/yamamao/patchcore_git/corevenv/bin/activate
export PYTHONPATH=src

python run_patchcore_infer_only.py \
  --model_dir "/home/yamamao/patchcore_git/patchcore-inspection/results/pose_results/safe_pose_train_24/models/mvtec_safe_pose" \
  --data_path "/home/yamamao/patchcore_git/datasets/pose_dataset" \
  --subdataset safe_pose \
  --gpu 0 \
  --batch_size 1 \
  --num_workers 4 \
  --resize 384 \
  --imagesize 384 \
  --repeat 105 \
  --warmup 5 \
  --limit_images 1
