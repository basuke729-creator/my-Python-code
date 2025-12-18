#!/usr/bin/env bash

cd "$(dirname "$0")"

export PYTHONPATH=src

datapath="/home/yamamaoo/patchcore_git/datasets/pose_dataset"

python bin/run_patchcore.py \
  --gpu 0 \
  --seed 0 \
  --log_group bench_infer_safe_pose \
  --log_project pose_results \
  results \
  patch_core \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1024 \
    --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 10 \
    --patchsize 5 \
  sampler \
    approx_greedy_coreset \
    -p 0.1 \
  dataset \
    --augment \
    -d safe_pose \
    mvtec "$datapath"
