#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
export PYTHONPATH=src

DATAPATH="/home/yamamaoo/patchcore_git/datasets/pose_dataset"
MODEL_DIR="/home/yamamaoo/patchcore_git/patchcore-inspection/results/pose_results/safe_pose_train_24/models/mvtec_safe_pose"

python - <<EOF
import torch
import patchcore.patchcore
import patchcore.backbones
import patchcore.common
import patchcore.datasets.mvtec as mvtec
from run_patchcore_infer_only import benchmark_inference

device = torch.device("cuda:0")

# Dataset（testのみ）
dataset = mvtec.MVTecDataset(
    "${DATAPATH}",
    classname="safe_pose",
    split=mvtec.DatasetSplit.TEST,
    resize=384,
    imagesize=384,
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# PatchCore 復元
pc = patchcore.patchcore.PatchCore(device)
pc.load_from_path("${MODEL_DIR}")

benchmark_inference(pc, dataloader)
EOF
