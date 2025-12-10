import torch
import patchcore.patchcore
import patchcore.utils
import numpy as np
import os

# モデルの読み込み
model_path = "results/.../models/safe_pose/Ensemble-1-1_.pc"
patchcore_model = patchcore.patchcore.PatchCore(device=torch.device("cuda:0"))
patchcore_model.load_from_path(model_path)

# 推論対象の画像を読み込み（あなたの環境に合わせて）
test_loader = ...

# 推論だけ行う
scores, segmentations, labels_gt, masks_gt = patchcore_model.predict(test_loader)

print("推論完了")
print(scores)
