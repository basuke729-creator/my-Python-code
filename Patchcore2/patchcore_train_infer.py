import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report

# --------------------------------------------------------
# ★★★ ここだけ環境に合わせて書き換えればOK ★★★
# --------------------------------------------------------
DATASET_ROOT = "/home/yamamao/Patchcore/dataset"  # ← ここを合わせる

NORMAL_CLASS_NAME = "normal"
ABNORMAL_CLASS_NAME = "abnormal"

IMAGE_SIZE = 384
BATCH_SIZE = 8
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SAVE_PATH = "patchcore_memorybank.pth"
# --------------------------------------------------------


# ============================
# 前処理（transform）
# ============================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ============================
# Dataset & DataLoader
# ============================
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_ROOT, "train"),
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_ROOT, "val"),
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_ROOT, "test"),
    transform=transform
)

print("=== Classes check ===")
print("Train classes:", train_dataset.classes)
print("Val   classes:", val_dataset.classes)
print("Test  classes:", test_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS)


# ============================
# PatchCore モデル本体
# ============================
from anomalib.models.image.patchcore import Patchcore

model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    pre_trained=True
).to(DEVICE)


# ============================
# ★★★ PatchCore 学習：normal の特徴抽出 → メモリバンク作成 ★★★
# ============================

print("\n=== Extracting train(normal) features ===")

train_features = []

with torch.no_grad():
    for imgs, labs in train_loader:
        imgs = imgs.to(DEVICE)

        # 特徴抽出
        out = model(imgs)
        feats = out["features"]       # ★ features を取る

        train_features.append(feats.cpu())

train_features = torch.cat(train_features, dim=0)
print("Train features:", train_features.shape)

# ★ メモリバンク構築（学習）
print("\n=== Building memory bank ===")
model.fit(train_features)

# 保存
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved: {MODEL_SAVE_PATH}")


# ============================
# ★★★ 推論関数：val/test 共通 ★★★
# ============================
def get_scores_and_labels(loader, dataset):
    model.eval()
    scores = []
    labels = []

    class_to_binary = {
        dataset.class_to_idx[NORMAL_CLASS_NAME]: 0,
        dataset.class_to_idx[ABNORMAL_CLASS_NAME]: 1,
    }

    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(DEVICE)

            out = model(imgs)
            anom = out["anomaly_score"].cpu().numpy()

            scores.extend(anom)
            labels.extend([class_to_binary[int(l)] for l in labs])

    return np.array(scores), np.array(labels)


# ============================
# ★★★ val セットで最適閾値決定 ★★★
# ============================
print("\n=== Inference on val (threshold selection) ===")

val_scores, val_labels = get_scores_and_labels(val_loader, val_dataset)

fpr, tpr, thr = roc_curve(val_labels, val_scores)
youden = tpr - fpr
best_idx = np.argmax(youden)
best_threshold = thr[best_idx]

print(f"Best threshold = {best_threshold:.4f}")


# ============================
# ★★★ test セット最終評価（混同行列） ★★★
# ============================
print("\n=== Inference on test ===")

test_scores, test_labels = get_scores_and_labels(test_loader, test_dataset)

pred_labels = (test_scores >= best_threshold).astype(int)

cm = confusion_matrix(test_labels, pred_labels)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    test_labels, pred_labels,
    target_names=["normal (safe)", "abnormal (danger)"]
))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["normal (safe)", "abnormal (danger)"])
disp.plot()
plt.title("PatchCore Confusion Matrix (test)")
plt.tight_layout()
plt.show()


