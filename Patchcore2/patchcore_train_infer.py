import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report

# ======================
# ★ 設定ゾーン（要編集）
# ======================
DATASET_ROOT = "/home/yamamao/Patchcore/dataset"  # ← あなたの環境に合わせる

NORMAL_CLASS_NAME = "normal"
ABNORMAL_CLASS_NAME = "abnormal"

IMAGE_SIZE = 384
BATCH_SIZE = 8
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SAVE_PATH = "patchcore_memorybank.pth"


# =======================
# 前処理
# =======================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# =======================
# Dataset
# =======================
train_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, "val"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, "test"), transform=transform)

print("=== Class check ===")
print("train:", train_dataset.classes)
print("val:", val_dataset.classes)
print("test:", test_dataset.classes)

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# =======================
# PatchCore モデル
# =======================
from anomalib.models.image.patchcore import Patchcore

model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    pre_trained=True
).to(DEVICE)


# =======================
# ★ 学習（特徴抽出 → メモリバンク作成）
# =======================
print("\n=== Extracting train(normal) features ===")

train_features = []

model.eval()  # post_processor を止める

with torch.no_grad():
    for imgs, _ in train_loader:
        imgs = imgs.to(DEVICE)

        # ★ 学習時は features_only=True
        feats = model(imgs, features_only=True)   # Tensor が返る
        train_features.append(feats.cpu())

train_features = torch.cat(train_features, dim=0)
print("Train features:", train_features.shape)

print("\n=== Building memory bank ===")
model.fit(train_features)

# 保存
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved:", MODEL_SAVE_PATH)


# =======================
# ★ 推論共通関数
# =======================
def get_scores_and_labels(loader, dataset):
    model.eval()
    scores = []
    labels = []

    class_to_bin = {
        dataset.class_to_idx[NORMAL_CLASS_NAME]: 0,
        dataset.class_to_idx[ABNORMAL_CLASS_NAME]: 1
    }

    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(DEVICE)

            # ★ 推論は return_predictions=True
            out = model(imgs, return_predictions=True)

            # anomaly_score を取得
            anomaly_scores = out["anomaly_score"].detach().cpu().numpy()

            scores.extend(anomaly_scores)
            labels.extend([class_to_bin[int(l)] for l in labs])

    return np.array(scores), np.array(labels)


# =======================
# ★ 閾値決定（val）
# =======================
print("\n=== Running validation for threshold ===")
val_scores, val_labels = get_scores_and_labels(val_loader, val_dataset)

fpr, tpr, thr = roc_curve(val_labels, val_scores)
youden = tpr - fpr
best_idx = np.argmax(youden)
best_threshold = thr[best_idx]

print(f"Best threshold (Youden J): {best_threshold:.4f}")


# =======================
# ★ test セット評価（混同行列）
# =======================
print("\n=== Running test evaluation ===")
test_scores, test_labels = get_scores_and_labels(test_loader, test_dataset)

pred_labels = (test_scores >= best_threshold).astype(int)

cm = confusion_matrix(test_labels, pred_labels)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    test_labels,
    pred_labels,
    target_names=["normal (safe)", "abnormal (danger)"]
))

# 図表示
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["normal (safe)", "abnormal (danger)"])
disp.plot()
plt.title("PatchCore Confusion Matrix (test)")
plt.tight_layout()
plt.show()
