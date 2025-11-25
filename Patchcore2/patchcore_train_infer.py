import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ===============================
# 設定
# ===============================
DATASET_ROOT = "/home/yamamao/Patchcore/dataset"
NORMAL_CLASS_NAME = "normal"
ABNORMAL_CLASS_NAME = "abnormal"

IMAGE_SIZE = 384
BATCH_SIZE = 8
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEMORY_BANK_PATH = "memory_bank.npy"
CONF_MAT_PATH = "confusion_matrix_norm.png"  # ★ 正規化版の画像ファイル名


# ===============================
# 前処理
# ===============================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ===============================
# データセット
# ===============================
train_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, "train"), transform=transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATASET_ROOT, "val"),   transform=transform)
test_dataset  = datasets.ImageFolder(os.path.join(DATASET_ROOT, "test"),  transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print("train:", train_dataset.classes)
print("val:",   val_dataset.classes)
print("test:",  test_dataset.classes)


# ===============================
# backbone（ResNet18）
# ===============================
from torchvision.models import resnet18, ResNet18_Weights

backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
backbone.fc = nn.Identity()   # 最終層を無効化（特徴ベクトルだけ取得）
backbone = backbone.to(DEVICE)
backbone.eval()


# ===============================
# 特徴抽出関数
# ===============================
def extract_features(loader):
    feats = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            f = backbone(imgs)              # (B, 512)
            feats.append(f.cpu().numpy())

    feats = np.concatenate(feats, axis=0)
    return feats   # shape = (N, 512)


# ===============================
# MemoryBank作成（train normal）
# ===============================
print("\n=== Building MemoryBank ===")
memory_bank = extract_features(train_loader)
print("Memory bank shape:", memory_bank.shape)

np.save(MEMORY_BANK_PATH, memory_bank)
print("Memory bank saved:", MEMORY_BANK_PATH)


# ===============================
# 距離計算（kNN距離）
# ===============================
def anomaly_score(features, bank):
    dists = np.sqrt(((features[:, None, :] - bank[None, :, :]) ** 2).sum(axis=2))
    score = dists.min(axis=1)
    return score


# ===============================
# 推論関数
# ===============================
def get_scores_and_labels(loader, dataset):
    feats = extract_features(loader)
    scores = anomaly_score(feats, memory_bank)

    class_to_bin = {
        dataset.class_to_idx[NORMAL_CLASS_NAME]: 0,
        dataset.class_to_idx[ABNORMAL_CLASS_NAME]: 1,
    }

    labels = []
    for _, labs in loader:
        for l in labs:
            labels.append(class_to_bin[int(l)])

    return np.array(scores), np.array(labels)


# ===============================
# 閾値（val）
# ===============================
print("\n=== Validation ===")
val_scores, val_labels = get_scores_and_labels(val_loader, val_dataset)

fpr, tpr, thr = roc_curve(val_labels, val_scores)
best_idx = np.argmax(tpr - fpr)
best_threshold = thr[best_idx]

print(f"Best threshold = {best_threshold:.4f}")


# ===============================
# test 評価
# ===============================
print("\n=== Test ===")
test_scores, test_labels = get_scores_and_labels(test_loader, test_dataset)
pred = (test_scores >= best_threshold).astype(int)

cm = confusion_matrix(test_labels, pred)
print("Confusion Matrix (counts):\n", cm)

print("\nClassification Report:")
print(classification_report(
    test_labels, pred,
    target_names=["normal (safe)", "abnormal (danger)"]
))


# ===============================
# 混同行列（行正規化＆%表示）の画像保存
# ===============================
labels = ["normal", "abnormal"]

# 行ごと（Trueごと）に 0〜100% に正規化
cm_float = cm.astype(np.float64)
row_sums = cm_float.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm_float, row_sums, where=row_sums != 0) * 100.0  # 百分率

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Percentage (%)")

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("PatchCore (ResNet18 kNN)\nConfusion Matrix (row-normalized, %)")

# マスの中に%を表示（小数1桁）
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        ax.text(j, i, f"{cm_norm[i, j]:.1f}",
                ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig(CONF_MAT_PATH, dpi=300)
plt.close()

print(f"\nSaved normalized confusion matrix as: {CONF_MAT_PATH}")

