import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report
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
from torchvision.models import resnet18

backbone = resnet18(pretrained=True)
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


# ===============================
# 距離計算（kNN距離）
# ===============================
def anomaly_score(features, bank):
    # L2距離の最小値（各サンプルごと）
    dists = np.sqrt(((features[:, None, :] - bank[None, :, :]) ** 2).sum(axis=2))
    score = dists.min(axis=1)
    return score


# ===============================
# 推論関数（val / test 共通）
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
print("Confusion Matrix:\n", cm)

print("\nClassification Report:")
print(classification_report(
    test_labels, pred,
    target_names=["normal (safe)", "abnormal (danger)"]
))

disp = ConfusionMatrixDisplay(cm, display_labels=["normal", "abnormal"])
disp.plot()
plt.title("PatchCore (ResNet18 kNN) Confusion Matrix")
plt.tight_layout()
plt.show()
