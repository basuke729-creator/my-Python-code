import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
import pandas as pd
import shutil


# ================================
# 設定
# ================================
DATASET_ROOT = "/home/yamamao/Patchcore/dataset"
NORMAL_CLASS = "normal"
ABNORMAL_CLASS = "abnormal"

IMAGE_SIZE = 384
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PatchCore用
PATCH_LAYERS = ["layer2", "layer3"]   # PatchCore本来の推奨設定
CORESET_RATIO = 0.01                  # メモリ圧縮: 1%だけ残す


# ================================
# 前処理
# ================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ================================
# PatchCore：パッチ特徴抽出クラス
# ================================
class PatchCoreExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-2])  # conv1〜layer4
        self.layer2 = m.layer2
        self.layer3 = m.layer3

        self.return_layers = {
            "layer2": self.layer2,
            "layer3": self.layer3,
        }

    def forward(self, x):
        # conv1〜layer1まで進める
        x = self.backbone[0](x)
        x = self.backbone[1](x)
        x = self.backbone[2](x)
        x = self.backbone[3](x)

        outputs = {}
        x = self.backbone[4](x)      # layer1
        x = self.return_layers["layer2"](x)
        outputs["layer2"] = x

        x = self.return_layers["layer3"](x)
        outputs["layer3"] = x

        return outputs


# ================================
# パッチ特徴を flatten して1次元に
# ================================
def flatten_patches(feat_dict):
    """
    feat_dict = {"layer2": (B,C,H,W), "layer3": (B,C,H,W)}
    → return (B, H*W, C_all)
    """
    patches = []
    for k, feat in feat_dict.items():
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1)  # (B,H,W,C)
        feat = feat.reshape(B, H * W, C)
        patches.append(feat)

    # (B, num_patches, total_channels)
    return torch.cat(patches, dim=2)


# ================================
# MemoryBank構築（train normalのみ）
# ================================
def build_memory_bank(loader, model):
    memory = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            feats = model(imgs)
            flat = flatten_patches(feats)         # (B, P, D)
            flat = flat.cpu().numpy()
            memory.append(flat)

    memory = np.concatenate(memory, axis=0)        # (N, P, D)
    N, P, D = memory.shape
    memory = memory.reshape(N * P, D)              # 全パッチを1次元へ
    print("Raw MemoryBank shape =", memory.shape)
    return memory


# ================================
# Coreset（Greedy k-Center）
# ================================
def coreset_sampling(memory, ratio=0.01):
    n = memory.shape[0]
    m = int(n * ratio)

    selected = [np.random.randint(n)]
    distances = np.linalg.norm(memory - memory[selected[-1]], axis=1)

    for _ in range(m - 1):
        idx = np.argmax(distances)
        selected.append(idx)
        new_dist = np.linalg.norm(memory - memory[idx], axis=1)
        distances = np.minimum(distances, new_dist)

    print(f"Coreset sampled: {m}/{n}")
    return memory[selected]

# ================================
# kNN 距離で anomaly score
# ================================
def anomaly_score(patches, memory_bank):
    # patches: (P, D)
    dists = np.linalg.norm(memory_bank[None, :, :] - patches[:, None, :], axis=2)
    score = dists.min(axis=1).max()    # PatchCore論文のスコア
    return score


# ================================
# 推論（test/val 用）
# ================================
def infer_scores(loader, dataset, model, memory_bank):
    scores = []
    labels = []
    paths = []

    class_to_bin = {
        dataset.class_to_idx[NORMAL_CLASS]: 0,
        dataset.class_to_idx[ABNORMAL_CLASS]: 1,
    }

    with torch.no_grad():
        idx = 0
        for imgs, labs in loader:
            imgs = imgs.to(DEVICE)
            feats = model(imgs)
            flat = flatten_patches(feats)  # (B,P,D)

            flat = flat.cpu().numpy()
            B, P, D = flat.shape

            for b in range(B):
                score = anomaly_score(flat[b], memory_bank)
                scores.append(score)
                labels.append(class_to_bin[int(labs[b])])
                sample_path, _ = dataset.samples[idx]
                paths.append(sample_path)
                idx += 1

    return np.array(scores), np.array(labels), paths


# ================================
# ここからメイン処理
# ================================
model = PatchCoreExtractor().to(DEVICE)
model.eval()

train_ds = datasets.ImageFolder(os.path.join(DATASET_ROOT, "train"), transform=transform)
val_ds   = datasets.ImageFolder(os.path.join(DATASET_ROOT, "val"),   transform=transform)
test_ds  = datasets.ImageFolder(os.path.join(DATASET_ROOT, "test"),  transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)


# --- MemoryBank構築 ---
print("\n=== Building MemoryBank ===")
memory = build_memory_bank(train_loader, model)
memory = coreset_sampling(memory, Coreset_RATIO := 0.01)  # 1%
print("Final MemoryBank =", memory.shape)


# --- 閾値（val） ---
val_scores, val_labels, _ = infer_scores(val_loader, val_ds, model, memory)
fpr, tpr, thr = roc_curve(val_labels, val_scores)
best_idx = np.argmax(tpr - fpr)
threshold = thr[best_idx]
print("Best threshold =", threshold)


# --- test 推論 ---
test_scores, test_labels, test_paths = infer_scores(test_loader, test_ds, model, memory)
pred = (test_scores >= threshold).astype(int)


# --- 混同行列 ---
cm = confusion_matrix(test_labels, pred)
print("Confusion matrix:\n", cm)

print("\nClassification report:")
print(classification_report(test_labels, pred))


# 保存：正規化コンフマト
cm_f = cm.astype(float)
cm_norm = cm_f / cm_f.sum(axis=1, keepdims=True) * 100

plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
plt.title("PatchCore Confusion Matrix (%)")
plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()


# --- 画像仕分け & CSV ---
os.makedirs("results/normal", exist_ok=True)
os.makedirs("results/abnormal", exist_ok=True)

true_name = {0: "normal", 1: "abnormal"}

rows = []
for p, t, y, s in zip(test_paths, test_labels, pred, test_scores):
    fname = os.path.basename(p)
    dst = f"results/{true_name[y]}/{fname}"
    shutil.copy2(p, dst)

    rows.append([p, fname, true_name[t], true_name[y], s])


df = pd.DataFrame(rows, columns=["filepath", "filename", "true", "pred", "score"])
df.to_csv("predictions.csv", index=False)
print("Saved predictions.csv")

