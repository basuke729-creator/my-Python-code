import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from tqdm import tqdm

# ================================
# 設定
# ================================
DATASET_ROOT = "/home/yamamao/Patchcore/dataset"
NORMAL_CLASS = "normal"
ABNORMAL_CLASS = "abnormal"

IMAGE_SIZE = 384
BATCH_SIZE = 8               # ↑ ここを増やすとさらに速くなる（GPUメモリと相談）
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORESET_RATIO = 0.01         # 1% coreset（前と同じなので精度は変わらない）

CONF_MAT_PATH = "confusion_matrix_norm.png"
CSV_PATH = "predictions.csv"
OUTPUT_DIR = "results"


# ================================
# 前処理
# ================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ================================
# PatchCore 用 ResNet50
# ================================
class PatchCoreExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        f2 = self.layer2(x)
        f3 = self.layer3(f2)

        return {"layer2": f2, "layer3": f3}


# ================================
# layer2 を layer3 サイズに合わせてパッチ化
# ================================
def flatten_patches(feat_dict):
    """
    feat_dict = {"layer2": (B,C2,H2,W2), "layer3": (B,C3,H3,W3)}
    → layer2 を layer3 のサイズにアップサンプルしてチャンネル結合
    → (B, H3*W3, C2+C3)
    """
    f2 = feat_dict["layer2"]
    f3 = feat_dict["layer3"]

    B, C2, H2, W2 = f2.shape
    B, C3, H3, W3 = f3.shape

    # layer2 を layer3 の空間サイズにアップサンプル
    f2_up = F.interpolate(f2, size=(H3, W3), mode="bilinear", align_corners=False)

    # チャンネル方向で結合 → (B, C2+C3, H3, W3)
    feat = torch.cat([f2_up, f3], dim=1)

    # (B, H3*W3, C_total) へ
    feat = feat.permute(0, 2, 3, 1)          # (B, H3, W3, C_total)
    feat = feat.reshape(B, H3 * W3, C2 + C3) # (B, P, D)

    return feat


# ================================
# MemoryBank構築（train normal）
# ================================
def build_memory_bank(loader, model):
    memory = []

    print("\n=== Building MemoryBank ===")
    with torch.no_grad():
        for imgs, _ in tqdm(loader, total=len(loader),
                            desc="Extracting train(normal) patches"):
            imgs = imgs.to(DEVICE)
            feats = model(imgs)
            flat = flatten_patches(feats)         # (B, P, D)
            flat = flat.cpu().numpy()
            memory.append(flat)

    memory = np.concatenate(memory, axis=0)       # (N, P, D)
    N, P, D = memory.shape
    memory = memory.reshape(N * P, D)             # (N*P, D)

    print("Raw MemoryBank shape =", memory.shape)
    return memory


# ================================
# Coreset（Greedy k-Center）
# ================================
def coreset_sampling(memory, ratio=0.01):
    n = memory.shape[0]
    m = max(1, int(n * ratio))

    # n が小さい場合はそのまま
    if m >= n:
        print(f"Coreset skipped (m>=n). Final MemoryBank = {memory.shape}")
        return memory

    print(f"Running coreset sampling (ratio={ratio}, select {m}/{n}) ...")

    # ここは numpy のまま（アルゴリズムは前と同じ → 精度も同じ）
    selected = [np.random.randint(n)]
    distances = np.linalg.norm(memory - memory[selected[-1]], axis=1)

    for _ in tqdm(range(m - 1), desc="Coreset k-center"):
        idx = np.argmax(distances)
        selected.append(idx)
        new_dist = np.linalg.norm(memory - memory[idx], axis=1)
        distances = np.minimum(distances, new_dist)

    sampled = memory[selected]
    print(f"Coreset sampled: {len(selected)}/{n}")
    print("Final MemoryBank =", sampled.shape)
    return sampled


# ================================
# kNN 距離（GPU / 行列計算で“厳密”計算）
# ================================
def anomaly_score_torch(patches, memory_bank_t):
    """
    patches: (P, D) torch.Tensor (DEVICE)
    memory_bank_t: (M, D) torch.Tensor (DEVICE)

    距離^2 = ||x||^2 + ||y||^2 - 2 x·y を使って
    L2距離を厳密に計算（高速 & 精度そのまま）
    """
    # (P,1)
    patch_norm = (patches ** 2).sum(dim=1, keepdim=True)
    # (1,M)
    mem_norm = (memory_bank_t ** 2).sum(dim=1).unsqueeze(0)

    # (P,M)
    dists_sq = patch_norm + mem_norm - 2.0 * patches @ memory_bank_t.T
    dists_sq = torch.clamp(dists_sq, min=0.0)
    dists = torch.sqrt(dists_sq)

    # 各パッチが最も近いメモリからの距離 → その最大値を anomaly score とする
    min_dists_per_patch, _ = torch.min(dists, dim=1)
    score = torch.max(min_dists_per_patch).item()
    return score


# ================================
# val 用：スコア & ラベル（高速版）
# ================================
def infer_scores(loader, dataset, model, memory_bank_t):
    scores = []
    labels = []

    class_to_bin = {
        dataset.class_to_idx[NORMAL_CLASS]: 0,
        dataset.class_to_idx[ABNORMAL_CLASS]: 1,
    }

    with torch.no_grad():
        idx = 0
        for imgs, labs in tqdm(loader, total=len(loader), desc="Infer val"):
            imgs = imgs.to(DEVICE)
            feats = model(imgs)
            flat = flatten_patches(feats)   # (B,P,D)

            B, P, D = flat.shape
            for b in range(B):
                patches = flat[b]           # (P,D) torch
                score = anomaly_score_torch(patches, memory_bank_t)
                scores.append(score)
                labels.append(class_to_bin[int(labs[b])])
                idx += 1

    return np.array(scores), np.array(labels)


# ================================
# test 用：スコア & ラベル & パス（高速版）
# ================================
def infer_scores_labels_and_paths(loader, dataset, model, memory_bank_t):
    scores = []
    labels = []
    paths = []

    class_to_bin = {
        dataset.class_to_idx[NORMAL_CLASS]: 0,
        dataset.class_to_idx[ABNORMAL_CLASS]: 1,
    }

    idx_ds = 0
    with torch.no_grad():
        for imgs, labs in tqdm(loader, total=len(loader), desc="Infer test"):
            imgs = imgs.to(DEVICE)
            feats = model(imgs)
            flat = flatten_patches(feats)   # (B,P,D)
            B, P, D = flat.shape

            for b in range(B):
                patches = flat[b]
                score = anomaly_score_torch(patches, memory_bank_t)
                scores.append(score)
                labels.append(class_to_bin[int(labs[b])])

                sample_path, _ = dataset.samples[idx_ds]
                paths.append(sample_path)
                idx_ds += 1

    return np.array(scores), np.array(labels), paths


# ================================
# メイン処理
# ================================
def main():
    # モデル
    model = PatchCoreExtractor().to(DEVICE)
    model.eval()

    # データセット & ローダ
    train_ds = datasets.ImageFolder(os.path.join(DATASET_ROOT, "train"), transform=transform)
    val_ds   = datasets.ImageFolder(os.path.join(DATASET_ROOT, "val"),   transform=transform)
    test_ds  = datasets.ImageFolder(os.path.join(DATASET_ROOT, "test"),  transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    # val/test は1枚ずつでもいいけど、バッチにしてもOK（ここでは1のままにしておく）
    val_loader   = DataLoader(val_ds,  batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    print("train classes:", train_ds.classes)
    print("val classes:  ", val_ds.classes)
    print("test classes: ", test_ds.classes)

    # 1) MemoryBank 構築
    memory_np = build_memory_bank(train_loader, model)
    memory_np = coreset_sampling(memory_np, CORESET_RATIO)

    # PyTorch Tensor (GPU) に乗せる
    memory_bank_t = torch.from_numpy(memory_np).to(DEVICE)

    # 2) Validation → 閾値決定
    print("\n=== Validation ===")
    val_scores, val_labels = infer_scores(val_loader, val_ds, model, memory_bank_t)
    fpr, tpr, thr = roc_curve(val_labels, val_scores)
    best_idx = np.argmax(tpr - fpr)
    threshold = thr[best_idx]
    print(f"Best threshold = {threshold:.4f}")

    # 3) Test → 評価
    print("\n=== Test ===")
    test_scores, test_labels, test_paths = infer_scores_labels_and_paths(
        test_loader, test_ds, model, memory_bank_t
    )
    pred = (test_scores >= threshold).astype(int)

    cm = confusion_matrix(test_labels, pred)
    print("Confusion Matrix (counts):\n", cm)

    print("\nClassification Report:")
    print(classification_report(
        test_labels, pred,
        target_names=["normal (safe)", "abnormal (danger)"]
    ))

    # 4) 正規化コンフマトの保存
    labels = ["normal", "abnormal"]
    cm_float = cm.astype(np.float64)
    row_sums = cm_float.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_float, row_sums, where=row_sums != 0) * 100.0

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
    ax.set_title("PatchCore (ResNet50)\nConfusion Matrix (row-normalized, %)")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.1f}",
                    ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(CONF_MAT_PATH, dpi=300)
    plt.close()
    print(f"Saved normalized confusion matrix as: {CONF_MAT_PATH}")

    # 5) 画像仕分け & CSV
    print("\n=== Saving images & CSV ===")
    normal_dir = os.path.join(OUTPUT_DIR, "normal")
    abnormal_dir = os.path.join(OUTPUT_DIR, "abnormal")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(abnormal_dir, exist_ok=True)

    inv_map = {0: "normal", 1: "abnormal"}

    rows = []
    for p, t, y, s in zip(test_paths, test_labels, pred, test_scores):
        fname = os.path.basename(p)
        dst_dir = normal_dir if y == 0 else abnormal_dir
        shutil.copy2(p, os.path.join(dst_dir, fname))

        rows.append([
            p,
            fname,
            inv_map[int(t)],
            inv_map[int(y)],
            s,
        ])

    df = pd.DataFrame(rows, columns=["filepath", "filename", "true", "pred", "anomaly_score"])
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved prediction table to: {CSV_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
