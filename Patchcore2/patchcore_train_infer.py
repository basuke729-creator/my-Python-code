import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from tqdm import tqdm

# ================================
#  基本設定
# ================================
DATASET_ROOT = "/home/yamamao/Patchcore/dataset"
NORMAL_CLASS = "normal"
ABNORMAL_CLASS = "abnormal"

IMAGE_SIZE = 384
BATCH_SIZE = 8
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# coreset は使わない（精度優先＆高速化）
USE_CORESET = False       # True にすると Greedy coreset を使えるが激遅
CORESET_RATIO = 0.01      # coreset を使う場合の割合

CONF_MAT_PATH = "confusion_matrix_norm.png"
CSV_PATH = "predictions.csv"

# 画像保存用ディレクトリ
RESULTS_PRED_DIR = "results_pred"      # 予測ラベルごと（元画像）
RESULTS_SPLIT_DIR = "results_split"    # TP/TN/FP/FN ごと（元画像）

# ヒートマップを「予測 normal / abnormal」で保存
HEATMAP_PRED_DIR = "heatmaps_pred"     # heatmaps_pred/normal, heatmaps_pred/abnormal

# ================================
#  再現性用のシード
# ================================
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================================
#  前処理
# ================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ================================
#  PatchCore 用 ResNet50 特徴抽出器
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
#  layer2 を layer3 のサイズに合わせてパッチ化
# ================================
def flatten_patches(feat_dict):
    """
    feat_dict = {"layer2": (B,C2,H2,W2), "layer3": (B,C3,H3,W3)}
      -> layer2 を layer3 の空間サイズにアップサンプルしてチャンネル結合
      -> (B, H3*W3, C2+C3), 併せて (H3,W3) も返す
    """
    f2 = feat_dict["layer2"]
    f3 = feat_dict["layer3"]

    B, C2, H2, W2 = f2.shape
    B, C3, H3, W3 = f3.shape

    # layer2 → layer3 のサイズにアップサンプル
    f2_up = F.interpolate(f2, size=(H3, W3), mode="bilinear", align_corners=False)

    # チャンネル方向で結合 → (B, C2+C3, H3, W3)
    feat = torch.cat([f2_up, f3], dim=1)

    # (B, H3*W3, C_total) へ変換
    feat = feat.permute(0, 2, 3, 1)          # (B,H3,W3,C)
    feat = feat.reshape(B, H3 * W3, C2 + C3) # (B,P,D)

    return feat, (H3, W3)

# ================================
#  MemoryBank 構築（train/normal のみ）
# ================================
def build_memory_bank(loader, model):
    memory = []

    print("\n=== Building MemoryBank ===")
    with torch.no_grad():
        for imgs, _ in tqdm(loader, total=len(loader),
                            desc="Extracting train(normal) patches"):
            imgs = imgs.to(DEVICE)
            feats = model(imgs)
            flat, _ = flatten_patches(feats)     # (B,P,D)
            flat = flat.cpu().numpy()
            memory.append(flat)

    memory = np.concatenate(memory, axis=0)       # (N,P,D)
    N, P, D = memory.shape
    memory = memory.reshape(N * P, D)             # (N*P,D)

    print("MemoryBank shape (no coreset yet) =", memory.shape)
    return memory

# ================================
#  （オプション）coreset k-center
# ================================
def coreset_sampling(memory, ratio=0.01):
    n = memory.shape[0]
    m = max(1, int(n * ratio))
    if m >= n:
        print("Coreset skipped (m>=n).")
        return memory

    print(f"Running coreset sampling (ratio={ratio}, select {m}/{n}) ...")

    selected = [0]  # 再現性のため固定開始
    distances = np.linalg.norm(memory - memory[selected[-1]], axis=1)

    for _ in tqdm(range(m - 1), desc="Coreset k-center"):
        idx = np.argmax(distances)
        selected.append(idx)
        new_dist = np.linalg.norm(memory - memory[idx], axis=1)
        distances = np.minimum(distances, new_dist)

    sampled = memory[selected]
    print(f"Coreset sampled: {len(selected)}/{n}")
    print("Final MemoryBank (coreset) =", sampled.shape)
    return sampled

# ================================
#  厳密 kNN 距離＋各パッチの最小距離
# ================================
def anomaly_score_and_min_dists(patches, memory_bank_t, chunk_size=50000):
    """
    patches: (P,D) torch.Tensor (DEVICE)
    memory_bank_t: (M,D) torch.Tensor (DEVICE)

    return:
      score: scalar (max of per-patch min-dist)
      min_dists_per_patch: (P,) torch.Tensor (DEVICE)
    """
    P, D = patches.shape
    M = memory_bank_t.shape[0]

    patch_norm = (patches ** 2).sum(dim=1, keepdim=True)   # (P,1)
    min_dists_per_patch = None

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        mb_chunk = memory_bank_t[start:end]                # (m,D)

        mem_norm = (mb_chunk ** 2).sum(dim=1).unsqueeze(0) # (1,m)
        dists_sq = patch_norm + mem_norm - 2.0 * patches @ mb_chunk.T
        dists_sq = torch.clamp(dists_sq, min=0.0)
        dists = torch.sqrt(dists_sq)                       # (P,m)

        cur_min, _ = torch.min(dists, dim=1)               # (P,)

        if min_dists_per_patch is None:
            min_dists_per_patch = cur_min
        else:
            min_dists_per_patch = torch.minimum(min_dists_per_patch, cur_min)

    score = torch.max(min_dists_per_patch).item()
    return score, min_dists_per_patch

# ================================
#  Validation 用推論（スコアのみ）
# ================================
def infer_scores(loader, dataset, model, memory_bank_t):
    scores = []
    labels = []

    class_to_bin = {
        dataset.class_to_idx[NORMAL_CLASS]: 0,
        dataset.class_to_idx[ABNORMAL_CLASS]: 1,
    }

    with torch.no_grad():
        for imgs, labs in tqdm(loader, total=len(loader), desc="Infer val"):
            imgs = imgs.to(DEVICE)
            feats = model(imgs)
            flat, _ = flatten_patches(feats)  # (B,P,D)

            B, P, D = flat.shape
            for b in range(B):
                patches = flat[b]          # (P,D)
                score, _ = anomaly_score_and_min_dists(patches, memory_bank_t)
                scores.append(score)
                labels.append(class_to_bin[int(labs[b])])

    return np.array(scores), np.array(labels)

# ================================
#  Test 用推論（スコア＋ラベル＋パス）
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
            flat, _ = flatten_patches(feats)  # (B,P,D)
            B, P, D = flat.shape

            for b in range(B):
                patches = flat[b]
                score, _ = anomaly_score_and_min_dists(patches, memory_bank_t)
                scores.append(score)
                labels.append(class_to_bin[int(labs[b])])

                sample_path, _ = dataset.samples[idx_ds]
                paths.append(sample_path)
                idx_ds += 1

    return np.array(scores), np.array(labels), paths

# ================================
#  Val 上で F1 が最大となるしきい値を探す
# ================================
def choose_best_threshold(val_scores, val_labels, metric="f1"):
    thresholds = np.unique(val_scores)
    best_thr = thresholds[0]
    best_score = -1.0

    for thr in thresholds:
        pred = (val_scores >= thr).astype(int)
        if metric == "f1":
            m = f1_score(val_labels, pred)
        elif metric == "acc":
            m = accuracy_score(val_labels, pred)
        else:
            m = f1_score(val_labels, pred)

        if m > best_score:
            best_score = m
            best_thr = thr

    return best_thr, best_score

# ================================
#  ★ここが 3 パネル版ヒートマップ★
# ================================
def save_anomaly_heatmap(img_tensor, anomaly_map, out_path, mask_thr=0.7):
    """
    img_tensor : (C,H,W) [0,1] Tensor
    anomaly_map: (h,w) numpy or tensor
    mask_thr   : 異常マスクを作るときのしきい値（0〜1）
    """
    # 元画像
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)   # (H,W,C)
    img_np = np.clip(img_np, 0, 1)

    # anomaly map を numpy 化
    if isinstance(anomaly_map, torch.Tensor):
        amap = anomaly_map.detach().cpu().numpy()
    else:
        amap = anomaly_map

    # (h,w) -> IMAGE_SIZE にアップサンプル
    amap_t = torch.from_numpy(amap)[None, None, :, :].float()
    amap_up = F.interpolate(
        amap_t,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    # 0〜1に正規化
    amap_up = amap_up - amap_up.min()
    if amap_up.max() > 0:
        amap_up = amap_up / amap_up.max()

    # 3 パネルの図を作成
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 左：元画像
    axes[0].imshow(img_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    # 中央：元画像 + ヒートマップ
    axes[1].imshow(img_np)
    axes[1].imshow(amap_up, cmap="jet", alpha=0.5)
    axes[1].set_title("Image + Anomaly Map")
    axes[1].axis("off")

    # 右：元画像 + Pred Mask（輪郭）
    axes[2].imshow(img_np)
    # しきい値 mask_thr で等高線を描く
    axes[2].contour(
        amap_up,
        levels=[mask_thr],
        colors="red",
        linewidths=2,
    )
    axes[2].set_title("Image + Pred Mask")
    axes[2].axis("off")

    plt.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

# ================================
#  テスト画像の仕分け＆ヒートマップ生成
# ================================
def save_split_and_heatmaps(test_ds,
                            model,
                            memory_bank_t,
                            test_labels,
                            pred,
                            test_scores,
                            test_paths):
    """
    - 予測ラベルごとに results_pred/normal, abnormal へコピー（元画像）
    - TP/TN/FP/FN ごとに results_split/TP.. へコピー（元画像）
    - ヒートマップは予測ラベルごとに heatmaps_pred/normal, abnormal へ保存
    """
    # 予測ラベルごとの元画像
    os.makedirs(RESULTS_PRED_DIR, exist_ok=True)
    pred_normal_dir = os.path.join(RESULTS_PRED_DIR, "normal")
    pred_abnormal_dir = os.path.join(RESULTS_PRED_DIR, "abnormal")
    os.makedirs(pred_normal_dir, exist_ok=True)
    os.makedirs(pred_abnormal_dir, exist_ok=True)

    # TP/TN/FP/FN ごとの元画像
    os.makedirs(RESULTS_SPLIT_DIR, exist_ok=True)
    split_dirs = {}
    for name in ["TP", "TN", "FP", "FN"]:
        d_split = os.path.join(RESULTS_SPLIT_DIR, name)
        os.makedirs(d_split, exist_ok=True)
        split_dirs[name] = d_split

    # 予測ラベルごとのヒートマップ
    os.makedirs(HEATMAP_PRED_DIR, exist_ok=True)
    heat_pred_normal = os.path.join(HEATMAP_PRED_DIR, "normal")
    heat_pred_abnormal = os.path.join(HEATMAP_PRED_DIR, "abnormal")
    os.makedirs(heat_pred_normal, exist_ok=True)
    os.makedirs(heat_pred_abnormal, exist_ok=True)

    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    print("\n=== Saving split images & heatmaps ===")
    with torch.no_grad():
        idx = 0
        for imgs, labs in tqdm(loader, total=len(loader), desc="Save & heatmap"):
            img = imgs[0]  # (C,H,W)
            true_bin = int(test_labels[idx])
            pred_bin = int(pred[idx])
            path = test_paths[idx]
            fname = os.path.basename(path)

            # 予測ラベルごとの元画像
            dst_pred_dir = pred_normal_dir if pred_bin == 0 else pred_abnormal_dir
            shutil.copy2(path, os.path.join(dst_pred_dir, fname))

            # TP/TN/FP/FN の判定（元画像用）
            if true_bin == 0 and pred_bin == 0:
                group = "TN"
            elif true_bin == 0 and pred_bin == 1:
                group = "FP"
            elif true_bin == 1 and pred_bin == 0:
                group = "FN"
            else:
                group = "TP"
            shutil.copy2(path, os.path.join(split_dirs[group], fname))

            # ヒートマップ生成（予測ラベルごと）
            imgs_gpu = imgs.to(DEVICE)
            feats = model(imgs_gpu)
            flat, (H3, W3) = flatten_patches(feats)
            patches = flat[0]  # (P,D)
            _, min_dists = anomaly_score_and_min_dists(patches, memory_bank_t)
            amap = min_dists.reshape(H3, W3)  # (H3,W3)

            dst_heat_dir = heat_pred_normal if pred_bin == 0 else heat_pred_abnormal
            heat_path = os.path.join(dst_heat_dir, fname)
            save_anomaly_heatmap(img, amap, heat_path, mask_thr=0.7)

            idx += 1

# ================================
#  メイン
# ================================
def main():
    model = PatchCoreExtractor().to(DEVICE)
    model.eval()

    train_ds = datasets.ImageFolder(os.path.join(DATASET_ROOT, "train"), transform=transform)
    val_ds   = datasets.ImageFolder(os.path.join(DATASET_ROOT, "val"),   transform=transform)
    test_ds  = datasets.ImageFolder(os.path.join(DATASET_ROOT, "test"),  transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,  batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    print("train classes:", train_ds.classes)
    print("val classes:  ", val_ds.classes)
    print("test classes: ", test_ds.classes)

    # 1) MemoryBank 構築
    memory_np = build_memory_bank(train_loader, model)

    if USE_CORESET:
        memory_np = coreset_sampling(memory_np, CORESET_RATIO)
    else:
        print("Skipping coreset. Using full MemoryBank for maximum accuracy.")

    memory_bank_t = torch.from_numpy(memory_np).to(DEVICE)

    # 2) Validation でベストしきい値（F1最大）を決定
    print("\n=== Validation ===")
    val_scores, val_labels = infer_scores(val_loader, val_ds, model, memory_bank_t)
    threshold, best_f1 = choose_best_threshold(val_scores, val_labels, metric="f1")
    print(f"Best threshold (by F1) = {threshold:.6f}")
    print(f"Best F1 on val = {best_f1:.4f}")

    val_pred = (val_scores >= threshold).astype(int)
    val_acc = accuracy_score(val_labels, val_pred)
    print(f"Val accuracy at best F1 threshold = {val_acc:.4f}")

    # 3) Test 評価
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

    # 4) 正規化コンフマト画像の保存（％表示）
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
    ax.set_title("PatchCore Confusion Matrix (row-normalized, %)")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.1f}",
                    ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(CONF_MAT_PATH, dpi=300)
    plt.close()
    print(f"Saved normalized confusion matrix as: {CONF_MAT_PATH}")

    # 5) CSV 出力
    print("\n=== Saving CSV ===")
    inv_map = {0: "normal", 1: "abnormal"}
    rows = []
    for p, t, y, s in zip(test_paths, test_labels, pred, test_scores):
        fname = os.path.basename(p)
        rows.append([
            p,
            fname,
            inv_map[int(t)],   # true
            inv_map[int(y)],   # pred
            s,                 # anomaly score
        ])

    df = pd.DataFrame(rows, columns=["filepath", "filename", "true", "pred", "anomaly_score"])
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved prediction table to: {CSV_PATH}")

    # 6) テスト画像の仕分け＆ヒートマップ生成
    save_split_and_heatmaps(test_ds, model, memory_bank_t,
                            test_labels, pred, test_scores, test_paths)

    print("Done.")


if __name__ == "__main__":
    main()

