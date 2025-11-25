import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
)
import numpy as np

# anomalib 関連
from anomalib.models.image import Patchcore
from anomalib.engine import Engine

# =========================================================
# ★★ ここだけ書き換えれば基本 OK という設定ゾーン ★★
# =========================================================
# データセットのルートフォルダ
DATASET_ROOT = r"./dataset_root"  # ← あなたの環境に合わせてパスを変更

# クラス名（フォルダ名）をここで定義
NORMAL_CLASS_NAME = "normal"      # 安全クラスのフォルダ名
ABNORMAL_CLASS_NAME = "abnormal"  # 危険クラスのフォルダ名

IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "patchcore_checkpoint.ckpt"
# =========================================================


# =========================
# 変換（前処理）
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =========================
# データセット作成
# =========================
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

# 【重要】フォルダ名とクラス名が合っているか確認
assert NORMAL_CLASS_NAME in val_dataset.classes, f"{NORMAL_CLASS_NAME} が val にありません"
assert ABNORMAL_CLASS_NAME in val_dataset.classes, f"{ABNORMAL_CLASS_NAME} が val にありません"

# torch.utils.data.DataLoader
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# =========================
# PatchCore モデルの用意
# =========================
model = Patchcore(
    backbone="resnet18",
    pre_trained=True,
    layers=["layer2", "layer3"],
    input_size=(IMAGE_SIZE, IMAGE_SIZE),
).to(DEVICE)

engine = Engine(model=model, device=DEVICE)

# =========================
# 1. 学習（train: normal のみを使用）
# =========================
# PatchCore 的には "normal" しか学習しないのが前提なので、
# train データは normal のみを入れておく運用にしておくのがベスト。
print("=== Training PatchCore on train/normal ===")
engine.fit(train_loader)

# モデル保存
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Saved model to {MODEL_SAVE_PATH}")


# =========================
# ヘルパー：クラスID → 0/1（normal/abnormal）変換
# =========================
def make_class_to_binary_map(dataset):
    """ImageFolder の class_to_idx から 0/1 ラベル用の dict を作る"""
    c2i = dataset.class_to_idx
    return {
        c2i[NORMAL_CLASS_NAME]: 0,
        c2i[ABNORMAL_CLASS_NAME]: 1,
    }


def get_scores_and_labels(data_loader, dataset, model, device):
    """指定の DataLoader から異常スコアと 0/1 ラベルを取得"""
    model.eval()
    all_scores = []
    all_labels = []

    class_to_binary = make_class_to_binary_map(dataset)

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)

            # anomalib の Patchcore は一般的に dict を返す想定
            if isinstance(outputs, dict):
                scores = outputs["anomaly_score"].detach().cpu().numpy()
            else:
                scores = outputs.detach().cpu().numpy()

            bin_labels = [class_to_binary[int(l)] for l in labels]

            all_scores.extend(scores)
            all_labels.extend(bin_labels)

    all_scores = np.array(all_scores).reshape(-1)
    all_labels = np.array(all_labels).reshape(-1)
    return all_scores, all_labels


# =========================
# 2. val で閾値を決定
# =========================
print("=== Inference on validation set (for threshold) ===")
val_scores, val_labels = get_scores_and_labels(val_loader, val_dataset, model, DEVICE)

print("Val scores shape:", val_scores.shape)
print("Val labels shape:", val_labels.shape)

fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]

print(f"Best threshold by Youden J on val: {best_threshold:.4f}")


# =========================
# 3. test で最終評価（混同行列）
# =========================
print("=== Inference on test set (final evaluation) ===")
test_scores, test_labels = get_scores_and_labels(test_loader, test_dataset, model, DEVICE)

pred_labels = (test_scores >= best_threshold).astype(int)

cm = confusion_matrix(test_labels, pred_labels, labels=[0, 1])
print("Confusion Matrix (rows=true, cols=pred):")
print(cm)

print("\nClassification report:")
print(classification_report(
    test_labels, pred_labels,
    target_names=["normal (safe)", "abnormal (danger)"]
))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["normal (safe)", "abnormal (danger)"]
)
disp.plot()
plt.title("Confusion Matrix - PatchCore (test set)")
plt.tight_layout()
plt.show()

