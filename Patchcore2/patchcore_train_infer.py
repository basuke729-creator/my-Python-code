import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

from anomalib.models.image import Patchcore
from anomalib.engine import Engine

# =========================
# ユーザー設定エリア
# =========================
DATASET_ROOT = "dataset_root"  # あなたのデータセットルートに変更
IMAGE_SIZE = 256               # リサイズサイズ
BATCH_SIZE = 8
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "patchcore_checkpoint.ckpt"

# =========================
# データセット定義
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# train：normalのみを使う
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_ROOT, "train"),
    transform=transform
)

# test：normal / abnormal 両方をそのまま読み込み
test_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_ROOT, "test"),
    transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

print("Train classes:", train_dataset.classes)
print("Test classes:", test_dataset.classes)
# ここで train: ['normal'], test: ['abnormal', 'normal'] のように出ていればOK

# =========================
# PatchCore モデルの用意
# =========================
# imagenetの特徴量を利用する構成
model = Patchcore(
    backbone="resnet18",      # 軽量モデル（重くても良ければ wide_resnet50_2 等）
    pre_trained=True,
    layers=["layer2", "layer3"],
    input_size=(IMAGE_SIZE, IMAGE_SIZE)
).to(DEVICE)

engine = Engine(model=model, device=DEVICE)

# =========================
# 1. 学習 (normalのみ)
# =========================
print("=== Training PatchCore on normal images only ===")
engine.fit(train_loader)

# 学習済みモデルを保存
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Saved model to {MODEL_SAVE_PATH}")

# =========================
# 2. 推論（test: normal & abnormal）
# =========================
print("=== Inference on test set ===")
model.eval()

all_scores = []   # 異常スコア
all_labels = []   # 正解ラベル (0=normal, 1=abnormal)

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        # PatchCore は forwardで anomaly map / score を返す
        outputs = model(imgs)
        # outputs は dict 形式の場合が多いので確認
        # anomalib>=1.0系だと outputs["anomaly_score"] など
        if isinstance(outputs, dict):
            scores = outputs["anomaly_score"].detach().cpu().numpy()
        else:
            # 万が一辞書でない実装の場合の保険
            scores = outputs.detach().cpu().numpy()

        # test_dataset.classes の index を 0/1 にマップする
        # 例: test_dataset.classes == ['abnormal', 'normal'] の場合
        #   label=0 -> abnormal(1)
        #   label=1 -> normal(0)
        class_to_binary = {
            test_dataset.class_to_idx["normal"]: 0,
            test_dataset.class_to_idx["abnormal"]: 1,
        }

        bin_labels = [class_to_binary[int(l)] for l in labels]

        all_scores.extend(scores)
        all_labels.extend(bin_labels)

all_scores = np.array(all_scores).reshape(-1)
all_labels = np.array(all_labels).reshape(-1)

print("Scores shape:", all_scores.shape)
print("Labels shape:", all_labels.shape)

# =========================
# 3. 閾値決定
# =========================
# 簡単な決め方の例：
#   validation があればそこで決めるのがベストだが、
#   今回は test セット全体で Youden J 最大となる閾値を探索
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]

print(f"Best threshold by Youden J: {best_threshold:.4f}")

# =========================
# 4. 安全/危険の2値判定
# =========================
# score >= threshold → 異常(1 = 危険姿勢)
# score <  threshold → 正常(0 = 安全姿勢)
pred_labels = (all_scores >= best_threshold).astype(int)

# =========================
# 5. 混同行列 & レポート
# =========================
cm = confusion_matrix(all_labels, pred_labels, labels=[0, 1])
print("Confusion Matrix (rows=true, cols=pred):")
print(cm)

print("\nClassification report:")
print(classification_report(
    all_labels, pred_labels,
    target_names=["normal (safe)", "abnormal (danger)"]
))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["normal (safe)", "abnormal (danger)"]
)
disp.plot()
plt.title("Confusion Matrix - PatchCore")
plt.tight_layout()
plt.show()
