import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report

# =================================================
# ★ 設定：ここだけ自分の環境に合わせて変える
# =================================================
DATASET_ROOT = "/home/yamamao/Patchcore/dataset"

NORMAL_CLASS_NAME = "normal"
ABNORMAL_CLASS_NAME = "abnormal"

IMAGE_SIZE = 384
BATCH_SIZE = 8
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "patchcore_memorybank.pth"


# =================================================
# 前処理
# =================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# =================================================
# データセット
# =================================================
train_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, "val"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, "test"), transform=transform)

print("=== Class check ===")
print("train:", train_dataset.classes)
print("val:", val_dataset.classes)
print("test:", test_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# =================================================
# PatchCore モデル
# =================================================
from anomalib.models.image.patchcore import Patchcore

model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    pre_trained=True
).to(DEVICE)


# =================================================
# ★ 学習：feature_extractor を使う（forward 禁止）
# =================================================
print("\n=== Extracting train(normal) features ===")

train_features = []

model.eval()

with torch.no_grad():
    for imgs, _ in train_loader:
        imgs = imgs.to(DEVICE)

        # ★ 正しい特徴抽出ルート
        feat_dict = model.feature_extractor(imgs)
        # feat_dict = {"layer2": Tensor[B,C,H,W], "layer3": Tensor[...]}

        # 特徴マップを flatten して1ベクトルに
        feat_vecs = []
        for k in feat_dict:
            f = feat_dict[k]
            f = torch.flatten(f, 1)     # (B, C*H*W)
            feat_vecs.append(f)

        feats = torch.cat(feat_vecs, dim=1)   # (B, D)
        train_features.append(feats.cpu())

train_features = torch.cat(train_features, dim=0)
print("Extracted train features:", train_features.shape)

print("\n=== Building memory bank ===")
model.fit(train_features)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Memory bank saved:", MODEL_SAVE_PATH)


# =================================================
# ★ 推論（predict API 使用）
# =================================================
def get_scores_and_labels(loader, dataset):
    scores = []
    labels = []

    class_to_bin = {
        dataset.class_to_idx[NORMAL_CLASS_NAME]: 0,
        dataset.class_to_idx[ABNORMAL_CLASS_NAME]: 1,
    }

    model.eval()
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(DEVICE)

            # ★ PatchCore 推論の正規ルート
            pred = model.predict(imgs)

            # pred は Tensor（B,）として anomaly_score を返す
            anomaly_scores = pred.cpu().numpy()

            scores.extend(anomaly_scores)
            labels.extend([class_to_bin[int(l)] for l in labs])

    return np.array(scores), np.array(labels)


# =================================================
# ★ 閾値（val）
# =================================================
print("\n=== Validation: threshold search ===")
val_scores, val_labels = get_scores_and_labels(val_loader, val_dataset)

fpr, tpr, thr = roc_curve(val_labels, val_scores)
best_idx = np.argmax(tpr - fpr)
best_threshold = thr[best_idx]

print(f"Best threshold = {best_threshold:.4f}")


# =================================================
# ★ テスト評価
# =================================================
print("\n=== Test Evaluation ===")
test_scores, test_labels = get_scores_and_labels(test_loader, test_dataset)
pred_labels = (test_scores >= best_threshold).astype(int)

cm = confusion_matrix(test_labels, pred_labels)
print("Confusion matrix:\n", cm)

print("\nClassification report:")
print(classification_report(
    test_labels, pred_labels,
    target_names=["normal (safe)", "abnormal (danger)"]
))

disp = ConfusionMatrixDisplay(cm, display_labels=["normal", "abnormal"])
disp.plot()
plt.title("PatchCore Confusion Matrix")
plt.tight_layout()
plt.show()
