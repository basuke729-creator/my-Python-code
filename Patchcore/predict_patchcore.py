import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from anomalib.data.utils import read_image


# ==========================================
# 1) ここをあなたの環境に変更
# ==========================================
MODEL_CKPT = "/home/yamamao/Patchcore/results/Patchcore/ladder_dataset/v23/weights/lightning/model.ckpt"

DATASET_ROOT = "/home/yamamao/Patchcore/dataset/test"
NORMAL_DIR = os.path.join(DATASET_ROOT, "normal")
ABNORMAL_DIR = os.path.join(DATASET_ROOT, "abnormal")

OUTPUT_DIR = "/home/yamamao/Patchcore/predict_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ==========================================


# ==============================
# 2) 推論モデルの読み込み
# ==============================
print("[INFO] Loading model...")
model = PatchcoreModel.load_from_checkpoint(MODEL_CKPT)
model.eval().cuda()


# ==============================
# 3) 前処理
# ==============================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ==============================
# 4) 画像を読み推論する関数
# ==============================
def infer(img_path):
    img = read_image(img_path)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        outputs = model(img)

    score = outputs["pred_scores"].item()
    label = 1 if score > 0.5 else 0  # 0: normal, 1: abnormal
    return label, score


# ==============================
# 5) データセットを推論
# ==============================
y_true = []
y_pred = []

def run_inference(folder, true_label):
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(("jpg", "jpeg", "png")):
            continue

        path = os.path.join(folder, f)
        pred, score = infer(path)

        y_true.append(true_label)
        y_pred.append(pred)

        print(f"{path} → pred={pred}, score={score:.4f}")


print("[INFO] Running inference (normal)...")
run_inference(NORMAL_DIR, 0)

print("[INFO] Running inference (abnormal)...")
run_inference(ABNORMAL_DIR, 1)


# ==============================
# 6) Confusion Matrix を作成
# ==============================
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_true, y_pred)
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Normal", "Abnormal"],
            yticklabels=["Normal", "Abnormal"],
            cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()


# ==============================
# 7) classification report
# ==============================
report = classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"])
print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

print(f"[INFO] Results saved to: {OUTPUT_DIR}")
