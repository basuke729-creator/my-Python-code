import json, os, csv
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === 入力: 元データと推論出力のパスを環境に合わせて ===
DATA_ROOT = Path("/home/yamamao/Patchcore/dataset/test")  # test のGT
PRED_ROOT = Path("/home/yamamao/Patchcore/pred_results/predict")  # predict の出力
OUT_PNG   = Path("/home/yamamao/Patchcore/pred_results/confusion_matrix.png")

# GT 取得（ファイル名 -> ラベル）
gt = {}
for cls in ["normal","abnormal"]:
    for p in (DATA_ROOT/cls).glob("*.jpg"):
        gt[p.name] = cls
    for p in (DATA_ROOT/cls).glob("*.png"):
        gt[p.name] = cls

# 予測ラベル取得（ファイル名 -> ラベル）
# ※ anomalib predict の出力が images/<pred_cls>/ に分かれている前提
pred = {}
for cls in ["normal","abnormal"]:
    d = PRED_ROOT / "images" / cls
    if d.exists():
        for p in d.glob("*.*"):
            pred[p.name] = cls

# マッチするファイルだけで評価
y_true, y_pred = [], []
for fname, y in gt.items():
    if fname in pred:
        y_true.append(0 if y=="normal" else 1)
        y_pred.append(0 if pred[fname]=="normal" else 1)

# 混同行列
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(cm, display_labels=["normal","abnormal"])
disp.plot(values_format="d")
plt.title("Confusion Matrix (image-level)")
plt.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=180)
print(f"saved: {OUT_PNG}")
