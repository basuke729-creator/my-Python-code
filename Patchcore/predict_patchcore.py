# predict_patchcore.py
# 学習済み Patchcore モデルで test データを推論し、
# confusion matrix と results.csv を出力するスクリプト

import os
import numpy as np
import torch
from pytorch_lightning import Trainer

from anomalib.data import Folder
from anomalib.models.image.patchcore import PatchcoreLightning

from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ========= ここだけ環境に合わせて確認 =========
DATA_ROOT = "/home/yamamao/Patchcore/dataset"
MODEL_CKPT = "/home/yamamao/Patchcore/results/Patchcore/ladder_dataset/v23/weights/Lightning/model.ckpt"
OUT_DIR = "/home/yamamao/Patchcore/test_results"
# ===========================================

os.makedirs(OUT_DIR, exist_ok=True)


def build_datamodule() -> Folder:
    """anomalib.data.Folder を使って datamodule を作成（train と同じ設定）"""
    datamodule = Folder(
        name="ladder_dataset",
        root=DATA_ROOT,
        normal_dir="train/normal",
        abnormal_dir="train/abnormal",
        normal_test_dir="test/normal",
        abnormal_test_dir="test/abnormal",
        normal_split_ratio=0.0,
        train_batch_size=32,
        test_batch_size=32,
        num_workers=8,
    )
    datamodule.setup()  # train/val/test を準備（今回は test だけ使う）
    return datamodule


def main():
    print("[INFO] Building datamodule...")
    dm = build_datamodule()

    print("[INFO] Loading model from checkpoint...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: PatchcoreLightning = PatchcoreLightning.load_from_checkpoint(
        MODEL_CKPT,
        map_location=device,
    )
    model.eval()

    trainer = Trainer(accelerator="auto", logger=False, enable_checkpointing=False)

    print("[INFO] Running prediction on test set...")
    # trainer.predict の戻り値は「バッチごとの dict のリスト」
    pred_batches = trainer.predict(model=model, datamodule=dm)

    y_true_list = []
    y_score_list = []

    for batch_out in pred_batches:
        # anomalib の PatchcoreLightning は image-level のスコアを pred_score に入れてくる想定
        if "pred_score" in batch_out:
            scores = batch_out["pred_score"]
        elif "image_score" in batch_out:
            scores = batch_out["image_score"]
        else:
            raise KeyError(f"pred_score / image_score が見つかりません: {batch_out.keys()}")

        labels = batch_out["label"]

        y_true_list.append(labels.detach().cpu().numpy())
        y_score_list.append(scores.detach().cpu().numpy())

    y_true = np.concatenate(y_true_list)   # 0=normal, 1=abnormal
    y_score = np.concatenate(y_score_list)

    # ====== スコア → 予測ラベルへの変換（閾値） ======
    try:
        # anomalib の image_threshold（バージョンによって value の有無が違うので両対応）
        thr = float(getattr(model.image_threshold, "value", model.image_threshold))
    except Exception:
        # 万一取れなければ ROC 的な意味で 0.5 を仮の閾値とする
        thr = 0.5
    print(f"[INFO] Using image threshold: {thr}")

    y_pred = (y_score > thr).astype(int)

    # ====== confusion matrix とレポート ======
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("[INFO] Confusion matrix (normal=0, abnormal=1):")
    print(cm)
    print()
    print("[INFO] Classification report:")
    print(classification_report(y_true, y_pred, target_names=["normal", "abnormal"]))

    # ====== CSV 保存 ======
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "score": y_score,
            "y_pred": y_pred,
        }
    )
    csv_path = os.path.join(OUT_DIR, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved results.csv to: {csv_path}")

    # ====== confusion matrix 画像保存 ======
    plt.figure(figsize=(5, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["normal", "abnormal"],
        yticklabels=["normal", "abnormal"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (image-level)")
    plt.tight_layout()

    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[INFO] Saved confusion matrix image to: {cm_path}")


if __name__ == "__main__":
    main()
 