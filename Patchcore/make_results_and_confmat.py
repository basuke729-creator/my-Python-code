#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学習済み PatchCore モデルを使って test データに対して推論を行い、
- results.csv（image_path, y_true, y_pred, score）
- confusion_matrix_patchcore.png
- metrics_report.txt
を作成するスクリプト。

前提:
- anomalib がインストール済み（train できているのでOK）
- patchcore_cls.yaml で学習済み
"""

from pathlib import Path
import inspect
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import torch
from lightning.pytorch import Trainer

from anomalib.data import Folder

# Patchcore クラス名の違いに両対応
try:
    from anomalib.models.image.patchcore import PatchcoreLightning as PatchcoreModel
except Exception:
    from anomalib.models.image.patchcore import Patchcore as PatchcoreModel


# ======== ここだけ自分の環境に合わせて修正してください ========
CONFIG_PATH = Path("/home/yamamao/Patchcore/patchcore_cls.yaml")
CKPT_PATH   = Path("/home/yamamao/Patchcore/results/patchcore/ladder_dataset/version_0/weights/last.ckpt")
OUT_DIR     = Path("/home/yamamao/Patchcore/test_results")
# =============================================================


def load_data_module_from_yaml(config_path: Path) -> Folder:
    """YAML の data.init_args をそのまま Folder に渡す"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]["init_args"]

    # この辺のキーは Folder が受け付けないバージョンもあるので、念のため落とす
    sig = inspect.signature(Folder.__init__)
    valid_keys = sig.parameters.keys()
    clean_cfg = {k: v for k, v in data_cfg.items() if k in valid_keys}

    print("[INFO] Folder init args (filtered):", clean_cfg)
    dm = Folder(**clean_cfg)
    return dm


def load_model(ckpt_path: Path) -> PatchcoreModel:
    print(f"[INFO] Loading model from ckpt: {ckpt_path}")
    model = PatchcoreModel.load_from_checkpoint(str(ckpt_path))
    model.eval()
    return model


def run_predict(model: PatchcoreModel, dm: Folder):
    """Trainer.predict で test データに対して推論"""
    trainer = Trainer(accelerator="auto", logger=False, enable_checkpointing=False)
    preds = trainer.predict(model=model, dataloaders=dm.test_dataloader())
    # preds はバッチごとの list。各要素が dict であることを期待。
    flat = []
    for batch_out in preds:
        if isinstance(batch_out, dict):
            flat.append(batch_out)
        elif isinstance(batch_out, (list, tuple)):
            # バッチ内の複数 dict
            for o in batch_out:
                flat.append(o)
        else:
            print("[WARN] 予期しない出力形式:", type(batch_out))
    return flat


def extract_rows(pred_outputs, out_dir: Path):
    """
    PatchCore の出力 dict から
    image_path, y_true, y_pred, score を取り出して DataFrame にする。
    バージョン差に対応するため、キーは動的に探す。
    """
    rows = []

    # まず、キー候補を推定
    if not pred_outputs:
        raise RuntimeError("predict 出力が空です。test データが 0 枚の可能性があります。")

    sample = pred_outputs[0]
    if not isinstance(sample, dict):
        raise RuntimeError(f"predict の要素が dict ではありません: {type(sample)}")

    print("[INFO] predict output sample keys:", sample.keys())

    # 典型的なキー候補
    path_keys = ["image_path", "path", "image_name"]
    label_keys = ["label", "label_index", "gt_label", "y_true"]
    pred_keys = ["pred_label", "prediction", "y_pred"]
    score_keys = ["score", "anomaly_score", "pred_score"]

    def pick_key(candidates):
        for k in candidates:
            if k in sample:
                return k
        return None

    path_key = pick_key(path_keys)
    label_key = pick_key(label_keys)
    pred_key = pick_key(pred_keys)
    score_key = pick_key(score_keys)

    if path_key is None or label_key is None or pred_key is None:
        raise RuntimeError(
            f"必要なキーが見つかりませんでした。\n"
            f"sample keys={list(sample.keys())}\n"
            f"path_key={path_key}, label_key={label_key}, pred_key={pred_key}"
        )

    print(f"[INFO] 使用するキー -> path:{path_key}, label:{label_key}, pred:{pred_key}, score:{score_key}")

    for out in pred_outputs:
        # バッチ出力が dict の想定
        paths = out[path_key]
        labels = out[label_key]
        preds = out[pred_key]
        scores = out[score_key] if score_key is not None and score_key in out else None

        # tensor → numpy → list
        def to_list(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().tolist()
            elif isinstance(x, (np.ndarray, list, tuple)):
                return list(x)
            else:
                return [x]

        paths = to_list(paths)
        labels = to_list(labels)
        preds = to_list(preds)
        if scores is not None:
            scores = to_list(scores)
        else:
            scores = [None] * len(paths)

        for p, y, yhat, s in zip(paths, labels, preds, scores):
            rows.append(
                {
                    "image_path": p,
                    "y_true": int(y) if isinstance(y, (int, np.integer)) or str(y).isdigit() else y,
                    "y_pred": int(yhat) if isinstance(yhat, (int, np.integer)) or str(yhat).isdigit() else yhat,
                    "score": s,
                }
            )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] results.csv を保存しました: {csv_path}")
    return df, csv_path


def save_confusion_matrix_and_report(df: pd.DataFrame, out_dir: Path):
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    labels = sorted(list(set(list(y_true) + list(y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("[INFO] Confusion matrix:\n", cm)

    # 画像保存
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Patchcore)",
    )

    # 数値をセルに表示
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center")

    fig.tight_layout()
    png_path = out_dir / "confusion_matrix_patchcore.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] confusion_matrix_patchcore.png を保存しました: {png_path}")

    # テキストレポート
    report = classification_report(y_true, y_pred, labels=labels, digits=4)
    txt_path = out_dir / "metrics_report.txt"
    txt_path.write_text(report, encoding="utf-8")
    print(f"[INFO] metrics_report.txt を保存しました: {txt_path}")
    print("\n===== classification_report =====")
    print(report)


def main():
    assert CONFIG_PATH.is_file(), f"CONFIG_PATH が存在しません: {CONFIG_PATH}"
    assert CKPT_PATH.is_file(), f"CKPT_PATH が存在しません: {CKPT_PATH}"

    dm = load_data_module_from_yaml(CONFIG_PATH)
    model = load_model(CKPT_PATH)

    print("[INFO] ===== PREDICT (test) =====")
    preds = run_predict(model, dm)

    print("[INFO] ===== BUILD CSV =====")
    df, csv_path = extract_rows(preds, OUT_DIR)

    print("[INFO] ===== CONFUSION MATRIX =====")
    save_confusion_matrix_and_report(df, OUT_DIR)

    print("\n[INFO] 完了しました。")
    print(f"  - results.csv: {csv_path}")
    print(f"  - confusion_matrix_patchcore.png / metrics_report.txt: {OUT_DIR}")


if __name__ == "__main__":
    main()
