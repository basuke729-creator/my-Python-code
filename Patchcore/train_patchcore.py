#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import json
import inspect
from typing import List, Tuple

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ===== あなたの環境に合わせてここだけ確認 =====
DATA_ROOT = Path("/home/yamamoao/Patchcore/dataset")  # 既にある train/ val/ test/ の親
OUT_ROOT  = Path("/home/yamamoao/Patchcore/results_auto")  # 出力先
MAX_EPOCHS = 1
BATCH = 32
NUM_WORKERS = 8
SEED = 42
# ===============================================

# anomalib import（Patchcore のクラス名差に両対応）
from anomalib.data import Folder as AnomFolder
try:
    from anomalib.models.image.patchcore import PatchcoreLightning as PatchcoreModel
except Exception:
    from anomalib.models.image.patchcore import Patchcore as PatchcoreModel


# --------- ユーティリティ ---------
def _count_files(p: Path) -> int:
    return sum(1 for f in p.rglob("*") if f.is_file())

def _exists_dir(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"必須フォルダがありません: {p}")

def sanity_check_structure(root: Path) -> None:
    """
    あなたの“標準構成”が揃っていることを先に強制確認
    dataset/
      train/ normal/
      val/   normal/ abnormal/
      test/  normal/ abnormal/
    """
    req = [
        root / "train" / "normal",
        root / "val" / "normal",
        root / "val" / "abnormal",
        root / "test" / "normal",
        root / "test" / "abnormal",
    ]
    for d in req:
        _exists_dir(d)

    print(
        f"[SANITY] train/normal={_count_files(root/'train'/'normal')}, "
        f"val(normal,abnormal)=({_count_files(root/'val'/'normal')},{_count_files(root/'val'/'abnormal')}), "
        f"test(normal,abnormal)=({_count_files(root/'test'/'normal')},{_count_files(root/'test'/'abnormal')})"
    )


def folder_accepts(*names: str) -> bool:
    """インストール済み anomalib.data.Folder が指定の引数名を受け付けるか判定"""
    sig = inspect.signature(AnomFolder.__init__)
    return all(n in sig.parameters for n in names)


def build_datamodule(root: Path) -> AnomFolder:
    """
    あなたの構成に確実対応するため、Folder.__init__ のシグネチャを見て
    - A系: `train_dir/val_dir/test_dir` を受け付ける版
    - B系: 受け付けず、デフォルトで train/val/test を使う版
    を自動選択。
    """
    kwargs = dict(
        name="ladder_dataset",
        root=str(root),
        normal_dir="normal",
        abnormal_dir="abnormal",
        train_batch_size=BATCH,
        eval_batch_size=BATCH,
        num_workers=NUM_WORKERS,
    )

    # v2.x の一部は split 指定が "val_split_mode/test_split_mode"
    if folder_accepts("val_split_mode", "test_split_mode"):
        kwargs["val_split_mode"] = "from_dir"
        kwargs["test_split_mode"] = "from_dir"

    # v2.x の別系は train_dir/val_dir/test_dir を受け付ける
    if folder_accepts("train_dir", "val_dir", "test_dir"):
        kwargs.update(
            train_dir="train",
            val_dir="val",
            test_dir="test",
        )
        print("[INFO] Folder: train_dir/val_dir/test_dir を使用します")
        return AnomFolder(**kwargs)

    # 受け付けない版 → そのままでも “標準構成” を読めるようになっている
    print("[INFO] Folder: デフォルトの train/val/test 認識で使用します")
    return AnomFolder(**kwargs)


def build_model() -> PatchcoreModel:
    return PatchcoreModel(
        backbone="resnet50",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.01,
    )


def save_confmat_and_report(y_true: List[int], y_pred: List[int], save_dir: Path) -> None:
    labels = [0, 1]  # 0: normal, 1: abnormal
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # 画像で保存
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["normal", "abnormal"],
        yticklabels=["normal", "abnormal"],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (image-level)",
    )
    # 数値表示
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center")
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # テキストレポート
    report = classification_report(y_true, y_pred, target_names=["normal", "abnormal"], digits=4)
    (save_dir / "metrics_report.txt").write_text(report, encoding="utf-8")
    print("[INFO] metrics_report.txt / confusion_matrix.png を保存しました")


def collect_preds_from_csv(result_csv: Path) -> Tuple[List[int], List[int]]:
    """
    anomalib の test 実行後に保存される predictions CSV（anomalib==2 系は
    `outputs.csv` や `predictions.csv` 相当）を幅広く拾う。
    ここでは pred.csv に統一保存しておく。
    """
    import pandas as pd

    # ありそうな候補を列挙して最初に見つかったものを使う
    candidates = [
        result_csv,
        result_csv.parent / "results.csv",
        result_csv.parent / "predictions.csv",
        result_csv.parent / "outputs.csv",
    ]
    csv_path = None
    for c in candidates:
        if c.exists():
            csv_path = c
            break
    if csv_path is None:
        raise FileNotFoundError(f"予測CSVが見つかりませんでした: {candidates}")

    df = pd.read_csv(csv_path)
    # 列名のゆらぎに対応
    ytrue_col = None
    ypred_col = None
    for cand in ["label_index", "label", "gt_label", "y_true"]:
        if cand in df.columns:
            ytrue_col = cand
            break
    for cand in ["pred_label", "prediction", "y_pred", "pred_index"]:
        if cand in df.columns:
            ypred_col = cand
            break
    if ytrue_col is None or ypred_col is None:
        raise RuntimeError(f"CSVの列名が想定外です。columns={list(df.columns)}")

    df_out = df[[ytrue_col, ypred_col]].copy()
    df_out.columns = ["y_true", "y_pred"]
    df_out.to_csv(result_csv.parent / "pred.csv", index=False)
    return df_out["y_true"].tolist(), df_out["y_pred"].tolist()


def main():
    seed_everything(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sanity_check_structure(DATA_ROOT)

    # データ/モデル
    dm = build_datamodule(DATA_ROOT)
    model = build_model()

    # Trainer
    ckpt_cb = ModelCheckpoint(
        dirpath=str(OUT_ROOT / "weights" / "Lightning"),
        filename="model",
        save_last=True,
        save_top_k=1,
        monitor=None,
    )
    trainer = Trainer(
        accelerator="auto",
        max_epochs=MAX_EPOCHS,
        default_root_dir=str(OUT_ROOT),
        callbacks=[ckpt_cb],
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    print("\n[INFO] ===== fit =====")
    trainer.fit(model=model, datamodule=dm)

    print("\n[INFO] ===== test =====")
    test_outs = trainer.test(model=model, datamodule=dm)

    # 予測CSVの場所を推定して集計（ゆらぎ対応）
    # lightning の default_root_dir 配下に test 結果が落ちます
    # 代表的には OUT_ROOT/"test"/"predictions.csv" 等
    possible = [
        OUT_ROOT / "test" / "pred.csv",
        OUT_ROOT / "test" / "predictions.csv",
        OUT_ROOT / "test" / "outputs.csv",
        OUT_ROOT / "results.csv",
    ]
    y_true, y_pred = collect_preds_from_csv(possible[0])
    save_confmat_and_report(y_true, y_pred, OUT_ROOT)

    print(f"\n[DONE] すべての成果物を {OUT_ROOT} に保存しました。")


if __name__ == "__main__":
    main()
