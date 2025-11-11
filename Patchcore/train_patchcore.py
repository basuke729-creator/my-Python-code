#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
from pathlib import Path

import torch
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# ==== ここだけ環境に合わせて編集 ====
DATA_ROOT = "/home/yamamao/Patchcore/dataset"  # 下のフォルダ構成を必ず満たすこと
# DATA_ROOT/
#   train/ normal/
#   val/   normal/ abnormal/
#   test/  normal/ abnormal/
OUT_ROOT  = "/home/yamamao/Patchcore/py_results"  # 本スクリプトの出力先
MAX_EPOCHS = 1
BATCH = 32
NUM_WORKERS = 8
BACKBONE = "resnet50"
LAYERS = ["layer2", "layer3"]
CORESET_RATIO = 0.01
SEED = 42
# ===================================

# 依存: anomalib==2.x が入っていること
from anomalib.models import Patchcore
from anomalib.data import Folder

def ensure_dirs():
    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)

def make_datamodule():
    """anomalib のバージョン差に合わせて Folder の引数を自動切替"""
    base = dict(
        name="ladder_dataset",
        root=DATA_ROOT,
        train_batch_size=BATCH,
        eval_batch_size=BATCH,
        num_workers=NUM_WORKERS,
    )

    # ---- 方式A: 新しめのAPI（val/test はディレクトリ指定）----
    try:
        from anomalib.data import Folder
        return Folder(
            **base,
            normal_dir="train/normal",   # 学習は normal のみ
            val_dir="val",               # val/ 配下に normal/ abnormal/
            test_dir="test",             # test/配下に normal/ abnormal/
            val_split_mode="from_dir",
            test_split_mode="from_dir",
        )
    except TypeError:
        pass  # この方式が使えない → 次の方式を試す

    # ---- 方式B: 旧API（val/test をクラス別に個別パスで指定）----
    try:
        from anomalib.data import Folder
        return Folder(
            **base,
            # train は normal のみ（abnormal は None でOK）
            normal_dir="train/normal",
            abnormal_dir=None,
            # 検証/テストはクラス別にフルパスを渡す
            normal_val_dir="val/normal",
            abnormal_val_dir="val/abnormal",
            normal_test_dir="test/normal",
            abnormal_test_dir="test/abnormal",
        )
    except TypeError as e:
        raise RuntimeError(
            "この環境の anomalib.Folder がどちらのシグネチャにも一致しません。"
            f" 例外: {e}"
        )


def make_model() -> Patchcore:
    return Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        coreset_sampling_ratio=CORESET_RATIO,
    )

def find_ckpt(search_dir: str) -> str | None:
    """anomalib が作る Lightning チェックポイントをざっくり探索"""
    pats = [
        os.path.join(search_dir, "**", "weights", "Lightning", "model.ckpt"),
        os.path.join(search_dir, "**", "model.ckpt"),
    ]
    for pat in pats:
        hits = glob.glob(pat, recursive=True)
        if hits:
            # 一番新しそうなもの
            return max(hits, key=os.path.getmtime)
    return None

def extract_preds_to_csv(pred_batches, csv_path: str):
    """
    trainer.predict の返り値（バッチのリスト）から
    汎用的に image_path / label / pred_label / score を吸い出して CSV にする。
    """
    rows = []
    def _todict(x):
        if isinstance(x, dict):
            return x
        # SimpleNamespace や Lightning の特殊オブジェクトにも対応
        return {k: getattr(x, k) for k in dir(x) if not k.startswith("_")}

    for batch in pred_batches:
        # batch は list[dict] だったり dict[tensor] だったりするので広げる
        if isinstance(batch, (list, tuple)):
            items = batch
        else:
            items = [batch]
        for it in items:
            d = _todict(it)

            # 代表的キー名を総当たり（2.2系でだいたい拾える）
            def pick(keys, default=None):
                for k in keys:
                    if k in d:
                        return d[k]
                # “scores” などテンソルも来るので最後に探索
                for k, v in d.items():
                    if any(kk in k for kk in keys):
                        return v
                return default

            path  = pick(["image_path", "filepath", "path", "image", "image_name"])
            label = pick(["label", "target", "gt_label", "ground_truth", "y"])
            plbl  = pick(["pred_label", "pred_labels", "y_hat", "pred"])
            score = pick(["pred_score", "anomaly_score", "score", "image_score"])

            # Tensor → 値
            def to_py(v):
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu()
                    if v.numel() == 1:
                        return v.item()
                    return v.tolist()
                return v

            rows.append({
                "image_path": str(path),
                "label": to_py(label),
                "pred_label": to_py(plbl),
                "score": to_py(score),
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df

def main():
    pl.seed_everything(SEED)
    ensure_dirs()

    dm = make_datamodule()
    model = make_model()
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=MAX_EPOCHS,
        default_root_dir=OUT_ROOT,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_checkpointing=True,
    )

    # ---------- Train ----------
    trainer.fit(model, datamodule=dm)

    # ---------- Locate checkpoint ----------
    ckpt = (trainer.checkpoint_callback.best_model_path
            if getattr(trainer, "checkpoint_callback", None)
            else None)
    if not ckpt:
        ckpt = find_ckpt(OUT_ROOT)
    if not ckpt:
        raise FileNotFoundError("model.ckpt が見つかりませんでした。学習ログ配下を確認してください。")

    # ---------- Test (AUROC/F1 などを出力) ----------
    test_metrics = trainer.test(model=model, datamodule=dm, ckpt_path=ckpt)
    with open(os.path.join(OUT_ROOT, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print("Test metrics:", test_metrics)

    # ---------- Predict (画像毎の予測をCSV化) ----------
    pred_batches = trainer.predict(model=model, datamodule=dm, ckpt_path=ckpt)
    csv_path = os.path.join(OUT_ROOT, "predictions.csv")
    df = extract_preds_to_csv(pred_batches, csv_path)
    print(f"Saved predictions to: {csv_path}")

    # ---------- Confusion Matrix / Report ----------
    # ラベルが 0/1 でない場合は安全に変換
    def to01(v):
        if v in (0, 1):
            return int(v)
        # "normal"/"abnormal" や True/False などを吸収
        s = str(v).lower()
        if "abnormal" in s or s == "1" or s == "true":
            return 1
        return 0

    y_true = [to01(v) for v in df["label"].tolist()]
    y_pred = [to01(v) for v in df["pred_label"].tolist()]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, labels=[0, 1],
                                   target_names=["normal", "abnormal"], digits=4)

    cm_csv = os.path.join(OUT_ROOT, "confusion_matrix.csv")
    pd.DataFrame(cm, index=["true_normal", "true_abnormal"],
                    columns=["pred_normal", "pred_abnormal"]).to_csv(cm_csv)
    with open(os.path.join(OUT_ROOT, "classification_report.txt"), "w") as f:
        f.write(report)

    print("Confusion Matrix:\n", cm)
    print("Report:\n", report)
    print(f"Artifacts saved under: {OUT_ROOT}")

if __name__ == "__main__":
    main()

