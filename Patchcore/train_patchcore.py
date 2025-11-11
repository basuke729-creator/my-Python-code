#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PatchCore end-to-end runner (version-robust)
- 学習: normal だけで学習
- 評価: val/test は normal/abnormal の2クラスで評価
- anomalib のバージョン差に自動対応（Folder の引数を動的に切替）
"""

import os
import sys
import json
import inspect
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
import random

# --------- ユーザー設定（必要なら変更） ----------
DATA_ROOT = Path("/home/yamamamoao/Patchcore/dataset")
OUT_ROOT  = Path("/home/yamamamoao/Patchcore/py_results")   # 生成物はここに出力
MAX_EPOCHS = 1
BATCH = 32
NUM_WORKERS = 8
SEED = 42
BACKBONE = "resnet50"
LAYERS = ["layer2", "layer3"]
CORESET_RATIO = 0.01
# -----------------------------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def check_required_paths():
    need = [
        "train/normal",
        "val/normal", "val/abnormal",
        "test/normal", "test/abnormal",
    ]
    missing = []
    for rel in need:
        p = DATA_ROOT / rel
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "必要なフォルダが見つかりません。\n" + "\n".join(missing)
        )


def build_datamodule():
    """anomalib.data.Folder を、バージョン差に合わせて安全に初期化"""
    from anomalib.data import Folder  # ここで import（環境にあるものを使う）
    sig = inspect.signature(Folder.__init__)
    params = set(sig.parameters.keys())

    base = dict(
        name="ladder_dataset",
        root=str(DATA_ROOT),
        train_batch_size=BATCH,
        eval_batch_size=BATCH,
        num_workers=NUM_WORKERS,
    )

    # 候補A: 新しめAPI（val/test をディレクトリ指定）
    cand_A = dict(
        **base,
        normal_dir="train/normal",   # train は normal のみ
        val_dir="val",               # val/ 以下に normal, abnormal
        test_dir="test",             # test/ 以下に normal, abnormal
        val_split_mode="from_dir",
        test_split_mode="from_dir",
    )

    # 候補B: 旧API（val/test をクラス別フルパスで指定）
    cand_B = dict(
        **base,
        normal_dir="train/normal",
        abnormal_dir=None,   # 学習は異常を使わない
        normal_val_dir="val/normal",
        abnormal_val_dir="val/abnormal",
        normal_test_dir="test/normal",
        abnormal_test_dir="test/abnormal",
    )

    # 候補C: さらに旧API（root/{train,val,test}/{normal,abnormal} という規約探索）
    cand_C = dict(
        **base,
        normal_dir="normal",
        abnormal_dir="abnormal",
        val_split_mode="from_dir",
        test_split_mode="from_dir",
    )

    def try_build(cand: Dict[str, Any]):
        filtered = {k: v for k, v in cand.items() if k in params}
        return Folder(**filtered)

    for cand in (cand_A, cand_B, cand_C):
        try:
            dm = try_build(cand)
            print(f"[INFO] Folder init succeeded with keys: {sorted({k for k in cand if k in params})}")
            return dm
        except TypeError as e:
            # 次の候補を試す
            continue

    # どれもダメなら詳細を表示
    import anomalib
    raise RuntimeError(
        "Folder の初期化に失敗しました。\n"
        f"anomalib version: {getattr(anomalib, '__version__', 'unknown')}\n"
        f"Folder.__init__ signature: {sig}\n"
        f"Accepted keys: {sorted(params)}\n"
    )


def build_model():
    """Patchcore モデルを生成（バージョン差に耐性）"""
    from anomalib.models import Patchcore
    # Patchcore は比較的安定した引数。念のためフィルタする
    sig = inspect.signature(Patchcore.__init__)
    params = set(sig.parameters.keys())
    cand = dict(
        backbone=BACKBONE,
        layers=LAYERS,
        coreset_sampling_ratio=CORESET_RATIO,
    )
    filtered = {k: v for k, v in cand.items() if k in params}
    print(f"[INFO] Build Patchcore with keys: {sorted(filtered.keys())}")
    return Patchcore(**filtered)


def train_and_test():
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    dm = build_datamodule()
    model = build_model()

    ckpt_dir = OUT_ROOT / "weights" / "Lightning"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="model",
        save_last=True,
        save_top_k=1,
        monitor=None,  # モニタ無指定（メトリクス名がバージョン差で変わるため）
    )

    trainer = Trainer(
        accelerator="auto",
        max_epochs=MAX_EPOCHS,
        default_root_dir=str(OUT_ROOT),
        enable_progress_bar=True,
        callbacks=[ckpt_cb],
        log_every_n_steps=1,
    )

    print("[INFO] === fit() ===")
    trainer.fit(model=model, datamodule=dm)

    print("[INFO] === test() ===")
    # 予測を Python 側で受け取れるバージョンでは返る。返らない版もあるので try。
    preds = None
    try:
        preds = trainer.test(model=model, datamodule=dm, verbose=True)
    except TypeError:
        # 一部の PL で test() が返り値を持たないことがある
        trainer.test(model=model, datamodule=dm, verbose=True)

    # 可能なら推定結果をCSV化（preds が dict list の場合）
    try:
        if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
            # メトリクスだけの場合もあるので、そのまま保存
            with open(OUT_ROOT / "metrics_from_test_return.json", "w") as f:
                json.dump(preds, f, indent=2)
            print(f"[INFO] metrics_from_test_return.json を保存しました: {OUT_ROOT}")
    except Exception as e:
        print(f"[WARN] test の戻り値保存で例外: {e}")


def main():
    set_seed(SEED)
    ensure_dirs()
    check_required_paths()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")
    train_and_test()
    print(f"[INFO] 完了。出力: {OUT_ROOT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", e)
        sys.exit(1)
