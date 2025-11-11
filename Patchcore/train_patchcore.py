# -*- coding: utf-8 -*-
"""
runpatchcore.py  — anomalib v2系で確実に通る最小版
- train は train/normal だけで学習（One-Class）
- val/test は normal / abnormal の2クラスで評価
- あなたの三層フォルダ構成をそのまま使用（train / val / test）
"""

import os
import torch
from pytorch_lightning import Trainer, seed_everything
from anomalib.models import Patchcore     # LightningModule を返すラッパ
from anomalib.data import Folder          # LightningDataModule（版により仕様差あり）

# === 環境に合わせてここだけ確認 ===
DATASET_DIR = "/home/yamamoao/Patchcore/dataset"
TRAIN_ROOT  = os.path.join(DATASET_DIR, "train")
VAL_ROOT    = os.path.join(DATASET_DIR, "val")
TEST_ROOT   = os.path.join(DATASET_DIR, "test")
OUT_DIR     = "/home/yamamoao/Patchcore/results_py"   # 出力先
MAX_EPOCHS  = 1

def assert_dir(p):
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Not found: {p}")

def make_dm_for(root_dir: str, use_abnormal: bool):
    """
    anomalib の Folder は版差が大きいので、
    どの版でも通るように最小限の引数だけ渡す。
    """
    base_kwargs = dict(
        root=root_dir,
        normal_dir="normal",
    )
    if use_abnormal:
        base_kwargs["abnormal_dir"] = "abnormal"   # val/test では必要
    # train は正常のみなので abnormal_dir は渡さない（古い版だと None も不可）

    # まずは新しめの版で通る形
    try:
        return Folder(**base_kwargs, train_batch_size=32, eval_batch_size=32, num_workers=8)
    except TypeError:
        # 古い版はバッチサイズなどを __init__ で受け付けない
        dm = Folder(**base_kwargs)
        # 代入式も版によっては無視されても害はない
        for k, v in (("train_batch_size", 32), ("eval_batch_size", 32), ("num_workers", 8)):
            try:
                setattr(dm, k, v)
            except Exception:
                pass
        return dm


def build_model():
    # Patchcore の LightningModule を返すラッパ API（backbone などはここで指定）
    return Patchcore(
        backbone="resnet50",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.01,
    )

def main():
    seed_everything(42)

    # フォルダ存在チェック（三層構成を前提）
    assert_dir(os.path.join(TRAIN_ROOT, "normal"))
    # train に abnormal は不要
    assert_dir(os.path.join(VAL_ROOT, "normal"))
    assert_dir(os.path.join(VAL_ROOT, "abnormal"))
    assert_dir(os.path.join(TEST_ROOT, "normal"))
    assert_dir(os.path.join(TEST_ROOT, "abnormal"))

    # DataModules を split 別に用意
    dm_train = make_dm_for(TRAIN_ROOT, split="train", use_abnormal=False)
    dm_val   = make_dm_for(VAL_ROOT,   split="val",   use_abnormal=True)
    dm_test  = make_dm_for(TEST_ROOT,  split="test",  use_abnormal=True)

    model = build_model()

    trainer = Trainer(
        accelerator="auto",
        max_epochs=MAX_EPOCHS,
        default_root_dir=OUT_DIR,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    # --- fit (train + val) ---
    # 版差を吸収するため、DataLoader を明示的に渡す
    trainer.fit(
        model,
        train_dataloaders=dm_train.train_dataloader(),
        val_dataloaders=dm_val.val_dataloader(),
    )

    # --- test ---
    trainer.test(model, dataloaders=dm_test.test_dataloader())

    # 予測画像やメトリクス類は OUT_DIR 以下に保存されます
    print(f"[INFO] Done. See results in: {OUT_DIR}")

if __name__ == "__main__":
    main()

