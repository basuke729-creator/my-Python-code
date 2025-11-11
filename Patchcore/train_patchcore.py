# runpatchcore.py
# 最小で確実に通る PatchCore 実行スクリプト（anomalib v2.x/古いFolder両対応）

from __future__ import annotations
import os
import torch
from lightning.pytorch import Trainer
from anomalib.models import Patchcore
from anomalib.data import Folder

# ========= ユーザー環境に合わせてここだけ確認 =========
DATA_ROOT = "/home/yamamoao/Patchcore/dataset"      # <- dataset 直下
TRAIN_ROOT = f"{DATA_ROOT}/train"                   # train/normal だけ
VAL_ROOT   = f"{DATA_ROOT}/val"                     # val/normal, val/abnormal
TEST_ROOT  = f"{DATA_ROOT}/test"                    # test/normal, test/abnormal
OUT_ROOT   = "/home/yamamoao/Patchcore/py_results"   # 出力先
BATCH = 32
NUM_WORKERS = 8
MAX_EPOCHS = 1
# ====================================================

def _assert_dir(p: str):
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Directory not found: {p}")

def _make_folder_safe(**kwargs) -> Folder:
    """
    anomalib.data.Folder の __init__ が受け付けない引数があっても動かすための
    フォールバック。まずできるだけ引数付きで生成し、TypeError が出たら
    最小引数で生成→属性を後から付与します。
    """
    try:
        return Folder(**kwargs)
    except TypeError:
        # 最小限の必須引数だけ取り出す
        minimal = {k: kwargs[k] for k in ("root", "normal_dir") if k in kwargs}
        if "abnormal_dir" in kwargs:
            minimal["abnormal_dir"] = kwargs["abnormal_dir"]
        dm = Folder(**minimal)  # ここは通るはず

        # 受け付けなかったパラメータは属性として後付け（古いFolder想定）
        for k in ("train_batch_size", "eval_batch_size", "num_workers"):
            if k in kwargs:
                setattr(dm, k, kwargs[k])
        return dm

def make_dm_train() -> Folder:
    """train: 正常のみ。abnormal は渡さない。"""
    _assert_dir(TRAIN_ROOT)
    _assert_dir(os.path.join(TRAIN_ROOT, "normal"))
    return _make_folder_safe(
        root=TRAIN_ROOT,
        normal_dir="normal",
        train_batch_size=BATCH,
        eval_batch_size=BATCH,
        num_workers=NUM_WORKERS,
    )

def make_dm_eval(split_root: str) -> Folder:
    """val/test: normal/abnormal の両方が必要。"""
    _assert_dir(split_root)
    _assert_dir(os.path.join(split_root, "normal"))
    _assert_dir(os.path.join(split_root, "abnormal"))
    return _make_folder_safe(
        root=split_root,
        normal_dir="normal",
        abnormal_dir="abnormal",
        train_batch_size=BATCH,
        eval_batch_size=BATCH,
        num_workers=NUM_WORKERS,
    )

def build_model() -> Patchcore:
    # Patchcore は LightningModule として提供される
    return Patchcore(
        backbone="resnet50",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.01,
    )

def build_trainer() -> Trainer:
    return Trainer(
        max_epochs=MAX_EPOCHS,
        default_root_dir=OUT_ROOT,
        accelerator="auto",
        log_every_n_steps=1,
        enable_checkpointing=True,
    )

def main():
    torch.manual_seed(42)

    # datamodule を分離して作成（古い Folder と相性が良い）
    dm_train = make_dm_train()
    dm_val   = make_dm_eval(VAL_ROOT)
    dm_test  = make_dm_eval(TEST_ROOT)

    model = build_model()
    trainer = build_trainer()

    # 学習：train でメモリバンクを構築
    trainer.fit(model=model, datamodule=dm_train)

    # 検証（ベスト ckpt があれば自動で使われる。無ければ最新 weights）
    trainer.validate(model=model, datamodule=dm_val, ckpt_path="best")

    # テスト
    trainer.test(model=model, datamodule=dm_test, ckpt_path="best")

if __name__ == "__main__":
    main()
