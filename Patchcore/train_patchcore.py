# runpatchcore.py  （Embedding store is empty 対策・完全版）
from __future__ import annotations
import os
import glob
import torch
from lightning.pytorch import Trainer
from anomalib.models import Patchcore
from anomalib.data import Folder

# ====== パス設定 ======
DATA_ROOT = "/home/yamamoao/Patchcore/dataset"  # ← ここだけ直せばOK
OUT_ROOT  = "/home/yamamoao/Patchcore/py_results"
BATCH = 32
NUM_WORKERS = 8
MAX_EPOCHS = 1
# =====================

def must_dir(p: str):
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Directory not found: {p}")

def count_images(d: str) -> int:
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
    return sum(len(glob.glob(os.path.join(d, e))) for e in exts)

def sanity_check():
    # 期待構成:
    # dataset/
    #   train/ normal/
    #   val/   normal/ abnormal/
    #   test/  normal/ abnormal/
    must_dir(DATA_ROOT)
    for split in ("train", "val", "test"):
        must_dir(os.path.join(DATA_ROOT, split))
        must_dir(os.path.join(DATA_ROOT, split, "normal"))
    # val/test は abnormal も（評価用に）必須
    for split in ("val", "test"):
        must_dir(os.path.join(DATA_ROOT, split, "abnormal"))

    n_train = count_images(os.path.join(DATA_ROOT, "train", "normal"))
    n_val_n = count_images(os.path.join(DATA_ROOT, "val", "normal"))
    n_val_a = count_images(os.path.join(DATA_ROOT, "val", "abnormal"))
    n_test_n = count_images(os.path.join(DATA_ROOT, "test", "normal"))
    n_test_a = count_images(os.path.join(DATA_ROOT, "test", "abnormal"))

    if n_train == 0:
        raise RuntimeError("train/normal の画像が 0 枚です。これだと埋め込みが貯まらずに落ちます。")
    print(f"[SANITY] train/normal={n_train}, val(normal,abnormal)=({n_val_n},{n_val_a}), "
          f"test(normal,abnormal)=({n_test_n},{n_test_a})")

def build_datamodule() -> Folder:
    # ← ここを差し替え
    dm = Folder(
        name="ladder_dataset",
        root=DATA_ROOT,
        # --- 重要ポイント：split ごとのサブフォルダを明示する ---
        normal_dir="train/normal",             # 学習で使うのは normal のみ
        normal_val_dir="val/normal",
        abnormal_val_dir="val/abnormal",
        normal_test_dir="test/normal",
        abnormal_test_dir="test/abnormal",
        # ---------------------------------------------------------
        train_batch_size=BATCH,
        eval_batch_size=BATCH,
        num_workers=NUM_WORKERS,
    )
    return dm


def build_model() -> Patchcore:
    # coreset の比率は 0.01 のままでOK。空ストアの場合でも今回は train=0 を防いでいるので通る。
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
    sanity_check()

    dm = build_datamodule()
    model = build_model()
    trainer = build_trainer()

    # fit では train(normal) を使って埋め込みを貯め、終了時に coreset 作成
    trainer.fit(model=model, datamodule=dm)

    # 検証・テスト（best チェックポイントでOK）
    trainer.validate(model=model, datamodule=dm, ckpt_path="best")
    trainer.test(model=model, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    main()
