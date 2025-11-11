#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from lightning.pytorch import Trainer, seed_everything
from anomalib.data import Folder

# Patchcore のクラス名は版差があるので両対応
try:
    from anomalib.models.image.patchcore import PatchcoreLightning as PatchcoreModel
except Exception:
    from anomalib.models.image.patchcore import Patchcore as PatchcoreModel

# ===== あなたの環境 =====
DATA_ROOT = Path("/home/yamamoao/Patchcore/dataset")   # 必須構成: train/val/test の下に normal/abnormal
OUT_ROOT  = Path("/home/yamamoao/Patchcore/results")
BATCH = 32
NUM_WORKERS = 8
MAX_EPOCHS = 1
SEED = 42
# ======================

def _count_files(p: Path) -> int:
    return sum(1 for f in p.rglob("*") if f.is_file())

def sanity_check(root: Path) -> None:
    req = [
        root / "train" / "normal",
        root / "val"   / "normal",
        root / "val"   / "abnormal",
        root / "test"  / "normal",
        root / "test"  / "abnormal",
    ]
    for d in req:
        if not d.exists():
            raise FileNotFoundError(
                f"必須フォルダなし: {d}\n"
                "構成は固定です:\n"
                "dataset/\n"
                "  train/ normal/\n"
                "  val/   normal/ abnormal/\n"
                "  test/  normal/ abnormal/\n"
            )
    print(f"[SANITY] train/normal={_count_files(root/'train'/'normal')}, "
          f"val(normal,abnormal)=({_count_files(root/'val'/'normal')},{_count_files(root/'val'/'abnormal')}), "
          f"test(normal,abnormal)=({_count_files(root/'test'/'normal')},{_count_files(root/'test'/'abnormal')}))")

def make_datamodule() -> Folder:
    # ★ この版は train_dir/val_dir/test_dir を受け付けません（デフォルトの train/val/test を使用）
    return Folder(
        name="ladder_dataset",
        root=str(DATA_ROOT),
        normal_dir="normal",
        abnormal_dir="abnormal",
        val_split_mode="from_dir",
        test_split_mode="from_dir",
        train_batch_size=BATCH,
        eval_batch_size=BATCH,
        num_workers=NUM_WORKERS,
        # 余計な引数は渡さない（task, normal_val_dir 等は NG）
    )

def make_model() -> PatchcoreModel:
    return PatchcoreModel(
        backbone="resnet50",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.01,
    )

def main():
    seed_everything(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sanity_check(DATA_ROOT)

    dm = make_datamodule()
    model = make_model()

    trainer = Trainer(
        accelerator="auto",
        max_epochs=MAX_EPOCHS,
        default_root_dir=str(OUT_ROOT),
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    print("\n[INFO] ===== fit =====")
    trainer.fit(model=model, datamodule=dm)

    print("\n[INFO] ===== test =====")
    trainer.test(model=model, datamodule=dm)

    print(f"\n[DONE] results -> {OUT_ROOT}\n")

if __name__ == "__main__":
    main()
