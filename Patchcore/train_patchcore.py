#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Tuple

from lightning.pytorch import Trainer, seed_everything
from anomalib.data import Folder

# ---- Patchcore (版によってクラス名が違うので両対応) ----
try:
    from anomalib.models.image.patchcore import PatchcoreLightning as PatchcoreModel
except Exception:
    from anomalib.models.image.patchcore import Patchcore as PatchcoreModel
# ------------------------------------------------------

# ===== 路径（あなたの環境に合わせてOK） =====
DATA_ROOT = Path("/home/yamamoao/Patchcore/dataset")     # dataset ルート
OUT_ROOT  = Path("/home/yamamoao/Patchcore/results")     # 出力先
BATCH = 32
NUM_WORKERS = 8
MAX_EPOCHS = 1
SEED = 42
# ==============================================


def _count_files(p: Path) -> int:
    return sum(1 for f in p.rglob("*") if f.is_file())


def assert_layout(root: Path) -> None:
    """フォルダ構成チェック（3階層固定）"""
    must = [
        root / "train" / "normal",
        root / "val"   / "normal",
        root / "val"   / "abnormal",
        root / "test"  / "normal",
        root / "test"  / "abnormal",
    ]
    for d in must:
        if not d.exists():
            raise FileNotFoundError(
                f"必須フォルダがありません: {d}\n"
                "構成は以下に固定です:\n"
                "dataset/\n"
                "  train/ normal/\n"
                "  val/   normal/ abnormal/\n"
                "  test/  normal/ abnormal/\n"
            )
    n_tr  = _count_files(root / "train" / "normal")
    n_vn  = _count_files(root / "val"   / "normal")
    n_va  = _count_files(root / "val"   / "abnormal")
    n_tn  = _count_files(root / "test"  / "normal")
    n_ta  = _count_files(root / "test"  / "abnormal")
    print(f"[SANITY] train/normal={n_tr}, val(normal,abnormal)=({n_vn},{n_va}), "
          f"test(normal,abnormal)=({n_tn},{n_ta})")


def make_datamodule() -> Folder:
    """v2.x の Folder が受け付ける引数だけを使用（※ normal_val_dir 等は絶対に渡さない）"""
    dm = Folder(
        name="ladder_dataset",
        root=str(DATA_ROOT),
        train_dir="train",
        val_dir="val",
        test_dir="test",
        normal_dir="normal",
        abnormal_dir="abnormal",
        val_split_mode="from_dir",
        test_split_mode="from_dir",
        train_batch_size=BATCH,
        eval_batch_size=BATCH,
        num_workers=NUM_WORKERS,
    )
    return dm


def make_model() -> PatchcoreModel:
    # どちらのクラスでも同じ init 引数で通ります
    return PatchcoreModel(
        backbone="resnet50",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.01,
    )


def main():
    seed_everything(SEED)
    assert_layout(DATA_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    dm = make_datamodule()
    model = make_model()

    trainer = Trainer(
        accelerator="auto",
        max_epochs=MAX_EPOCHS,
        default_root_dir=str(OUT_ROOT),
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    print("\n[INFO] ===== train (fit) =====")
    trainer.fit(model=model, datamodule=dm)

    print("\n[INFO] ===== test =====")
    trainer.test(model=model, datamodule=dm)

    print(f"\n[DONE] results -> {OUT_ROOT}\n")


if __name__ == "__main__":
    main()

