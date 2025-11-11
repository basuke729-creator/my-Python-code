#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
anomalib v2.x / PatchCore を最小で確実に通すワンファイル版
- 学習 (fit) → テスト (test) を実行
- フォルダ構成は "train/normal", "val/{normal,abnormal}", "test/{normal,abnormal}"
- datamodule は anomalib.data.Folder を、model は PatchcoreLightning を使用
"""

from pathlib import Path
from typing import Tuple

from lightning.pytorch import Trainer, seed_everything
from anomalib.data import Folder
from anomalib.models.image.patchcore import PatchcoreLightning


# ===== ユーザ環境に合わせて必要ならここだけ変更 =====
DATA_ROOT = Path("/home/yamamoao/Patchcore/dataset")   # ← 3階層のルート
OUT_ROOT  = Path("/home/yamamoao/Patchcore/results")   # ← 出力ルート

BATCH = 32
NUM_WORKERS = 8
MAX_EPOCHS = 1
SEED = 42
# ====================================================


def assert_dataset_layout(root: Path) -> Tuple[int, int, int, int, int, int]:
    """想定レイアウトをチェックし、枚数を返す"""
    def count(p: Path) -> int:
        return sum(1 for _ in p.rglob("*") if _.is_file())

    expected = [
        root / "train" / "normal",
        root / "val" / "normal",
        root / "val" / "abnormal",
        root / "test" / "normal",
        root / "test" / "abnormal",
    ]
    for p in expected:
        if not p.exists():
            raise FileNotFoundError(
                f"[DATA CHECK] 必須フォルダが見つかりません: {p}\n"
                "フォルダ構成は必ず次の形にしてください：\n"
                "dataset/\n"
                "  train/ normal/\n"
                "  val/   normal/ abnormal/\n"
                "  test/  normal/ abnormal/\n"
            )

    n_tr = count(root / "train" / "normal")
    n_vn = count(root / "val" / "normal")
    n_va = count(root / "val" / "abnormal")
    n_tn = count(root / "test" / "normal")
    n_ta = count(root / "test" / "abnormal")

    print(f"[SANITY] train/normal={n_tr}, val=(normal,abnormal)=({n_vn},{n_va}), "
          f"test(normal,abnormal)=({n_tn},{n_ta})")

    return n_tr, n_vn, n_va, n_tn, n_ta, 0


def build_datamodule() -> Folder:
    """
    v2.x の Folder は split ごとに normal/abnormal を指定する **1行方式** を使う。
    ※ normal_val_dir 等の“split個別ディレクトリ名”は v2.x の Folder.__init__ では受け付けません。
    """
    dm = Folder(
        name="ladder_dataset",
        root=str(DATA_ROOT),       # Path でも OK だが str で渡して互換確保
        train_dir="train",
        val_dir="val",
        test_dir="test",
        normal_dir="normal",
        abnormal_dir="abnormal",
        # val/test はフォルダ階層からそのまま読む
        val_split_mode="from_dir",
        test_split_mode="from_dir",

        train_batch_size=BATCH,
        eval_batch_size=BATCH,
        num_workers=NUM_WORKERS,
    )
    return dm


def build_model() -> PatchcoreLightning:
    """
    LightningModule を返すことが重要。
    PatchcoreLightning(backbone=..., layers=..., coreset_sampling_ratio=...) を使用。
    """
    model = PatchcoreLightning(
        backbone="resnet50",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.01,
    )
    return model


def main():
    seed_everything(SEED)

    # データ構成の事前チェック
    assert_dataset_layout(DATA_ROOT)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Datamodule / Model
    dm = build_datamodule()
    model = build_model()

    # Trainer
    trainer = Trainer(
        accelerator="auto",
        max_epochs=MAX_EPOCHS,
        default_root_dir=str(OUT_ROOT),
        log_every_n_steps=1,
        enable_progress_bar=True,
        deterministic=False,
    )

    # === 学習 ===
    print("\n[INFO] ======= FIT (train) =======")
    trainer.fit(model=model, datamodule=dm)

    # === テスト（AUROC, F1 が表示される）===
    print("\n[INFO] ======= TEST =======")
    # ckpt は自動保存先から最新を自動解決（無ければ現在の重みで評価）
    try:
        trainer.test(model=model, datamodule=dm)
    except Exception as e:
        print("[WARN] test 実行中に例外が発生しました。学習直後の重みで再試行します:", e)
        trainer.test(model=model, dataloaders=dm.test_dataloader())

    print(f"\n[DONE] 出力ルート: {OUT_ROOT}\n")


if __name__ == "__main__":
    main()
