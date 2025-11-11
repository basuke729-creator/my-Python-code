# runpatchcore.py  —— anomalib v2.x 用 / 最小で確実に通る版
# 目的: Patchcore の学習(=memory bank 構築) → test 評価(画像AUROC/F1) まで実施

from pathlib import Path
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

# ====== 環境に合わせてここだけ確認 ======
DATA_ROOT = Path("/home/yamamao/Patchcore/dataset")   # dataset の親ディレクトリ
OUT_DIR   = Path("/home/yamamao/Patchcore/results_py") # 出力先
# フォルダ構成（必須）
# dataset/
#   train/
#     normal/
#   val/
#     normal/
#     abnormal/
#   test/
#     normal/
#     abnormal/
# ======================================

def make_datamodule() -> Folder:
    """v2 の Folder DataModule を作成。
    train は normal のみ、val/test は normal+abnormal を `from_dir` で使用。
    """
    dm = Folder(
        name="ladder_dataset",
        root=str(DATA_ROOT),
        normal_dir="normal",
        abnormal_dir="abnormal",
        val_split_mode="from_dir",
        test_split_mode="from_dir",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
    )
    return dm

def make_model() -> Patchcore:
    """Patchcore モデルを作成。backbone/layers/coreset 比率は一般的な設定。"""
    model = Patchcore(
        backbone="resnet50",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.01,
    )
    return model

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dm = make_datamodule()
    model = make_model()

    # v2 は Lightning Trainer ではなく Engine を使用
    engine = Engine(
        default_root_dir=str(OUT_DIR),
        accelerator="auto",  # GPUがあればGPU、なければCPU
        max_epochs=1,
        log_every_n_steps=1,
    )

    # 1) 学習（memory bank 構築）
    engine.fit(model=model, datamodule=dm)

    # 2) 評価（test/ を使用）
    metrics = engine.test(model=model, datamodule=dm)
    # metrics は list[dict] で返る（例: [{'image_AUROC': 0.98, 'image_F1Score': 0.95, ...}]）
    print("Test metrics:", metrics)

if __name__ == "__main__":
    main()
