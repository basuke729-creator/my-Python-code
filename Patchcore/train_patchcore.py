# runpatchcore.py  （name 必須版の修正）
from __future__ import annotations
import os
import torch
from lightning.pytorch import Trainer
from anomalib.models import Patchcore
from anomalib.data import Folder

# ==== パスだけ確認 ====
DATA_ROOT = "/home/yamamoao/Patchcore/dataset"
TRAIN_ROOT = f"{DATA_ROOT}/train"   # train/normal
VAL_ROOT   = f"{DATA_ROOT}/val"     # val/normal, val/abnormal
TEST_ROOT  = f"{DATA_ROOT}/test"    # test/normal, test/abnormal
OUT_ROOT   = "/home/yamamoao/Patchcore/py_results"
BATCH = 32
NUM_WORKERS = 8
MAX_EPOCHS = 1
# =====================

def _assert_dir(p: str):
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Directory not found: {p}")

def _make_folder_safe(*, name: str, **kwargs) -> Folder:
    """
    anomalib.data.Folder のシグネチャ差異を吸収。
    まず全引数で try、TypeError なら最小引数（name, root, normal_dir, abnormal_dir）で生成し、
    バッチサイズ類は属性で後付け。
    """
    try:
        return Folder(name=name, **kwargs)
    except TypeError:
        minimal = {"name": name, "root": kwargs["root"], "normal_dir": kwargs["normal_dir"]}
        if "abnormal_dir" in kwargs:
            minimal["abnormal_dir"] = kwargs["abnormal_dir"]
        dm = Folder(**minimal)
        for k in ("train_batch_size", "eval_batch_size", "num_workers"):
            if k in kwargs:
                setattr(dm, k, kwargs[k])
        return dm

def make_dm_train() -> Folder:
    _assert_dir(TRAIN_ROOT)
    _assert_dir(os.path.join(TRAIN_ROOT, "normal"))
    return _make_folder_safe(
        name="train",
        root=TRAIN_ROOT,
        normal_dir="normal",
        train_batch_size=BATCH,
        eval_batch_size=BATCH,
        num_workers=NUM_WORKERS,
    )

def make_dm_eval(split_root: str, split_name: str) -> Folder:
    _assert_dir(split_root)
    _assert_dir(os.path.join(split_root, "normal"))
    _assert_dir(os.path.join(split_root, "abnormal"))
    return _make_folder_safe(
        name=split_name,
        root=split_root,
        normal_dir="normal",
        abnormal_dir="abnormal",
        train_batch_size=BATCH,
        eval_batch_size=BATCH,
        num_workers=NUM_WORKERS,
    )

def build_model() -> Patchcore:
    return Patchcore(backbone="resnet50", layers=["layer2", "layer3"], coreset_sampling_ratio=0.01)

def build_trainer() -> Trainer:
    return Trainer(max_epochs=MAX_EPOCHS, default_root_dir=OUT_ROOT, accelerator="auto",
                   log_every_n_steps=1, enable_checkpointing=True)

def main():
    torch.manual_seed(42)
    dm_train = make_dm_train()
    dm_val   = make_dm_eval(VAL_ROOT,  "val")
    dm_test  = make_dm_eval(TEST_ROOT, "test")

    model = build_model()
    trainer = build_trainer()

    trainer.fit(model=model, datamodule=dm_train)
    trainer.validate(model=model, datamodule=dm_val,  ckpt_path="best")
    trainer.test(model=model,    datamodule=dm_test, ckpt_path="best")

if __name__ == "__main__":
    main()
