#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
from typing import Optional

import numpy as np
import torch

import patchcore.common
import patchcore.patchcore
import patchcore.utils

# dataset loader (mvtec互換) を使う前提
_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


def parse_args():
    ap = argparse.ArgumentParser("PatchCore inference benchmark (repeat/warmup, correct FPS)")

    ap.add_argument("--model_dir", type=str, required=True, help="Path to saved PatchCore model directory")
    ap.add_argument("--data_path", type=str, required=True, help="Dataset root path")
    ap.add_argument("--subdataset", type=str, required=True, help="Subdataset/classname (e.g., safe_pose)")

    ap.add_argument("--gpu", type=int, default=0, help="GPU id (use -1 for CPU)")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--resize", type=int, default=384)
    ap.add_argument("--imagesize", type=int, default=384)

    ap.add_argument("--repeat", type=int, default=105, help="Total repeats")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup repeats (discarded)")

    ap.add_argument("--limit_images", type=int, default=0, help="If >0, limit number of test images")

    # FaissNN の環境差異に強くするため、フラグは持つが内部で安全に扱う
    ap.add_argument("--faiss_on_gpu", action="store_true", help="Try to use FAISS on GPU if supported")
    ap.add_argument("--faiss_num_workers", type=int, default=8, help="FAISS workers (CPU)")

    return ap.parse_args()


def load_dataset_mvtec(data_path: str, subdataset: str, resize: int, imagesize: int, seed: int = 0):
    dataset_info = _DATASETS["mvtec"]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    test_dataset = dataset_library.__dict__[dataset_info[1]](
        data_path,
        classname=subdataset,
        resize=resize,
        imagesize=imagesize,
        split=dataset_library.DatasetSplit.TEST,
        seed=seed,
    )
    return test_dataset


def make_subset_dataset(dataset, limit_images: int):
    """LimitLoaderのように wrapper を作らず、Subsetで制限する（.to()エラー回避）"""
    if limit_images and limit_images > 0 and limit_images < len(dataset):
        indices = list(range(limit_images))
        return torch.utils.data.Subset(dataset, indices)
    return dataset


def create_nn_method(faiss_on_gpu: bool, faiss_num_workers: int):
    """
    patchcore.common.FaissNN の引数が環境で違うことがあるので、
    いくつかの呼び出しを試して互換性を取る。
    """
    # 1) まずは positional で（あなたの環境で keyword が弾かれたため）
    try:
        return patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
    except TypeError:
        pass

    # 2) keyword が許される版
    try:
        return patchcore.common.FaissNN(faiss_on_gpu=faiss_on_gpu, faiss_num_workers=faiss_num_workers)
    except TypeError:
        pass

    # 3) workers だけの版
    try:
        return patchcore.common.FaissNN(faiss_num_workers)
    except TypeError:
        pass

    # 4) 引数無し版
    return patchcore.common.FaissNN()


def cuda_sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def run_once(pc: patchcore.patchcore.PatchCore, dataloader, device: torch.device) -> float:
    """
    1回のpredictを正確に計測（GPUの場合は同期を入れて「開始〜完了」だけ測る）
    """
    cuda_sync_if_needed(device)
    t0 = time.perf_counter()

    _scores, _segs, _labels_gt, _masks_gt = pc.predict(dataloader)

    cuda_sync_if_needed(device)
    t1 = time.perf_counter()
    return t1 - t0


def main():
    args = parse_args()

    # device
    if args.gpu is None or args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = patchcore.utils.set_torch_device([args.gpu])

    print("repeat      :", args.repeat, ", warmup=", args.warmup)
    print("batch_size  :", args.batch_size, ", num_workers=", args.num_workers)
    print("resize/imgsz:", args.resize, "/", args.imagesize)
    print("faiss_on_gpu:", bool(args.faiss_on_gpu), ", faiss_num_workers=", args.faiss_num_workers)
    print("model_dir   :", args.model_dir)
    print("data_path   :", args.data_path, "/ subdataset:", args.subdataset)

    # dataset + dataloader
    test_dataset = load_dataset_mvtec(args.data_path, args.subdataset, args.resize, args.imagesize)
    test_dataset = make_subset_dataset(test_dataset, args.limit_images)

    # ★ここが「FPS計算の母数」になる “総画像枚数”
    n_images = len(test_dataset)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # load patchcore model
    nn_method = create_nn_method(args.faiss_on_gpu, args.faiss_num_workers)

    pc = patchcore.patchcore.PatchCore(device)
    # load_from_path のシグネチャに合わせて device と nn_method を渡す
    pc.load_from_path(args.model_dir, device=device, nn_method=nn_method)

    # benchmark
    dts = []
    for i in range(args.repeat):
        dt = run_once(pc, test_loader, device)
        dts.append(dt)
        print(f"Inferring... {i+1}/{args.repeat}  dt={dt:.6f}s")

    # warmupを捨てる
    if args.warmup > 0:
        dts_eval = dts[args.warmup:]
    else:
        dts_eval = dts

    mean_sec = float(np.mean(dts_eval))
    std_sec = float(np.std(dts_eval, ddof=0))

    # ✅ 正しい FPS（1回のpredictで n_images 枚処理した前提）
    # 1回のpredict時間 = mean_sec（秒）
    # FPS = n_images / mean_sec
    fps = (n_images / mean_sec) if mean_sec > 0 else float("inf")

    # 参考：1枚あたり秒 / ms
    per_image_sec = mean_sec / max(1, n_images)
    per_image_ms = per_image_sec * 1000.0

    print("\n--- SUMMARY ---")
    print(f"n_images              = {n_images}")
    print(f"mean_sec_per_run      = {mean_sec:.6f}  (1 run = full testset)")
    print(f"std_sec_per_run       = {std_sec:.6f}")
    print(f"per_image_mean_sec    = {per_image_sec:.6f}  ({per_image_ms:.3f} ms/img)")
    print(f"FPS                   = {fps:.2f}")

    # 追加：batch_size>1 の場合は “images/s” は同じだが “batches/s” も参考に出す
    # batch数 = ceil(n_images / batch_size)
    n_batches = int(np.ceil(n_images / float(args.batch_size)))
    bps = (n_batches / mean_sec) if mean_sec > 0 else float("inf")
    print(f"batches_per_sec       = {bps:.2f}  (batch_size={args.batch_size})")


if __name__ == "__main__":
    main()


