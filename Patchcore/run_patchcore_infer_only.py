#!/usr/bin/env python3
# run_patchcore_infer_only.py
# PatchCore inference-only benchmark:
# - repeat N times, discard warmup, report mean latency
# - uses torch.cuda.synchronize() for accurate GPU timing
# - NO image saving / NO confusion matrix / NO copying

import argparse
import contextlib
import inspect
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

import patchcore.common
import patchcore.utils

# Dataset import (MVTecDataset)
_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}

LOGGER = logging.getLogger("infer_only")


def make_faiss_nn(use_gpu: bool, num_workers: int):
    """
    PatchCore実装のバージョン差異を吸収して FaissNN を作る。
    よくあるパターン:
      - FaissNN(use_gpu, num_workers)
      - FaissNN(on_gpu, num_workers)
      - FaissNN(num_workers)
    """
    cls = patchcore.common.FaissNN
    sig = None
    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        sig = None

    # まずはキーワード引数で試す（存在しない場合は TypeError になる）
    if sig is not None:
        params = sig.parameters
        # __init__(self, ..., use_gpu=?, num_workers=?)
        if "use_gpu" in params and "num_workers" in params:
            return cls(use_gpu=use_gpu, num_workers=num_workers)
        # __init__(self, ..., on_gpu=?, num_workers=?)
        if "on_gpu" in params and "num_workers" in params:
            return cls(on_gpu=use_gpu, num_workers=num_workers)
        # __init__(self, ..., faiss_on_gpu=?, faiss_num_workers=?)
        if "faiss_on_gpu" in params and "faiss_num_workers" in params:
            return cls(faiss_on_gpu=use_gpu, faiss_num_workers=num_workers)

    # 次に位置引数で試す
    try:
        return cls(use_gpu, num_workers)
    except TypeError:
        pass
    try:
        return cls(use_gpu)
    except TypeError:
        pass
    # 最後：num_workersのみ
    return cls(num_workers)


def build_test_dataloader(
    dataset_name: str,
    data_path: str,
    subdataset: str,
    resize: int,
    imagesize: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    limit_images: int,
):
    dataset_info = _DATASETS[dataset_name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
    DatasetCls = dataset_library.__dict__[dataset_info[1]]

    test_dataset = DatasetCls(
        data_path,
        classname=subdataset,
        resize=resize,
        imagesize=imagesize,
        split=dataset_library.DatasetSplit.TEST,
        seed=seed,
    )

    if limit_images > 0:
        limit_images = min(limit_images, len(test_dataset))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(limit_images)))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader


def load_patchcore_model(model_dir: str, device: torch.device, nn_method):
    """
    保存済みモデルを読み込む。
    多くの patchcore 実装では:
      PatchCore(device).load_from_path(path, device=device, nn_method=nn_method)
    が必要。
    """
    import patchcore.patchcore

    pc = patchcore.patchcore.PatchCore(device)

    # バージョン差異を吸収して load_from_path を呼ぶ
    if not hasattr(pc, "load_from_path"):
        raise RuntimeError("PatchCore instance has no load_from_path(). Your patchcore version differs.")

    fn = pc.load_from_path
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        kwargs = {}

        # 必須 or よくある引数名
        if "device" in params:
            kwargs["device"] = device
        if "nn_method" in params:
            kwargs["nn_method"] = nn_method
        if "faiss_nn" in params:
            kwargs["faiss_nn"] = nn_method  # 念のため別名対応

        fn(model_dir, **kwargs)
        return pc
    except TypeError:
        # 最後の砦：位置引数で
        try:
            fn(model_dir, device, nn_method)
            return pc
        except TypeError as e:
            raise RuntimeError(
                f"load_from_path call failed due to signature mismatch.\n"
                f"model_dir={model_dir}\n"
                f"device={device}\n"
                f"nn_method={type(nn_method)}\n"
                f"error={e}"
            )


def benchmark_predict(
    pc,
    dataloader,
    repeat: int,
    warmup: int,
    use_cuda_sync: bool,
):
    times = []
    total_images = 0
    # Subset の場合 len(dataloader.dataset) が取れないことがあるので安全に
    try:
        total_images = len(dataloader.dataset)
    except Exception:
        total_images = None

    # 実測
    for i in range(repeat):
        if use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        # PatchCore 推論（スコア/セグメンテーションなどを内部計算）
        _scores, _segs, _labels_gt, _masks_gt = pc.predict(dataloader)

        if use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        dt = t1 - t0

        if i >= warmup:
            times.append(dt)

        LOGGER.info("Iter %d/%d: %.4f sec%s", i + 1, repeat, dt, " (warmup)" if i < warmup else "")

    times = np.array(times, dtype=np.float64)
    return times, total_images


def parse_args():
    ap = argparse.ArgumentParser("PatchCore inference-only benchmark (repeat/warmup, CUDA sync).")
    ap.add_argument("--model_dir", type=str, required=True,
                    help="PatchCore saved model directory (folder containing patchcore_params.pkl etc.)")
    ap.add_argument("--dataset", type=str, default="mvtec", choices=list(_DATASETS.keys()))
    ap.add_argument("--data_path", type=str, required=True, help="Dataset root path (mvtec-style).")
    ap.add_argument("--subdataset", "-d", type=str, required=True, help="Class name, e.g., safe_pose")
    ap.add_argument("--gpu", type=int, default=0, help="GPU id. Set -1 for CPU.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--resize", type=int, default=384)
    ap.add_argument("--imagesize", type=int, default=384)
    ap.add_argument("--limit_images", type=int, default=1,
                    help="Use only first N test images for a light benchmark. 0 means use all test images (can be slow).")

    ap.add_argument("--repeat", type=int, default=105)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no_cuda_sync", action="store_true",
                    help="Disable torch.cuda.synchronize() around timing (not recommended if you want accurate GPU timing).")

    ap.add_argument("--faiss_num_workers", type=int, default=8)
    ap.add_argument("--faiss_use_gpu", action="store_true",
                    help="Try to use FAISS GPU (often not recommended for stable speed). Default: CPU FAISS.")
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    LOGGER.info("repeat=%d, warmup=%d", args.repeat, args.warmup)
    LOGGER.info("batch_size=%d, num_workers=%d", args.batch_size, args.num_workers)
    LOGGER.info("resize=%d, imagesize=%d, limit_images=%d", args.resize, args.imagesize, args.limit_images)
    LOGGER.info("faiss_use_gpu=%s, faiss_num_workers=%d", args.faiss_use_gpu, args.faiss_num_workers)

    patchcore.utils.fix_seeds(args.seed)

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        device_context = torch.cuda.device(device)
    else:
        device = torch.device("cpu")
        device_context = contextlib.suppress()

    # FAISS NN method (version-safe)
    nn_method = make_faiss_nn(use_gpu=args.faiss_use_gpu, num_workers=args.faiss_num_workers)

    # Load model
    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"--model_dir not found or not a directory: {args.model_dir}")
    with device_context:
        pc = load_patchcore_model(args.model_dir, device=device, nn_method=nn_method)

    # Build dataloader
    test_loader = build_test_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_path,
        subdataset=args.subdataset,
        resize=args.resize,
        imagesize=args.imagesize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        limit_images=args.limit_images,
    )

    # Benchmark
    with device_context:
        times, n_imgs = benchmark_predict(
            pc,
            test_loader,
            repeat=args.repeat,
            warmup=args.warmup,
            use_cuda_sync=(not args.no_cuda_sync),
        )

    # Report
    mean_s = float(times.mean()) if len(times) else float("nan")
    std_s = float(times.std(ddof=1)) if len(times) > 1 else 0.0
    min_s = float(times.min()) if len(times) else float("nan")
    max_s = float(times.max()) if len(times) else float("nan")

    LOGGER.info("==== RESULT ====")
    LOGGER.info("valid_runs=%d (repeat=%d - warmup=%d)", len(times), args.repeat, args.warmup)
    LOGGER.info("mean=%.6f sec, std=%.6f sec, min=%.6f sec, max=%.6f sec", mean_s, std_s, min_s, max_s)

    if n_imgs is not None and n_imgs > 0:
        LOGGER.info("per_image_mean=%.6f sec/img  (n_imgs=%d in one predict pass)", mean_s / n_imgs, n_imgs)

    print("\n--- SUMMARY ---")
    print(f"mean_sec={mean_s:.6f}")
    print(f"std_sec={std_s:.6f}")
    print(f"min_sec={min_s:.6f}")
    print(f"max_sec={max_s:.6f}")
    if n_imgs is not None and n_imgs > 0:
        print(f"per_image_mean_sec={mean_s / n_imgs:.6f}  (n_imgs={n_imgs})")


if __name__ == "__main__":
    main()

