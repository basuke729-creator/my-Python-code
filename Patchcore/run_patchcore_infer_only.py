#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatchCore inference-only benchmark script.

- Loads a trained PatchCore model from a folder (where patchcore_params.pkl exists)
- Runs inference 105 times, discards first 5 (warmup), averages remaining 100
- Uses torch.cuda.synchronize() to measure GPU execution accurately
- Does NOT save segmentation images / confusion matrices / sorted copies

Expected repo layout (recommended):
patchcore-inspection/
  bin/...
  src/patchcore/...
  run_patchcore_infer_only.py   <-- this file
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

import patchcore.common
import patchcore.patchcore
import patchcore.utils
import patchcore.datasets.mvtec as mvtec


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to trained PatchCore model directory (contains patchcore_params.pkl).",
    )
    ap.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Dataset root path (mvtec-style). Example: /home/.../datasets/pose_dataset",
    )
    ap.add_argument(
        "--subdataset",
        type=str,
        required=True,
        help="Class name (same as '-d safe_pose' etc). Example: safe_pose",
    )

    ap.add_argument("--gpu", type=int, default=0, help="GPU id. Use -1 for CPU.")
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size for test loader.")
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    ap.add_argument("--resize", type=int, default=384, help="Resize for dataset.")
    ap.add_argument("--imagesize", type=int, default=384, help="Image size for dataset.")

    ap.add_argument("--repeat", type=int, default=105, help="Total inference repetitions.")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup iterations to discard.")
    ap.add_argument(
        "--faiss_on_gpu",
        action="store_true",
        help="Use FAISS on GPU (if available). If not set, FAISS on CPU.",
    )
    ap.add_argument("--faiss_num_workers", type=int, default=8, help="FAISS worker threads.")
    return ap.parse_args()


def make_test_dataloader(data_path: str, classname: str, resize: int, imagesize: int,
                         batch_size: int, num_workers: int, seed: int = 0):
    test_dataset = mvtec.MVTecDataset(
        data_path,
        classname=classname,
        resize=resize,
        imagesize=imagesize,
        split=mvtec.DatasetSplit.TEST,
        seed=seed,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader.name = f"mvtec_{classname}"
    return test_loader


def benchmark_inference(pc: patchcore.patchcore.PatchCore, dataloader,
                        repeat: int, warmup: int, device: torch.device):
    """
    Measures time for pc.predict(dataloader) repeated 'repeat' times.
    Discards first 'warmup', returns mean/std over remaining iterations.

    NOTE:
    - This measures the whole "predict over the full test dataloader".
    - If you want per-image timing, we can change to single-batch or single-image loop.
    """
    times_ms = []

    # Ensure model in eval mode if supported
    try:
        pc.model.eval()
    except Exception:
        pass

    # GPU sync before starting
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    for i in range(repeat):
        t0 = time.perf_counter()

        # ---- inference ----
        _scores, _segs, _labels_gt, _masks_gt = pc.predict(dataloader)

        # ---- sync to include actual GPU time ----
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0

        if i >= warmup:
            times_ms.append(dt_ms)

        print(f"[{i+1:03d}/{repeat}] infer time = {dt_ms:.3f} ms"
              + ("" if i >= warmup else " (warmup)"))

    times_ms = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(times_ms.mean()) if times_ms.size else float("nan")
    std_ms = float(times_ms.std(ddof=1)) if times_ms.size > 1 else 0.0
    return mean_ms, std_ms, times_ms


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"--model_dir not found: {model_dir}")

    # device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = patchcore.utils.set_torch_device([args.gpu])
    else:
        device = torch.device("cpu")

    print("=== PatchCore inference-only benchmark ===")
    print(f"model_dir : {model_dir}")
    print(f"data_path : {args.data_path}")
    print(f"subdataset: {args.subdataset}")
    print(f"device    : {device}")
    print(f"repeat    : {args.repeat}, warmup={args.warmup}")
    print(f"batch_size: {args.batch_size}, num_workers={args.num_workers}")
    print(f"resize    : {args.resize}, imagesize={args.imagesize}")
    print(f"faiss_on_gpu={args.faiss_on_gpu}, faiss_num_workers={args.faiss_num_workers}")
    print("=========================================")

    # nn_method is required by your PatchCore.load_from_path() signature
    nn_method = patchcore.common.FaissNN(
        faiss_on_gpu=bool(args.faiss_on_gpu),
        faiss_num_workers=int(args.faiss_num_workers),
    )

    # load patchcore instance
    pc = patchcore.patchcore.PatchCore(device)
    # IMPORTANT: your env requires (path, device, nn_method)
    pc.load_from_path(str(model_dir), device, nn_method)

    # dataloader (test split only)
    test_loader = make_test_dataloader(
        data_path=args.data_path,
        classname=args.subdataset,
        resize=args.resize,
        imagesize=args.imagesize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=0,
    )

    # benchmark
    mean_ms, std_ms, times_ms = benchmark_inference(
        pc=pc,
        dataloader=test_loader,
        repeat=args.repeat,
        warmup=args.warmup,
        device=device if isinstance(device, torch.device) else device,
    )

    n = len(times_ms)
    print("\n=== Result ===")
    print(f"used_iters: {n}  (repeat={args.repeat} - warmup={args.warmup})")
    print(f"mean_ms   : {mean_ms:.3f} ms  (per full test-set pass)")
    print(f"std_ms    : {std_ms:.3f} ms")
    if n > 0:
        p50 = float(np.percentile(times_ms, 50))
        p90 = float(np.percentile(times_ms, 90))
        p95 = float(np.percentile(times_ms, 95))
        print(f"p50/p90/p95: {p50:.3f} / {p90:.3f} / {p95:.3f} ms")
    print("=============\n")


if __name__ == "__main__":
    main()
