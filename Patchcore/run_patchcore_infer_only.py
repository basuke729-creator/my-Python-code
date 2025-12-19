#!/usr/bin/env python3
# run_patchcore_infer_only.py
# - Load trained PatchCore model (from patchcore_params.pkl etc.)
# - Run inference multiple times for benchmarking
# - Drop warmup runs
# - Measure end-to-end predict() time with torch.cuda.synchronize()
# - Output mean time + per-image time + FPS

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.datasets.mvtec as mvtec


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True,
                    help="Trained model folder (contains patchcore_params.pkl etc.)")
    ap.add_argument("--data_path", type=str, required=True,
                    help="Dataset root folder (MVTec format root)")
    ap.add_argument("--subdataset", type=str, required=True,
                    help="Class name (e.g., safe_pose)")
    ap.add_argument("--gpu", type=int, default=0, help="GPU id (use -1 for CPU)")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--resize", type=int, default=384)
    ap.add_argument("--imagesize", type=int, default=384)

    ap.add_argument("--repeat", type=int, default=105, help="Total repeats")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup repeats to discard")
    ap.add_argument("--limit_images", type=int, default=1,
                    help="How many test images to use per repeat (1 is recommended for pure speed). 0 means use all.")

    # FAISS settings (some environments are CPU-only FAISS)
    ap.add_argument("--faiss_on_gpu", action="store_true", help="Use FAISS on GPU (if supported)")
    ap.add_argument("--faiss_num_workers", type=int, default=8)

    return ap.parse_args()


def set_device(gpu_id: int):
    if gpu_id is None or gpu_id < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu_id}")


def make_test_dataloader(data_path, classname, resize, imagesize, batch_size, num_workers):
    test_dataset = mvtec.MVTecDataset(
        data_path,
        classname=classname,
        resize=resize,
        imagesize=imagesize,
        split=mvtec.DatasetSplit.TEST,
        seed=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader


def trim_dataloader_to_n_images(dataloader, n_images: int):
    """
    Create a light-weight "view" using the dataset's data_to_iterate if present.
    This works for MVTecDataset in this repo.
    """
    if n_images <= 0:
        return dataloader

    ds = dataloader.dataset
    if not hasattr(ds, "data_to_iterate"):
        # Fallback: can't trim safely
        return dataloader

    # clone-like: keep only first n entries
    ds.data_to_iterate = ds.data_to_iterate[:n_images]
    return dataloader


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_inference(pc, dataloader, device, repeat, warmup):
    """
    Measures end-to-end time of pc.predict(dataloader)
    """
    assert repeat > warmup, "repeat must be > warmup"

    times = []
    n_imgs = None

    # Ensure stable mode
    torch.set_grad_enabled(False)

    for r in range(repeat):
        # sync before timing
        sync_if_cuda(device)
        t0 = time.perf_counter()

        # predict
        _scores, _segmentations, _labels_gt, _masks_gt = pc.predict(dataloader)

        # sync after timing (important!)
        sync_if_cuda(device)
        t1 = time.perf_counter()

        dt = t1 - t0

        # infer number of images from scores
        if n_imgs is None:
            try:
                n_imgs = int(np.asarray(_scores).shape[0])
            except Exception:
                n_imgs = None

        if r >= warmup:
            times.append(dt)

        print(f"Inferring... {r+1}/{repeat}  dt={dt:.6f}s", flush=True)

    times = np.asarray(times, dtype=np.float64)
    mean_s = float(times.mean())
    std_s = float(times.std())

    # per-image
    if n_imgs is None or n_imgs <= 0:
        per_img = None
        fps = None
    else:
        per_img = mean_s / float(n_imgs)
        fps = 1.0 / per_img if per_img > 0 else None

    return mean_s, std_s, per_img, fps, n_imgs


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    device = set_device(args.gpu)

    print("====================================")
    print(f"repeat      : {args.repeat}, warmup={args.warmup}")
    print(f"batch_size  : {args.batch_size}, num_workers={args.num_workers}")
    print(f"resize      : {args.resize}, imagesize={args.imagesize}")
    print(f"limit_images: {args.limit_images}")
    print(f"device      : {device}")
    print(f"faiss_on_gpu: {args.faiss_on_gpu}, faiss_num_workers={args.faiss_num_workers}")
    print("====================================")

    # Build dataloader
    test_loader = make_test_dataloader(
        args.data_path,
        args.subdataset,
        args.resize,
        args.imagesize,
        args.batch_size,
        args.num_workers,
    )
    test_loader = trim_dataloader_to_n_images(test_loader, args.limit_images)

    # Create PatchCore instance and load params
    # IMPORTANT: your repo's PatchCore.load_from_path requires (path, device, nn_method)
    nn_method = patchcore.common.FaissNN(args.faiss_on_gpu, args.faiss_num_workers)

    pc = patchcore.patchcore.PatchCore(device)
    pc.load_from_path(str(model_dir), device=device, nn_method=nn_method)

    # Warm GPU a bit
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    mean_s, std_s, per_img_s, fps, n_imgs = benchmark_inference(
        pc,
        test_loader,
        device,
        repeat=args.repeat,
        warmup=args.warmup,
    )

    print("\n--- SUMMARY ---")
    print(f"mean_sec={mean_s:.6f}")
    print(f"std_sec ={std_s:.6f}")
    if per_img_s is not None:
        print(f"per_image_mean_sec={per_img_s:.6f}  (n_imgs={n_imgs})")
        if fps is not None:
            print(f"FPS={fps:.2f}")
    else:
        print("per_image_mean_sec=NA  (could not infer n_imgs)")


if __name__ == "__main__":
    main()


