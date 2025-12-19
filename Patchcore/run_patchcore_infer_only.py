#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_patchcore_infer_only.py
Infer-only benchmark for PatchCore with detailed timing:
- dataloader fetch time (CPU/I-O/transform)
- faiss search time (NN search)
- end-to-end predict() time

repeat=105, warmup=5 などの設定で、mean/sec と FPS を出力する。
"""

import argparse
import contextlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import patchcore.common
import patchcore.patchcore
import patchcore.utils

# dataset
_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


# ---------------------------
# Timing helpers
# ---------------------------

@dataclass
class TimingStats:
    fetch_sec: List[float]
    predict_sec: List[float]
    faiss_sec: List[float]

    def __init__(self):
        self.fetch_sec = []
        self.predict_sec = []
        self.faiss_sec = []

    def summary(self, name: str, vals: List[float]) -> Dict[str, float]:
        if len(vals) == 0:
            return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        arr = np.array(vals, dtype=np.float64)
        return {
            "count": float(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }


class FaissNNTimingWrapper:
    """
    patchcore.common.FaissNN の search() をラップして時間を積算する。
    """
    def __init__(self, nn_obj: Any, stats: TimingStats, device: torch.device):
        self.nn = nn_obj
        self.stats = stats
        self.device = device

    def __getattr__(self, name):
        return getattr(self.nn, name)

    def search(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = self.nn.search(*args, **kwargs)
        # FAISS search 自体はCPU実装のことも多いが、GPU FAISS の可能性もあるので一応同期
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        self.stats.faiss_sec.append(t1 - t0)
        return out


def make_faiss_nn(faiss_on_gpu: bool, faiss_num_workers: int):
    """
    patchcore の FaissNN の初期化シグネチャ差異に耐える。
    - FaissNN(faiss_on_gpu, faiss_num_workers) 形式
    - FaissNN(faiss_num_workers) 形式
    - FaissNN() 形式
    """
    cls = patchcore.common.FaissNN
    try:
        return cls(faiss_on_gpu, faiss_num_workers)
    except TypeError:
        pass
    try:
        return cls(faiss_num_workers)
    except TypeError:
        pass
    return cls()


def dataloader_fetch_one(dl_iter):
    """
    DataLoader から 1バッチ取る（ここが I/O + transform の塊）
    """
    return next(dl_iter)


def sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------
# Main
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, type=str, help=".../models/<dataset_name> など patchcore_params.pkl がある場所")
    ap.add_argument("--data_path", required=True, type=str, help="dataset root path")
    ap.add_argument("--subdataset", required=True, type=str, help="e.g. safe_pose")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--resize", type=int, default=384)
    ap.add_argument("--imagesize", type=int, default=384)
    ap.add_argument("--repeat", type=int, default=105)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--limit_images", type=int, default=1, help="1にすると毎ループ1枚だけで測れる")
    ap.add_argument("--faiss_on_gpu", action="store_true")
    ap.add_argument("--faiss_num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset_name", type=str, default="mvtec", choices=list(_DATASETS.keys()))
    return ap.parse_args()


def build_test_dataloader(args) -> torch.utils.data.DataLoader:
    dataset_info = _DATASETS[args.dataset_name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    # TEST split
    test_dataset = dataset_library.__dict__[dataset_info[1]](
        args.data_path,
        classname=args.subdataset,
        resize=args.resize,
        imagesize=args.imagesize,
        split=dataset_library.DatasetSplit.TEST,
        seed=args.seed,
    )

    dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dl.name = f"{args.dataset_name}_{args.subdataset}"
    return dl


def main():
    args = parse_args()

    # device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    device_context = torch.cuda.device(device) if device.type == "cuda" else contextlib.suppress()

    print("==== CONFIG ====")
    print(f"device              : {device}")
    print(f"repeat              : {args.repeat}, warmup={args.warmup}")
    print(f"batch_size          : {args.batch_size}, num_workers={args.num_workers}")
    print(f"resize/imagesize    : {args.resize}/{args.imagesize}")
    print(f"limit_images        : {args.limit_images}")
    print(f"faiss_on_gpu        : {args.faiss_on_gpu}, faiss_num_workers={args.faiss_num_workers}")
    print(f"model_dir           : {args.model_dir}")
    print(f"data_path/subdataset: {args.data_path} / {args.subdataset}")
    print("================\n")

    # dataloader
    test_loader = build_test_dataloader(args)

    # PatchCore load
    stats = TimingStats()

    with device_context:
        patchcore.utils.fix_seeds(args.seed, device)

        nn_method = make_faiss_nn(args.faiss_on_gpu, args.faiss_num_workers)
        # Wrap for timing
        nn_method = FaissNNTimingWrapper(nn_method, stats, device)

        pc = patchcore.patchcore.PatchCore(device)
        # ここがあなたの環境で必須（エラーに出てた通り）:
        # load_from_path(model_dir, device, nn_method)
        pc.load_from_path(args.model_dir, device=device, nn_method=nn_method)

        # warmup & repeat
        # “毎回 dataloader を先頭から使う” ＝キャッシュ/IOの影響を見たい場合はこれが分かりやすい
        def run_once() -> Tuple[float, float]:
            # dataloader fetch time
            t0 = time.perf_counter()
            dl_iter = iter(test_loader)

            # limit_images=1 の場合は1バッチだけ取る
            # （DataLoaderの遅さが見える）
            _ = dataloader_fetch_one(dl_iter)
            sync_if_cuda(device)
            t1 = time.perf_counter()
            fetch_dt = t1 - t0

            # predict time（画像保存などを入れてない“推論だけ”想定）
            # ただし PatchCore.predict は内部で dataloader を回すので、
            # limit_images=1 を強く推奨（重すぎる場合の回避）
            # ここは “predict全体” の時間として測る
            t2 = time.perf_counter()
            sync_if_cuda(device)
            # NOTE: PatchCore.predict(dataloader) をそのまま呼ぶ
            # limit_images は環境差があるので、対応していなければ「最初のN枚だけ」にする
            if args.limit_images is not None and args.limit_images > 0:
                # 手元で確実にN枚にするため、一時的にラッパーを作る
                class _LimitLoader:
                    def __init__(self, base, n_batches):
                        self.base = base
                        self.n_batches = n_batches
                    def __iter__(self):
                        it = iter(self.base)
                        for _i in range(self.n_batches):
                            yield next(it)
                    def __len__(self):
                        return min(len(self.base), self.n_batches)

                # n_imgs -> n_batches（batch_sizeで割って切り上げ）
                n_batches = int(np.ceil(args.limit_images / max(1, args.batch_size)))
                limited = _LimitLoader(test_loader, n_batches)
                _ = pc.predict(limited)
            else:
                _ = pc.predict(test_loader)

            sync_if_cuda(device)
            t3 = time.perf_counter()
            pred_dt = t3 - t2
            return fetch_dt, pred_dt

        # 実行
        all_pred = []
        for i in range(args.repeat):
            fetch_dt, pred_dt = run_once()
            stats.fetch_sec.append(fetch_dt)
            stats.predict_sec.append(pred_dt)

            print(f"Inferring... {i+1:3d}/{args.repeat}  dt={pred_dt:.6f}s  fetch={fetch_dt:.6f}s")

        # warmup drop
        w = args.warmup
        pred_use = stats.predict_sec[w:] if len(stats.predict_sec) > w else []
        fetch_use = stats.fetch_sec[w:] if len(stats.fetch_sec) > w else []
        faiss_use = stats.faiss_sec[w:] if len(stats.faiss_sec) > w else stats.faiss_sec

        # per-image seconds and FPS（limit_images枚を処理したと仮定）
        n_imgs = max(1, int(args.limit_images))
        per_img_mean = (np.mean(pred_use) / n_imgs) if len(pred_use) > 0 else float("nan")
        fps = (1.0 / per_img_mean) if per_img_mean and per_img_mean > 0 else float("nan")

        print("\n=== SUMMARY (after warmup drop) ===")
        s_pred = stats.summary("predict", pred_use)
        s_fetch = stats.summary("fetch", fetch_use)
        s_faiss = stats.summary("faiss", faiss_use)

        print(f"predict   mean={s_pred['mean']:.6f}s  std={s_pred['std']:.6f}s  min={s_pred['min']:.6f}s  max={s_pred['max']:.6f}s  (n={int(s_pred['count'])})")
        print(f"fetch     mean={s_fetch['mean']:.6f}s  std={s_fetch['std']:.6f}s  min={s_fetch['min']:.6f}s  max={s_fetch['max']:.6f}s  (n={int(s_fetch['count'])})")
        if len(faiss_use) > 0:
            print(f"faiss     mean={s_faiss['mean']:.6f}s  std={s_faiss['std']:.6f}s  min={s_faiss['min']:.6f}s  max={s_faiss['max']:.6f}s  (calls={int(s_faiss['count'])})")
        else:
            print("faiss     (no timing captured)  -> FaissNN.search が通っていない/別経路の可能性")

        print(f"\nper_image_mean_sec = {per_img_mean:.6f}s  (limit_images={n_imgs})")
        print(f"FPS                = {fps:.2f}")
        print("==================================\n")

        # 追加の目安表示
        if device.type == "cuda":
            print("[Hint] GPU util が低い場合、fetch(mean) が大きい or CPU側が詰まっている可能性が高いです。")
        print("[Hint] faiss(mean) が大きい場合、近傍探索（メモリバンク/NN検索）がボトルネックです。")


if __name__ == "__main__":
    main()


