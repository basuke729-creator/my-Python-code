#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOX -> crop -> EfficientNetV2 -> JPG/CSV/crops/cross-matrix."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from matplotlib import font_manager
from PIL import Image
from torchvision import transforms

DEFAULT_YOLOX_ROOT = "/home/yamanao/yolox_effnetv2/YOLOX"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_YOLOX_CLASS_NAMES = [
    "stepladder", "workplatform", "person N/A", "safe stepladder",
    "1st stepladder", "2nd stepladder", "straddling",
    "unstable stepladder", "safe workplatform", "unstable workplatform",
]


def sync_cuda(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_int_list(value: str) -> Optional[set[int]]:
    value = value.strip()
    if not value:
        return None
    try:
        return {int(x.strip()) for x in value.split(",") if x.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated class IDs.") from exc


def load_label_names(path: Optional[str], count: int) -> List[str]:
    if not path:
        return list(DEFAULT_YOLOX_CLASS_NAMES) if count == 10 else [str(i) for i in range(count)]
    p = Path(path)
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        names = [str(obj.get(str(i), obj.get(i, i))) for i in range(count)] if isinstance(obj, dict) else [str(x) for x in obj]
    else:
        names = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(names) != count:
        raise ValueError(f"Label count mismatch: {len(names)} != {count}")
    return names


def safe_label(label: str) -> str:
    invalid = '<>:"/\\|?*\0'
    return "".join("_" if c in invalid else c for c in label).strip() or "unknown"


def letterbox_pad(img: Image.Image, size: int, bg: str) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), Image.BICUBIC)
    if bg == "white":
        color = (255, 255, 255)
    elif bg == "mean":
        color = tuple(img.resize((1, 1), Image.BILINEAR).getpixel((0, 0)))
    else:
        color = (0, 0, 0)
    canvas = Image.new("RGB", (size, size), color)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def build_transform(size: int, mode: str, bg: str):
    if mode == "pad":
        return transforms.Compose([
            transforms.Lambda(lambda im: letterbox_pad(im, size, bg)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if mode == "crop":
        return transforms.Compose([
            transforms.Resize(size), transforms.CenterCrop(size),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def infer_num_classes(state: dict) -> int:
    candidates = []
    for k, v in state.items():
        if k.endswith("weight") and getattr(v, "ndim", 0) == 2:
            bk = k[:-6] + "bias"
            if bk in state and getattr(state[bk], "ndim", 0) == 1:
                candidates.append(int(v.shape[0]))
    if candidates:
        return candidates[-1]
    raise RuntimeError("Could not infer EfficientNetV2 class count.")


def load_effnet(ckpt_path: str, model_name: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    names = ckpt.get("class_names")
    if names is None:
        n = infer_num_classes(state)
        names = [str(i) for i in range(n)]
    else:
        names = [str(x) for x in names]
        n = len(names)
    model = timm.create_model(model_name, pretrained=False, num_classes=n)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, names


def load_exp_direct(exp_path: str):
    path = str(Path(exp_path).expanduser().resolve())
    spec = importlib.util.spec_from_file_location(f"custom_exp_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load exp file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Exp"):
        raise ImportError(f"{path} does not contain class Exp")
    return module.Exp()


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    yolo_class_id: int
    yolo_score: float
    yolo_class_name: str
    eff_class_id: Optional[int] = None
    eff_class_name: str = ""
    eff_score: float = 0.0


@dataclass
class TimerStats:
    values: Dict[str, List[float]] = field(default_factory=dict)
    def add(self, key: str, value: float) -> None:
        self.values.setdefault(key, []).append(float(value))
    def mean(self, key: str) -> float:
        vals = self.values.get(key, [])
        return float(np.mean(vals)) if vals else 0.0
    def count(self, key: str) -> int:
        return len(self.values.get(key, []))


class YOLOXDetector:
    def __init__(self, args, device):
        root = str(Path(args.yolox_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from yolox.data.data_augment import ValTransform
        from yolox.utils import fuse_model, postprocess

        self.preproc = ValTransform(legacy=args.yolox_legacy)
        self.postprocess = postprocess
        self.device = device
        self.sync = not args.no_sync
        self.fp16 = args.yolox_fp16

        exp = load_exp_direct(args.yolox_exp)
        exp.test_conf = args.yolox_conf
        exp.nmsthre = args.yolox_nms
        if args.yolox_tsize_h is not None or args.yolox_tsize_w is not None:
            h = args.yolox_tsize_h if args.yolox_tsize_h is not None else exp.test_size[0]
            w = args.yolox_tsize_w if args.yolox_tsize_w is not None else exp.test_size[1]
            exp.test_size = (h, w)
        elif args.yolox_tsize is not None:
            exp.test_size = (args.yolox_tsize, args.yolox_tsize)

        self.num_classes = int(exp.num_classes)
        self.conf = float(exp.test_conf)
        self.nms = float(exp.nmsthre)
        self.test_size = tuple(exp.test_size)
        self.names = load_label_names(args.yolox_class_names, self.num_classes)

        model = exp.get_model().to(device).eval()
        if self.fp16:
            if device.type != "cuda":
                raise ValueError("--yolox-fp16 requires CUDA")
            model.half()

        self.decoder = None
        if args.yolox_trt:
            from torch2trt import TRTModule
            trt = TRTModule()
            trt.load_state_dict(torch.load(args.yolox_trt_file, map_location=device))
            dummy = torch.ones(1, 3, *self.test_size, device=device)
            if self.fp16:
                dummy = dummy.half()
            with torch.no_grad():
                model(dummy)
            model.head.decode_in_inference = False
            self.decoder = model.head.decode_outputs
            model = trt
        else:
            ckpt = torch.load(args.yolox_ckpt, map_location="cpu")
            model.load_state_dict(ckpt.get("model", ckpt), strict=True)
            if args.yolox_fuse:
                model = fuse_model(model)

        self.model = model

    def infer(self, frame: np.ndarray):
        oh, ow = frame.shape[:2]
        ratio = min(self.test_size[0] / oh, self.test_size[1] / ow)

        sync_cuda(self.device, self.sync)
        t0 = time.perf_counter()
        img, _ = self.preproc(frame, None, self.test_size)
        x = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            x = x.half()
        sync_cuda(self.device, self.sync)
        t1 = time.perf_counter()

        with torch.no_grad():
            out = self.model(x)
            if self.decoder is not None:
                out = self.decoder(out, dtype=out.type())
        sync_cuda(self.device, self.sync)
        t2 = time.perf_counter()

        if isinstance(out, torch.Tensor):
            out = out.clone()
        out = self.postprocess(out, self.num_classes, self.conf, self.nms, class_agnostic=True)
        sync_cuda(self.device, self.sync)
        t3 = time.perf_counter()

        detections = []
        pred = out[0]
        if pred is not None:
            pred = pred.detach().cpu()
            boxes = pred[:, :4] / ratio
            scores = pred[:, 4] * pred[:, 5]
            ids = pred[:, 6].long()
            for box, score, cid_t in zip(boxes, scores, ids):
                cid = int(cid_t.item())
                x1, y1, x2, y2 = box.tolist()
                x1 = max(0, min(ow - 1, int(np.floor(x1))))
                y1 = max(0, min(oh - 1, int(np.floor(y1))))
                x2 = max(0, min(ow, int(np.ceil(x2))))
                y2 = max(0, min(oh, int(np.ceil(y2))))
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append(Detection(
                    (x1, y1, x2, y2), cid, float(score.item()),
                    self.names[cid] if 0 <= cid < len(self.names) else str(cid),
                ))

        return detections, {
            "yolox_preprocess": (t1 - t0) * 1000,
            "yolox_inference": (t2 - t1) * 1000,
            "yolox_postprocess": (t3 - t2) * 1000,
        }


class EfficientNetClassifier:
    def __init__(self, args, device):
        self.device = device
        self.sync = not args.no_sync
        self.model, self.names = load_effnet(args.eff_ckpt, args.eff_model, device)
        self.transform = build_transform(args.eff_img_size, args.eff_resize_mode, args.eff_bg)
        self.threshold = args.eff_threshold
        self.fp16 = args.eff_fp16
        if self.fp16:
            self.model.half()

    def infer_batch(self, crops: Sequence[np.ndarray]):
        if not crops:
            return [], {"eff_preprocess": 0.0, "eff_inference": 0.0, "eff_postprocess": 0.0}

        sync_cuda(self.device, self.sync)
        t0 = time.perf_counter()
        tensors = []
        for crop in crops:
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensors.append(self.transform(pil))
        batch = torch.stack(tensors).to(self.device, non_blocking=True)
        if self.fp16:
            batch = batch.half()
        sync_cuda(self.device, self.sync)
        t1 = time.perf_counter()

        with torch.no_grad():
            logits = self.model(batch)
        sync_cuda(self.device, self.sync)
        t2 = time.perf_counter()

        probs = torch.softmax(logits, dim=1)
        confs, ids = probs.max(dim=1)
        results = []
        for cid, conf in zip(ids.cpu().tolist(), confs.cpu().tolist()):
            cid, conf = int(cid), float(conf)
            if self.threshold > 0 and conf < self.threshold:
                results.append((-1, "unknown", conf))
            else:
                results.append((cid, self.names[cid] if cid < len(self.names) else str(cid), conf))
        t3 = time.perf_counter()

        return results, {
            "eff_preprocess": (t1 - t0) * 1000,
            "eff_inference": (t2 - t1) * 1000,
            "eff_postprocess": (t3 - t2) * 1000,
        }


def expand_bbox(bbox, fw, fh, scale):
    x1, y1, x2, y2 = bbox
    if scale <= 1:
        return bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return (
        max(0, round(cx - w / 2)), max(0, round(cy - h / 2)),
        min(fw, round(cx + w / 2)), min(fh, round(cy + h / 2)),
    )


def draw_detection(frame, det):
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"yolo={det.yolo_class_id} {det.yolo_score:.2f} | eff={det.eff_class_id} {det.eff_score:.2f}"
    cv2.putText(frame, text, (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)


def set_matrix_font(font_path: Optional[str]) -> None:
    if font_path:
        p = Path(font_path).expanduser()
        font_manager.fontManager.addfont(str(p))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(p)).get_name()
    else:
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in ["Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic", "DejaVu Sans"]:
            if name in available:
                plt.rcParams["font.family"] = name
                break
    plt.rcParams["axes.unicode_minus"] = False


def plot_matrix(matrix: pd.DataFrame, path: Path, title: str, normalized: bool) -> None:
    values = matrix.to_numpy(dtype=float)
    rows, cols = values.shape
    fig, ax = plt.subplots(figsize=(max(6, cols * 0.9), max(5, rows * 0.7)))
    image = ax.imshow(values, interpolation="nearest", cmap="Blues", aspect="auto")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels(matrix.columns, rotation=60, ha="right")
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("EfficientNetV2 classification")
    ax.set_ylabel("YOLOX detection")
    threshold = values.max() / 2 if values.size else 0.5
    for i in range(rows):
        for j in range(cols):
            value = values[i, j]
            text = f"{value:.1f}" if normalized else f"( {int(value)} )"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if value > threshold else "black", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_cross_matrices(prediction_csv: Path, output_dir: Path, font_path: Optional[str]) -> None:
    df = pd.read_csv(prediction_csv).dropna(subset=["yolox_class_name", "eff_class_name"])
    if df.empty:
        print("[WARN] No classified detections; cross matrices were not generated.")
        return
    count = pd.crosstab(df["yolox_class_name"], df["eff_class_name"])
    ratio = pd.crosstab(df["yolox_class_name"], df["eff_class_name"], normalize="index") * 100
    output_dir.mkdir(parents=True, exist_ok=True)
    count_csv = output_dir / "yolox_effnet_count_matrix.csv"
    ratio_csv = output_dir / "yolox_effnet_ratio_matrix.csv"
    count_png = output_dir / "yolox_effnet_count_heatmap.png"
    ratio_png = output_dir / "yolox_effnet_ratio_heatmap.png"
    count.to_csv(count_csv, encoding="utf-8-sig")
    ratio.to_csv(ratio_csv, encoding="utf-8-sig", float_format="%.2f")
    set_matrix_font(font_path)
    plot_matrix(count, count_png, "YOLOX–EfficientNetV2 Cross Matrix", False)
    plot_matrix(ratio, ratio_png, "YOLOX–EfficientNetV2 Cross Matrix (row normalized, %)", True)
    print("\n=== Cross matrix outputs ===")
    print(f"Count CSV   : {count_csv}")
    print(f"Ratio CSV   : {ratio_csv}")
    print(f"Count image : {count_png}")
    print(f"Ratio image : {ratio_png}")
    print("[INFO] These are prediction cross tables, not ground-truth confusion matrices.")


def make_parser():
    p = argparse.ArgumentParser("YOLOX -> EfficientNetV2 integrated inference")
    p.add_argument("--input-video", required=True)
    p.add_argument("--output-video", default=None)
    p.add_argument("--output-frames", default="runs/pipeline/frames")
    p.add_argument("--output-csv", default="runs/pipeline/predictions.csv")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--save-only-detected", action="store_true")
    p.add_argument("--jpg-quality", type=int, default=95)
    p.add_argument("--save-crops", default=None)
    p.add_argument("--crop-jpg-quality", type=int, default=95)
    p.add_argument("--display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--cross-matrix-dir", default=None)
    p.add_argument("--no-cross-matrix", action="store_true")
    p.add_argument("--matrix-font", default=None)

    p.add_argument("--yolox-root", default=DEFAULT_YOLOX_ROOT)
    p.add_argument("--yolox-exp", required=True)
    p.add_argument("--yolox-name", default=None)
    p.add_argument("--yolox-ckpt", required=True)
    p.add_argument("--yolox-class-names", default=None)
    p.add_argument("--yolox-conf", type=float, default=0.3)
    p.add_argument("--yolox-nms", type=float, default=0.3)
    p.add_argument("--yolox-tsize", type=int, default=None)
    p.add_argument("--yolox-tsize-h", type=int, default=None)
    p.add_argument("--yolox-tsize-w", type=int, default=None)
    p.add_argument("--yolox-fp16", action="store_true")
    p.add_argument("--yolox-legacy", action="store_true")
    p.add_argument("--yolox-fuse", action="store_true")
    p.add_argument("--yolox-trt", action="store_true")
    p.add_argument("--yolox-trt-file", default=None)
    p.add_argument("--classify-yolox-classes", type=parse_int_list, default=None)
    p.add_argument("--crop-scale", type=float, default=1.0)
    p.add_argument("--min-crop-size", type=int, default=8)

    p.add_argument("--eff-ckpt", required=True)
    p.add_argument("--eff-model", required=True)
    p.add_argument("--eff-img-size", type=int, default=384)
    p.add_argument("--eff-resize-mode", choices=["pad", "crop", "stretch"], default="pad")
    p.add_argument("--eff-bg", choices=["black", "white", "mean"], default="black")
    p.add_argument("--eff-threshold", type=float, default=0.0)
    p.add_argument("--eff-fp16", action="store_true")
    return p


def main():
    args = make_parser().parse_args()
    if args.crop_scale <= 0 or args.save_every <= 0:
        raise ValueError("crop-scale and save-every must be > 0")
    if not 1 <= args.jpg_quality <= 100 or not 1 <= args.crop_jpg_quality <= 100:
        raise ValueError("JPG quality must be 1..100")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    detector = YOLOXDetector(args, device)
    classifier = EfficientNetClassifier(args, device)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input_video}")
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 and np.isfinite(fps) else 30.0

    output_frames = Path(args.output_frames)
    output_frames.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    crop_root = Path(args.save_crops) if args.save_crops else None
    if crop_root:
        crop_root.mkdir(parents=True, exist_ok=True)

    writer = None
    output_video = Path(args.output_video) if args.output_video else None
    if output_video:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    csv_file = output_csv.open("w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_index", "detection_index", "x1", "y1", "x2", "y2",
        "yolox_class_id", "yolox_class_name", "yolox_score",
        "eff_class_id", "eff_class_name", "eff_score",
    ])

    stats = TimerStats()
    frame_index = 0
    total_crops = 0

    try:
        while True:
            read_t0 = time.perf_counter()
            ok, frame = cap.read()
            read_t1 = time.perf_counter()
            if not ok or (args.max_frames > 0 and frame_index >= args.max_frames):
                break

            pipeline_t0 = time.perf_counter()
            detections, yolo_times = detector.infer(frame)

            crop_t0 = time.perf_counter()
            valid, crops = [], []
            for det in detections:
                if args.classify_yolox_classes is not None and det.yolo_class_id not in args.classify_yolox_classes:
                    continue
                det.bbox = expand_bbox(det.bbox, fw, fh, args.crop_scale)
                x1, y1, x2, y2 = det.bbox
                if x2 - x1 < args.min_crop_size or y2 - y1 < args.min_crop_size:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    valid.append(det)
                    crops.append(crop.copy())
            crop_t1 = time.perf_counter()

            results, eff_times = classifier.infer_batch(crops)
            for det, result in zip(valid, results):
                det.eff_class_id, det.eff_class_name, det.eff_score = result

            draw_t0 = time.perf_counter()
            for det_idx, (det, crop) in enumerate(zip(valid, crops)):
                draw_detection(frame, det)
                x1, y1, x2, y2 = det.bbox
                csv_writer.writerow([
                    frame_index, det_idx, x1, y1, x2, y2,
                    det.yolo_class_id, det.yolo_class_name, f"{det.yolo_score:.6f}",
                    det.eff_class_id, det.eff_class_name, f"{det.eff_score:.6f}",
                ])
                if crop_root:
                    d = crop_root / safe_label(det.eff_class_name)
                    d.mkdir(parents=True, exist_ok=True)
                    path = d / f"frame_{frame_index:08d}_det_{det_idx:03d}.jpg"
                    if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, args.crop_jpg_quality]):
                        raise RuntimeError(f"Failed to save crop: {path}")
            draw_t1 = time.perf_counter()

            video_t0 = time.perf_counter()
            if writer:
                writer.write(frame)
            video_t1 = time.perf_counter()

            save_t0 = time.perf_counter()
            should_save = frame_index % args.save_every == 0
            if args.save_only_detected and not valid:
                should_save = False
            if should_save:
                path = output_frames / f"frame_{frame_index:08d}.jpg"
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality]):
                    raise RuntimeError(f"Failed to save frame: {path}")
            save_t1 = time.perf_counter()
            pipeline_t1 = time.perf_counter()

            if args.display:
                cv2.imshow("YOLOX -> EfficientNetV2", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                    break

            if frame_index >= args.warmup:
                stats.add("video_read", (read_t1 - read_t0) * 1000)
                for k, v in yolo_times.items():
                    stats.add(k, v)
                stats.add("crop", (crop_t1 - crop_t0) * 1000)
                for k, v in eff_times.items():
                    stats.add(k, v)
                stats.add("draw_csv_crop_save", (draw_t1 - draw_t0) * 1000)
                stats.add("video_write", (video_t1 - video_t0) * 1000)
                stats.add("image_save", (save_t1 - save_t0) * 1000)
                stats.add("pipeline_all", (pipeline_t1 - pipeline_t0) * 1000)
                stats.add("total_with_read", (pipeline_t1 - read_t0) * 1000)

            total_crops += len(crops)
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"Processed {frame_index} frames, classified crops={total_crops}")
    finally:
        cap.release()
        if writer:
            writer.release()
        csv_file.close()
        if args.display:
            cv2.destroyAllWindows()

    avg_pipeline = stats.mean("pipeline_all")
    avg_total = stats.mean("total_with_read")
    print("\n=== Speed summary (warmup excluded) ===")
    print(f"frames: {frame_index}, measured_frames: {stats.count('pipeline_all')}, classified_crops: {total_crops}")
    for label, key in [
        ("video read", "video_read"), ("YOLOX preprocess", "yolox_preprocess"),
        ("YOLOX inference", "yolox_inference"), ("YOLOX postprocess", "yolox_postprocess"),
        ("crop", "crop"), ("EffNet preprocess", "eff_preprocess"),
        ("EffNet inference", "eff_inference"), ("EffNet postprocess", "eff_postprocess"),
        ("draw/CSV/crop-save", "draw_csv_crop_save"), ("video write", "video_write"),
        ("JPG image save", "image_save"),
    ]:
        print(f"{label:<23}: {stats.mean(key):.3f} ms/frame")
    print(f"{'pipeline all':<23}: {avg_pipeline:.3f} ms/frame")
    print(f"{'total incl. video read':<23}: {avg_total:.3f} ms/frame")
    print(f"FPS(pipeline)          : {1000 / avg_pipeline:.2f}" if avg_pipeline else "FPS(pipeline): 0.00")
    print(f"FPS(total with read)   : {1000 / avg_total:.2f}" if avg_total else "FPS(total with read): 0.00")
    print(f"Output JPGs : {output_frames}")
    print(f"Output CSV  : {output_csv}")
    if crop_root:
        print(f"Crop output : {crop_root}")

    if not args.no_cross_matrix:
        matrix_dir = Path(args.cross_matrix_dir) if args.cross_matrix_dir else output_csv.parent / "cross_matrix"
        save_cross_matrices(output_csv, matrix_dir, args.matrix_font)

    print("\n[INFO] No ground-truth confusion matrix was generated for ordinary video inference.")


if __name__ == "__main__":
    main()
