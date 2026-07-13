#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOX -> EfficientNetV2 integrated video inference pipeline.

Flow:
  video frame -> YOLOX detection -> crop bounding boxes in memory
  -> batched EfficientNetV2 classification -> draw -> JPG/CSV -> speed summary

Notes:
- Crops are not written to disk unless --save-crops is specified.
- A confusion matrix is not generated because ordinary video inference has no
  ground-truth class labels. Prediction CSV is saved instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

# Change this default with --yolox-root when your environment differs.
DEFAULT_YOLOX_ROOT = "/home/ryokumer/project/kanden_eng/src/YOLOX"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_YOLOX_CLASS_NAMES = [
    "stepladder",
    "workplatform",
    "person N/A",
    "safe stepladder",
    "1st stepladder",
    "2nd stepladder",
    "straddling",
    "unstable stepladder",
    "safe workplatform",
    "unstable workplatform",
]


def sync_cuda(device: torch.device, enabled: bool = True) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_int_list(value: str) -> Optional[set[int]]:
    value = value.strip()
    if not value:
        return None
    try:
        return {int(x.strip()) for x in value.split(",") if x.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Class IDs must be comma-separated integers, e.g. 0,2,5"
        ) from exc


def load_label_names(path: Optional[str], expected_count: int) -> List[str]:
    if not path:
        if expected_count == len(DEFAULT_YOLOX_CLASS_NAMES):
            return list(DEFAULT_YOLOX_CLASS_NAMES)
        return [str(i) for i in range(expected_count)]

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Label file not found: {p}")

    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            names = [str(obj.get(str(i), obj.get(i, i))) for i in range(expected_count)]
        elif isinstance(obj, list):
            names = [str(x) for x in obj]
        else:
            raise ValueError("JSON label file must contain a list or dictionary.")
    else:
        names = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    if len(names) != expected_count:
        raise ValueError(
            f"Label count mismatch: file has {len(names)}, model expects {expected_count}."
        )
    return names


def safe_label_for_path(label: str) -> str:
    invalid = '<>:"/\\|?*\0'
    out = "".join("_" if c in invalid else c for c in label).strip()
    return out or "unknown"


def letterbox_pad(img: Image.Image, dst_size: int, bg: str = "black") -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (dst_size, dst_size), (0, 0, 0))

    scale = dst_size / max(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.BICUBIC)

    if bg == "white":
        canvas = Image.new("RGB", (dst_size, dst_size), (255, 255, 255))
    elif bg == "mean":
        mean_pixel = tuple(img.resize((1, 1), Image.BILINEAR).getpixel((0, 0)))
        canvas = Image.new("RGB", (dst_size, dst_size), mean_pixel)
    else:
        canvas = Image.new("RGB", (dst_size, dst_size), (0, 0, 0))

    canvas.paste(resized, ((dst_size - nw) // 2, (dst_size - nh) // 2))
    return canvas


def build_effnet_transform(img_size: int, resize_mode: str, bg: str):
    if resize_mode == "pad":
        return transforms.Compose([
            transforms.Lambda(lambda im: letterbox_pad(im, img_size, bg)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if resize_mode == "crop":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def infer_num_classes_from_state_dict(state_dict: dict) -> int:
    candidates: List[Tuple[str, int]] = []
    for key, value in state_dict.items():
        if not (key.endswith("weight") and getattr(value, "ndim", 0) == 2):
            continue
        bias_key = key[:-6] + "bias"
        if bias_key in state_dict and getattr(state_dict[bias_key], "ndim", 0) == 1:
            candidates.append((key, int(value.shape[0])))
    if candidates:
        return candidates[-1][1]

    dims = [
        int(v.shape[0])
        for k, v in state_dict.items()
        if k.endswith("weight") and getattr(v, "ndim", 0) == 2
    ]
    if not dims:
        raise RuntimeError("Could not infer EfficientNetV2 class count from checkpoint.")
    return min(dims)


def load_effnet_model(
    ckpt_path: str,
    model_name: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, List[str]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    class_names = ckpt.get("class_names")
    if class_names is None:
        num_classes = infer_num_classes_from_state_dict(state)
        class_names = [str(i) for i in range(num_classes)]
    else:
        class_names = [str(x) for x in class_names]
        num_classes = len(class_names)

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, class_names


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

    def add(self, key: str, value_ms: float) -> None:
        self.values.setdefault(key, []).append(float(value_ms))

    def mean(self, key: str) -> float:
        vals = self.values.get(key, [])
        return float(np.mean(vals)) if vals else 0.0

    def count(self, key: str) -> int:
        return len(self.values.get(key, []))


class YOLOXDetector:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        yolox_root = str(Path(args.yolox_root).expanduser())
        if yolox_root not in sys.path:
            sys.path.append(yolox_root)

        try:
            from yolox.data.data_augment import ValTransform
            from yolox.exp import get_exp
            from yolox.utils import fuse_model, postprocess
        except ImportError as exc:
            raise RuntimeError(
                f"Could not import YOLOX from {yolox_root}. Check --yolox-root."
            ) from exc

        self.postprocess_fn = postprocess
        self.preproc = ValTransform(legacy=args.yolox_legacy)
        self.device = device
        self.fp16 = args.yolox_fp16
        self.sync_enabled = not args.no_sync

        exp = get_exp(args.yolox_exp, args.yolox_name)
        if args.yolox_conf is not None:
            exp.test_conf = args.yolox_conf
        if args.yolox_nms is not None:
            exp.nmsthre = args.yolox_nms
        if args.yolox_tsize_h is not None or args.yolox_tsize_w is not None:
            h = args.yolox_tsize_h if args.yolox_tsize_h is not None else exp.test_size[0]
            w = args.yolox_tsize_w if args.yolox_tsize_w is not None else exp.test_size[1]
            exp.test_size = (h, w)
        elif args.yolox_tsize is not None:
            exp.test_size = (args.yolox_tsize, args.yolox_tsize)

        self.num_classes = int(exp.num_classes)
        self.confthre = float(exp.test_conf)
        self.nmsthre = float(exp.nmsthre)
        self.test_size = tuple(exp.test_size)
        self.class_names = load_label_names(args.yolox_class_names, self.num_classes)

        model = exp.get_model()
        model.to(device).eval()
        if self.fp16:
            if device.type != "cuda":
                raise ValueError("--yolox-fp16 requires CUDA.")
            model.half()

        decoder = None
        if args.yolox_trt:
            if device.type != "cuda":
                raise ValueError("--yolox-trt requires CUDA.")
            from torch2trt import TRTModule

            trt_file = args.yolox_trt_file
            if not trt_file:
                raise ValueError("--yolox-trt requires --yolox-trt-file.")
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file, map_location=device))

            dummy = torch.ones(1, 3, self.test_size[0], self.test_size[1], device=device)
            if self.fp16:
                dummy = dummy.half()
            with torch.inference_mode():
                model(dummy)
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            model = model_trt
        else:
            ckpt = torch.load(args.yolox_ckpt, map_location="cpu")
            state = ckpt.get("model", ckpt)
            model.load_state_dict(state, strict=True)
            if args.yolox_fuse:
                model = fuse_model(model)

        self.model = model
        self.decoder = decoder

    def infer(self, frame_bgr: np.ndarray) -> Tuple[List[Detection], Dict[str, float]]:
        original_h, original_w = frame_bgr.shape[:2]
        ratio = min(self.test_size[0] / original_h, self.test_size[1] / original_w)

        sync_cuda(self.device, self.sync_enabled)
        t0 = time.perf_counter()
        img, _ = self.preproc(frame_bgr, None, self.test_size)
        tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            tensor = tensor.half()
        sync_cuda(self.device, self.sync_enabled)
        t1 = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model(tensor)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
        sync_cuda(self.device, self.sync_enabled)
        t2 = time.perf_counter()

        outputs = self.postprocess_fn(
            outputs,
            self.num_classes,
            self.confthre,
            self.nmsthre,
            class_agnostic=True,
        )
        sync_cuda(self.device, self.sync_enabled)
        t3 = time.perf_counter()

        detections: List[Detection] = []
        output = outputs[0]
        if output is not None:
            output = output.detach().cpu()
            boxes = output[:, :4] / ratio
            scores = output[:, 4] * output[:, 5]
            class_ids = output[:, 6].to(torch.int64)

            for box, score, class_id_tensor in zip(boxes, scores, class_ids):
                class_id = int(class_id_tensor.item())
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                x1i = max(0, min(original_w - 1, int(np.floor(x1))))
                y1i = max(0, min(original_h - 1, int(np.floor(y1))))
                x2i = max(0, min(original_w, int(np.ceil(x2))))
                y2i = max(0, min(original_h, int(np.ceil(y2))))
                if x2i <= x1i or y2i <= y1i:
                    continue
                name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else str(class_id)
                detections.append(
                    Detection(
                        bbox=(x1i, y1i, x2i, y2i),
                        yolo_class_id=class_id,
                        yolo_score=float(score.item()),
                        yolo_class_name=name,
                    )
                )

        times = {
            "yolox_preprocess": (t1 - t0) * 1000.0,
            "yolox_inference": (t2 - t1) * 1000.0,
            "yolox_postprocess": (t3 - t2) * 1000.0,
        }
        return detections, times


class EfficientNetClassifier:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.device = device
        self.sync_enabled = not args.no_sync
        self.model, self.class_names = load_effnet_model(
            args.eff_ckpt, args.eff_model, device
        )
        self.transform = build_effnet_transform(
            args.eff_img_size, args.eff_resize_mode, args.eff_bg
        )
        self.threshold = float(args.eff_threshold)
        self.fp16 = bool(args.eff_fp16)
        if self.fp16:
            if device.type != "cuda":
                raise ValueError("--eff-fp16 requires CUDA.")
            self.model.half()

    def infer_batch(
        self, crops_bgr: Sequence[np.ndarray]
    ) -> Tuple[List[Tuple[int, str, float]], Dict[str, float]]:
        if not crops_bgr:
            return [], {
                "eff_preprocess": 0.0,
                "eff_inference": 0.0,
                "eff_postprocess": 0.0,
            }

        sync_cuda(self.device, self.sync_enabled)
        t0 = time.perf_counter()
        tensors: List[torch.Tensor] = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensors.append(self.transform(pil_img))
        batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
        if self.fp16:
            batch = batch.half()
        sync_cuda(self.device, self.sync_enabled)
        t1 = time.perf_counter()

        with torch.inference_mode():
            logits = self.model(batch)
        sync_cuda(self.device, self.sync_enabled)
        t2 = time.perf_counter()

        probs = torch.softmax(logits, dim=1)
        confs, indices = torch.max(probs, dim=1)
        confs_cpu = confs.detach().cpu().tolist()
        indices_cpu = indices.detach().cpu().tolist()
        results: List[Tuple[int, str, float]] = []
        for idx, conf in zip(indices_cpu, confs_cpu):
            idx = int(idx)
            conf = float(conf)
            if self.threshold > 0.0 and conf < self.threshold:
                results.append((-1, "unknown", conf))
            else:
                name = self.class_names[idx] if 0 <= idx < len(self.class_names) else str(idx)
                results.append((idx, name, conf))
        t3 = time.perf_counter()

        times = {
            "eff_preprocess": (t1 - t0) * 1000.0,
            "eff_inference": (t2 - t1) * 1000.0,
            "eff_postprocess": (t3 - t2) * 1000.0,
        }
        return results, times


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    scale: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    if scale <= 1.0:
        return bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    nx1 = max(0, int(round(cx - w / 2.0)))
    ny1 = max(0, int(round(cy - h / 2.0)))
    nx2 = min(frame_w, int(round(cx + w / 2.0)))
    ny2 = min(frame_h, int(round(cy + h / 2.0)))
    return nx1, ny1, nx2, ny2


def draw_detection(frame: np.ndarray, det: Detection) -> None:
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # OpenCV putText cannot render Japanese reliably. Use class ID in video and
    # preserve the full class name in CSV/crop folder names.
    eff_text = (
        f"cls={det.eff_class_id} {det.eff_score:.2f}"
        if det.eff_class_id is not None
        else "cls=NA"
    )
    text = f"det={det.yolo_class_id} {det.yolo_score:.2f} | {eff_text}"
    text_y = max(18, y1 - 7)
    cv2.putText(
        frame,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "YOLOX video detection -> crop -> EfficientNetV2 classification"
    )

    # General input/output
    ap.add_argument("--input-video", required=True)
    ap.add_argument("--output-video", default=None, help="Optional annotated video output")
    ap.add_argument("--output-frames", default="runs/pipeline/frames", help="Annotated JPG frame output directory")
    ap.add_argument("--output-csv", default="runs/pipeline/predictions.csv")
    ap.add_argument("--save-every", type=int, default=1, help="Save one JPG every N frames")
    ap.add_argument("--save-only-detected", action="store_true", help="Save JPG only when at least one valid detection exists")
    ap.add_argument("--jpg-quality", type=int, default=95)
    ap.add_argument("--save-crops", default=None, help="Optional crop output root")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-sync", action="store_true", help="Disable CUDA synchronization for rough timing")
    ap.add_argument("--cpu", action="store_true")

    # YOLOX
    ap.add_argument("--yolox-root", default=DEFAULT_YOLOX_ROOT)
    ap.add_argument("--yolox-exp", required=True)
    ap.add_argument("--yolox-name", default=None)
    ap.add_argument("--yolox-ckpt", required=True)
    ap.add_argument("--yolox-class-names", default=None, help="txt or JSON; defaults to numeric IDs")
    ap.add_argument("--yolox-conf", type=float, default=0.3)
    ap.add_argument("--yolox-nms", type=float, default=0.3)
    ap.add_argument("--yolox-tsize", type=int, default=None, help="Square test size")
    ap.add_argument("--yolox-tsize-h", type=int, default=None, help="Rectangular test height")
    ap.add_argument("--yolox-tsize-w", type=int, default=None, help="Rectangular test width")
    ap.add_argument("--yolox-fp16", action="store_true")
    ap.add_argument("--yolox-legacy", action="store_true")
    ap.add_argument("--yolox-fuse", action="store_true")
    ap.add_argument("--yolox-trt", action="store_true")
    ap.add_argument("--yolox-trt-file", default=None)
    ap.add_argument(
        "--classify-yolox-classes",
        type=parse_int_list,
        default=None,
        help="Comma-separated YOLOX class IDs to send to EfficientNetV2; default=all",
    )
    ap.add_argument("--crop-scale", type=float, default=1.0)
    ap.add_argument("--min-crop-size", type=int, default=8)

    # EfficientNetV2
    ap.add_argument("--eff-ckpt", required=True)
    ap.add_argument("--eff-model", required=True)
    ap.add_argument("--eff-img-size", type=int, default=384)
    ap.add_argument("--eff-resize-mode", choices=["pad", "crop", "stretch"], default="pad")
    ap.add_argument("--eff-bg", choices=["black", "white", "mean"], default="black")
    ap.add_argument("--eff-threshold", type=float, default=0.0)
    ap.add_argument("--eff-fp16", action="store_true")

    return ap


def main() -> None:
    args = make_parser().parse_args()
    if args.crop_scale <= 0:
        raise ValueError("--crop-scale must be greater than 0.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be greater than 0.")
    if not 1 <= args.jpg_quality <= 100:
        raise ValueError("--jpg-quality must be between 1 and 100.")

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")

    detector = YOLOXDetector(args, device)
    classifier = EfficientNetClassifier(args, device)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input_video}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if input_fps <= 0 or not np.isfinite(input_fps):
        input_fps = 30.0

    output_video_path = Path(args.output_video) if args.output_video else None
    writer = None
    if output_video_path is not None:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            input_fps,
            (frame_w, frame_h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video writer: {output_video_path}")

    output_frames_dir = Path(args.output_frames)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = output_csv_path.open("w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_index",
        "detection_index",
        "x1", "y1", "x2", "y2",
        "yolox_class_id",
        "yolox_class_name",
        "yolox_score",
        "eff_class_id",
        "eff_class_name",
        "eff_score",
    ])

    crop_root = Path(args.save_crops) if args.save_crops else None
    if crop_root:
        crop_root.mkdir(parents=True, exist_ok=True)

    stats = TimerStats()
    frame_index = 0
    total_classified_crops = 0

    try:
        while True:
            read_t0 = time.perf_counter()
            ok, frame = cap.read()
            read_t1 = time.perf_counter()
            if not ok:
                break
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            pipeline_t0 = time.perf_counter()
            detections, yolo_times = detector.infer(frame)

            crop_t0 = time.perf_counter()
            valid_detections: List[Detection] = []
            crops: List[np.ndarray] = []
            for det in detections:
                if (
                    args.classify_yolox_classes is not None
                    and det.yolo_class_id not in args.classify_yolox_classes
                ):
                    continue
                expanded = expand_bbox(det.bbox, frame_w, frame_h, args.crop_scale)
                x1, y1, x2, y2 = expanded
                if x2 - x1 < args.min_crop_size or y2 - y1 < args.min_crop_size:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                det.bbox = expanded
                valid_detections.append(det)
                crops.append(crop)
            crop_t1 = time.perf_counter()

            class_results, eff_times = classifier.infer_batch(crops)
            for det, result in zip(valid_detections, class_results):
                det.eff_class_id, det.eff_class_name, det.eff_score = result

            draw_t0 = time.perf_counter()
            for det_idx, det in enumerate(valid_detections):
                draw_detection(frame, det)
                x1, y1, x2, y2 = det.bbox
                csv_writer.writerow([
                    frame_index,
                    det_idx,
                    x1, y1, x2, y2,
                    det.yolo_class_id,
                    det.yolo_class_name,
                    f"{det.yolo_score:.6f}",
                    det.eff_class_id,
                    det.eff_class_name,
                    f"{det.eff_score:.6f}",
                ])

                if crop_root is not None:
                    label_dir = crop_root / safe_label_for_path(det.eff_class_name)
                    label_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = label_dir / f"frame_{frame_index:08d}_det_{det_idx:03d}.jpg"
                    x1, y1, x2, y2 = det.bbox
                    cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2])
            draw_t1 = time.perf_counter()

            write_t0 = time.perf_counter()
            if writer is not None:
                writer.write(frame)
            write_t1 = time.perf_counter()

            image_save_t0 = time.perf_counter()
            should_save_frame = (frame_index % args.save_every == 0)
            if args.save_only_detected and not valid_detections:
                should_save_frame = False
            if should_save_frame:
                frame_path = output_frames_dir / f"frame_{frame_index:08d}.jpg"
                ok_save = cv2.imwrite(
                    str(frame_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, int(args.jpg_quality)],
                )
                if not ok_save:
                    raise RuntimeError(f"Failed to save frame: {frame_path}")
            image_save_t1 = time.perf_counter()
            pipeline_t1 = time.perf_counter()

            if args.display:
                cv2.imshow("YOLOX -> EfficientNetV2", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break

            if frame_index >= args.warmup:
                stats.add("video_read", (read_t1 - read_t0) * 1000.0)
                for key, value in yolo_times.items():
                    stats.add(key, value)
                stats.add("crop", (crop_t1 - crop_t0) * 1000.0)
                for key, value in eff_times.items():
                    stats.add(key, value)
                stats.add("draw_csv_crop_save", (draw_t1 - draw_t0) * 1000.0)
                stats.add("video_write", (write_t1 - write_t0) * 1000.0)
                stats.add("image_save", (image_save_t1 - image_save_t0) * 1000.0)
                stats.add("pipeline_all", (pipeline_t1 - pipeline_t0) * 1000.0)
                stats.add("total_with_read", (pipeline_t1 - read_t0) * 1000.0)

            total_classified_crops += len(crops)
            frame_index += 1

            if frame_index % 100 == 0:
                print(f"Processed {frame_index} frames, classified crops={total_classified_crops}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        csv_file.close()
        if args.display:
            cv2.destroyAllWindows()

    measured_frames = stats.count("pipeline_all")
    avg_pipeline = stats.mean("pipeline_all")
    avg_total_read = stats.mean("total_with_read")
    print("\n=== Speed summary (warmup excluded) ===")
    print(f"frames: {frame_index}, measured_frames: {measured_frames}, classified_crops: {total_classified_crops}")
    print(f"video read             : {stats.mean('video_read'):.3f} ms/frame")
    print(f"YOLOX preprocess       : {stats.mean('yolox_preprocess'):.3f} ms/frame")
    print(f"YOLOX inference        : {stats.mean('yolox_inference'):.3f} ms/frame")
    print(f"YOLOX postprocess      : {stats.mean('yolox_postprocess'):.3f} ms/frame")
    print(f"crop                   : {stats.mean('crop'):.3f} ms/frame")
    print(f"EffNet preprocess      : {stats.mean('eff_preprocess'):.3f} ms/frame")
    print(f"EffNet inference       : {stats.mean('eff_inference'):.3f} ms/frame")
    print(f"EffNet postprocess     : {stats.mean('eff_postprocess'):.3f} ms/frame")
    print(f"draw/CSV/crop-save     : {stats.mean('draw_csv_crop_save'):.3f} ms/frame")
    print(f"video write            : {stats.mean('video_write'):.3f} ms/frame")
    print(f"JPG image save         : {stats.mean('image_save'):.3f} ms/frame")
    print(f"pipeline all           : {avg_pipeline:.3f} ms/frame")
    print(f"total incl. video read : {avg_total_read:.3f} ms/frame")
    print(f"FPS(pipeline)          : {1000.0 / avg_pipeline:.2f}" if avg_pipeline > 0 else "FPS(pipeline): 0.00")
    print(f"FPS(total with read)   : {1000.0 / avg_total_read:.2f}" if avg_total_read > 0 else "FPS(total with read): 0.00")
    if output_video_path is not None:
        print(f"Output video: {output_video_path}")
    print(f"Output JPGs : {output_frames_dir}")
    print(f"Output CSV  : {output_csv_path}")
    if crop_root:
        print(f"Crop output : {crop_root}")

    print(
        "\n[INFO] Confusion matrix was not generated. "
        "Video inference has no ground-truth class labels. "
        "Use the original predict_3.py for folder-based evaluation, or prepare "
        "frame/bbox-level ground-truth annotations for end-to-end evaluation."
    )


if __name__ == "__main__":
    main()
