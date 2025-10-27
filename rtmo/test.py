#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ExifTags


COCO_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),
    (11,12),(11,13),(12,14),
    (13,15),(14,16)
]

def _si(x):
    try:
        return int(round(float(x)))
    except Exception:
        return 0

def load_image_bgr_with_exif(path):
    pil = Image.open(path)
    try:
        exif = pil._getexif()
        if exif:
            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"),
                None,
            )
            if orientation_key is not None:
                orientation = exif.get(orientation_key, 1)
                if orientation == 3:
                    pil = pil.rotate(180, expand=True)
                elif orientation == 6:
                    pil = pil.rotate(270, expand=True)
                elif orientation == 8:
                    pil = pil.rotate(90,  expand=True)
    except Exception:
        pass

    rgb = np.array(pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def load_pipeline_config(pipeline_json_path):
    cfg = {
        "target_w": 416,
        "target_h": 416,
        "keep_ratio": True,
        "pad_val": 114,
        "normalize": {
            "to_rgb": False,
            "mean": [0, 0, 0],
            "std": [1, 1, 1],
            "to_float32": True
        }
    }

    try:
        with open(pipeline_json_path, "r", encoding="utf-8") as f:
            js = json.load(f)
    except Exception:
        return cfg

    transforms = []
    if isinstance(js, dict):
        if "pipeline" in js and isinstance(js["pipeline"], list):
            transforms = js["pipeline"]
        elif (
            "pipeline" in js
            and isinstance(js["pipeline"], dict)
            and "transforms" in js["pipeline"]
        ):
            transforms = js["pipeline"]["transforms"]
        elif "transforms" in js:
            transforms = js["transforms"]

    for t in transforms:
        tname = str(t.get("type", "")).lower()

        if "resize" in tname:
            size = (
                t.get("size")
                or t.get("img_scale")
                or t.get("img_size")
                or t.get("input_size")
            )
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                cfg["target_w"], cfg["target_h"] = int(size[0]), int(size[1])

            kr = t.get("keep_ratio", t.get("keepaspectratio", True))
            cfg["keep_ratio"] = bool(kr)

            pv = t.get("pad_val", t.get("pad_value", cfg["pad_val"]))
            if isinstance(pv, (list, tuple)) and len(pv) >= 1:
                cfg["pad_val"] = int(pv[0])
            elif isinstance(pv, (int, float)):
                cfg["pad_val"] = int(pv)

        if "normalize" in tname:
            norm = cfg["normalize"]
            norm["to_rgb"] = bool(t.get("to_rgb", norm["to_rgb"]))
            if "mean" in t:
                m = t["mean"]
                if isinstance(m, (list, tuple)) and len(m) >= 3:
                    norm["mean"] = [float(m[0]), float(m[1]), float(m[2])]
            if "std" in t:
                s = t["std"]
                if isinstance(s, (list, tuple)) and len(s) >= 3:
                    norm["std"] = [float(s[0]), float(s[1]), float(s[2])]
            norm["to_float32"] = bool(t.get("to_float32", norm["to_float32"]))

    return cfg

def letterbox_with_pad(bgr, Wt, Ht, keep_ratio=True, pad_val=114):
    H, W = bgr.shape[:2]

    if keep_ratio:
        scale = min(Wt / W, Ht / H)
        newW = int(round(W * scale))
        newH = int(round(H * scale))
        padX = (Wt - newW) // 2
        padY = (Ht - newH) // 2

        resized = cv2.resize(bgr, (newW, newH), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((Ht, Wt, 3), pad_val, dtype=np.uint8)
        canvas[padY:padY+newH, padX:padX+newW] = resized

        return canvas, scale, padX, padY

    else:
        resized = cv2.resize(bgr, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        scale_x = Wt / W
        return resized, scale_x, 0, 0

def apply_normalize(bgr_img, norm_cfg, force_flip_rgb=None):
    to_rgb = norm_cfg["to_rgb"] if force_flip_rgb is None else force_flip_rgb

    img = bgr_img.copy()
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)

    mean = np.array(norm_cfg["mean"], dtype=np.float32)
    std  = np.array(norm_cfg["std"],  dtype=np.float32)
    img  = (img - mean) / std

    return img

def box_to_xyxy_score(det_row, fmt):
    det_row = det_row.astype(float)
    if fmt == "xyxy":
        x1, y1, x2, y2, sc = det_row
    else:
        cx, cy, w, h, sc = det_row
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
    return x1, y1, x2, y2, sc

def swap_xy_boxes(bboxes_xyxy):
    out = bboxes_xyxy.copy()
    if out.size:
        out[:, [0,1,2,3]] = out[:, [1,0,3,2]]
    return out

def swap_xy_keypoints(kpts):
    out = kpts.copy()
    if out.size:
        out[..., :2] = out[..., :2][..., ::-1]
    return out

def looks_normalized(arr):
    if arr.size == 0:
        return False
    try:
        mx = float(np.nanmax(np.abs(arr)))
    except Exception:
        return False
    return mx <= 2.0

def remap_to_original_image(
    dets_raw, kpts_raw,
    fmt,
    swap_xy,
    pad_mode,
    Wt, Ht,
    scale, padX, padY,
):
    dets_work = dets_raw.copy()
    kpts_work = kpts_raw.copy()
    if swap_xy:
        dets_work = swap_xy_boxes(dets_work)
        kpts_work = swap_xy_keypoints(kpts_work)

    xyxy_list = [box_to_xyxy_score(row, fmt) for row in dets_work]
    x1,y1,x2,y2,score = [np.array(v) for v in zip(*xyxy_list)]

    if looks_normalized(dets_work[:, :4]):
        x1 *= Wt; x2 *= Wt
        y1 *= Ht; y2 *= Ht

    kpts_m = kpts_work.astype(float).copy()
    if looks_normalized(kpts_work[..., :2]):
        kpts_m[..., 0] *= Wt
        kpts_m[..., 1] *= Ht

    if pad_mode == "minus":
        x1 = (x1 - padX) / scale
        y1 = (y1 - padY) / scale
        x2 = (x2 - padX) / scale
        y2 = (y2 - padY) / scale

        kpts_m[..., 0] = (kpts_m[..., 0] - padX) / scale
        kpts_m[..., 1] = (kpts_m[..., 1] - padY) / scale

    elif pad_mode == "nopad":
        x1 = x1 / scale
        y1 = y1 / scale
        x2 = x2 / scale
        y2 = y2 / scale

        kpts_m[..., 0] = kpts_m[..., 0] / scale
        kpts_m[..., 1] = kpts_m[..., 1] / scale

    elif pad_mode == "plus":
        x1 = (x1 + padX) / scale
        y1 = (y1 + padY) / scale
        x2 = (x2 + padX) / scale
        y2 = (y2 + padY) / scale

        kpts_m[..., 0] = (kpts_m[..., 0] + padX) / scale
        kpts_m[..., 1] = (kpts_m[..., 1] + padY) / scale

    det_img = np.stack([x1, y1, x2, y2, score], axis=1)
    return det_img, kpts_m

def score_layout(det_img, kp_img, W, H):
    if det_img.size == 0 or kp_img.size == 0:
        return -1.0

    x1 = det_img[:, 0]; y1 = det_img[:, 1]
    x2 = det_img[:, 2]; y2 = det_img[:, 3]
    px = kp_img[:, :, 0]; py = kp_img[:, :, 1]

    in_box = (
        (px >= x1[:, None]) &
        (px <= x2[:, None]) &
        (py >= y1[:, None]) &
        (py <= y2[:, None])
    )

    in_img = (
        (px >= 0) & (px < W) &
        (py >= 0) & (py < H)
    )

    valid_box = (
        ((x2 - x1) > 2) &
        ((y2 - y1) > 2) &
        (x1 > -0.05 * W) & (y1 > -0.05 * H) &
        (x2 < 1.05 * W) & (y2 < 1.05 * H)
    )

    base_score = 0.95 * (in_box & in_img).mean() + 0.05 * valid_box.mean()

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side_penalty = max(
        np.mean(cx < 0.20 * W),
        np.mean(cx > 0.80 * W),
        np.mean(cy < 0.20 * H),
        np.mean(cy > 0.80 * H),
    )

    if side_penalty > 0.70:
        return -1.0

    score = float(base_score - 0.40 * side_penalty)
    return score

def draw_pose(canvas_bgr, dets_img, kpts_img,
              box_thr=0.05, kpt_thr=0.05):
    out = canvas_bgr.copy()
    H, W = out.shape[:2]

    pt_r = max(2, int(round(min(W, H) * 0.006)))
    ln_t = max(2, int(round(min(W, H) * 0.008)))

    if dets_img.size == 0:
        return out, 0

    order = np.argsort(dets_img[:, 4])[::-1]
    drawn = 0

    for i in order:
        score_i = dets_img[i, 4]
        if score_i < box_thr:
            continue

        x1, y1, x2, y2 = map(_si, dets_img[i, :4])
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), ln_t)

        pts = kpts_img[i][:, :2]
        scr = kpts_img[i][:, 2]
        has_confident = np.any(scr >= kpt_thr)

        for (x, y), s in zip(pts, scr):
            if has_confident and s < kpt_thr:
                continue
            cv2.circle(out, (_si(x), _si(y)), pt_r, (0, 255, 0), -1)

        for a, b in COCO_EDGES:
            if has_confident and (scr[a] < kpt_thr or scr[b] < kpt_thr):
                continue
            xa, ya = pts[a]
            xb, yb = pts[b]
            cv2.line(out, (_si(xa), _si(ya)), (_si(xb), _si(yb)), (0, 255, 0), ln_t)

        drawn += 1

    return out, drawn

def main():
    parser = argparse.ArgumentParser(
        description="RTMO ONNX inference and pose visualization"
    )
    parser.add_argument("--model", required=True, help="path to end2end.onnx")
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--pipeline", required=True, help="path to pipeline.json")
    parser.add_argument("--save", default="rtmo_out.png", help="output image path")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"model not found: {args.model}")
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"image not found: {args.image}")
    if not os.path.isfile(args.pipeline):
        raise FileNotFoundError(f"pipeline not found: {args.pipeline}")

    img_bgr = load_image_bgr_with_exif(args.image)
    H0, W0 = img_bgr.shape[:2]

    pipe_cfg = load_pipeline_config(args.pipeline)
    Wt = pipe_cfg["target_w"]
    Ht = pipe_cfg["target_h"]
    keep_ratio = pipe_cfg["keep_ratio"]
    pad_val = pipe_cfg["pad_val"]
    norm_cfg = pipe_cfg["normalize"]

    print(f"[pipeline] input_size=({Wt}x{Ht}) "
          f"keep_ratio={keep_ratio} pad_val={pad_val} "
          f"normalize={norm_cfg}")

    lb_bgr, scale, padX, padY = letterbox_with_pad(
        img_bgr, Wt, Ht, keep_ratio=keep_ratio, pad_val=pad_val
    )

    cv2.imwrite("debug_letterbox_input.png", lb_bgr)

    input_variants = []
    rgb_flip_candidates = [
        None,
        (not norm_cfg["to_rgb"])
    ]
    for flip_flag in rgb_flip_candidates:
        arr = apply_normalize(lb_bgr, norm_cfg, force_flip_rgb=flip_flag)
        input_variants.append((
            "flipRGB" if flip_flag is not None else "cfgRGB",
            arr
        ))

    sess = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"]
    )
    inp0 = sess.get_inputs()[0]
    in_shape = inp0.shape

    if len(in_shape) == 4 and (in_shape[1] == 3 or in_shape[1] is None):
        layout = "NCHW"
    else:
        layout = "NHWC"
    print(f"[model] layout_guess={layout} input_shape={in_shape} dtype={inp0.type}")

    def run_model(img_arr_float32):
        if layout == "NCHW":
            x = img_arr_float32.transpose(2, 0, 1)[None, ...]
        else:
            x = img_arr_float32[None, ...]
        x = x.astype(np.float32)
        outputs = sess.run(None, {inp0.name: x})
        return outputs

    raw_results = []
    for tag, arr in input_variants:
        outs = run_model(arr)
        if len(outs) < 2:
            continue

        dets = np.array(outs[0])
        kpts = np.array(outs[1])

        if dets.ndim == 3 and dets.shape[0] == 1:
            dets = dets[0]
        if kpts.ndim == 4 and kpts.shape[0] == 1:
            kpts = kpts[0]
        if kpts.ndim == 2 and kpts.shape[-1] == 51:
            kpts = kpts.reshape((-1, 17, 3))

        raw_results.append((tag, dets, kpts))

    if not raw_results:
        cv2.imwrite(args.save, img_bgr)
        print("[warn] empty model output. saved original image.")
        return

    candidates = []
    for (color_tag, dets_raw, kpts_raw) in raw_results:
        if dets_raw.size == 0 or kpts_raw.size == 0:
            continue

        for fmt in ["xyxy", "cxcywh"]:
            for swap_xy_flag in [False, True]:
                for pad_mode in ["minus", "nopad", "plus"]:
                    det_img, kpt_img = remap_to_original_image(
                        dets_raw,
                        kpts_raw,
                        fmt=fmt,
                        swap_xy=swap_xy_flag,
                        pad_mode=pad_mode,
                        Wt=Wt, Ht=Ht,
                        scale=scale,
                        padX=padX,
                        padY=padY,
                    )
                    sc = score_layout(det_img, kpt_img, W0, H0)
                    desc = f"{color_tag}:{fmt}_{'swap' if swap_xy_flag else 'noswap'}_{pad_mode}"
                    candidates.append((sc, desc, det_img, kpt_img))

    if not candidates:
        cv2.imwrite(args.save, img_bgr)
        print("[warn] remap failed. saved original image.")
        return

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_desc, dets_img, kpts_img = candidates[0]
    print(f"[mapping] best='{best_desc}', score={best_score:.3f}")

    drawn_img, person_count = draw_pose(
        img_bgr, dets_img, kpts_img,
        box_thr=0.05,
        kpt_thr=0.05
    )
    cv2.imwrite(args.save, drawn_img)
    print(f"[done] persons={person_count}, out='{args.save}'")


if __name__ == "__main__":
    main()
