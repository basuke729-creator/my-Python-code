# predict.py  (inference + evaluation + speed)
# v1.7-fixed
import argparse, csv, os, shutil, json, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import timm
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

print("predict.py v1.7-fixed (speed + per-class summary)")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ------------------------ 日本語フォント対応 ------------------------
def set_japanese_font(font_path: str = ""):
    try:
        if font_path and Path(font_path).exists():
            fp = font_manager.FontProperties(fname=font_path)
            matplotlib.rcParams["font.family"] = fp.get_name()
            return
        candidates = [
            "Noto Sans CJK JP", "Noto Sans JP", "Noto Sans CJK JP Regular",
            "IPAexGothic", "IPAGothic", "TakaoPGothic", "VL PGothic"
        ]
        avail = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in avail:
                matplotlib.rcParams["font.family"] = name
                return
    except Exception:
        pass

# ------------------------ 画像のサイズ変更 ------------------------
def letterbox_pad(img: Image.Image, dst_size: int, bg: str = "black") -> Image.Image:
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (dst_size, dst_size), (0, 0, 0))
    scale = dst_size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img_resized = img.resize((nw, nh), Image.BICUBIC)

    if bg == "white":
        canvas = Image.new("RGB", (dst_size, dst_size), (255, 255, 255))
    elif bg == "mean":
        thumb = img.resize((1, 1), Image.BILINEAR)
        canvas = Image.new("RGB", (dst_size, dst_size), tuple(thumb.getpixel((0, 0))))
    else:
        canvas = Image.new("RGB", (dst_size, dst_size), (0, 0, 0))

    x = (dst_size - nw) // 2
    y = (dst_size - nh) // 2
    canvas.paste(img_resized, (x, y))
    return canvas

def build_transform(img_size: int, resize_mode: str, bg: str):
    if resize_mode == "pad":
        tf = transforms.Compose([
            transforms.Lambda(lambda im: letterbox_pad(im, img_size, bg)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif resize_mode == "crop":
        tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return tf

# ------------------------ モデル読み込み ------------------------
def _infer_num_classes_from_state_dict(state_dict: dict) -> int:
    candidates = []
    for k, v in state_dict.items():
        if not (k.endswith("weight") and getattr(v, "ndim", 0) == 2):
            continue
        bias_key = k.replace("weight", "bias")
        if bias_key in state_dict and getattr(state_dict[bias_key], "ndim", 0) == 1:
            candidates.append((k, v.shape[0]))
    if not candidates:
        lin_dims = [v.shape[0] for k, v in state_dict.items() if k.endswith("weight") and getattr(v, "ndim", 0) == 2]
        if lin_dims:
            return int(min(lin_dims))
        raise RuntimeError("Could not infer num_classes from state_dict.")
    return int(candidates[-1][1])

def load_model(ckpt_path: str, model_name: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    class_names = ckpt.get("class_names", None)
    if class_names is not None:
        num_classes = len(class_names)
    else:
        num_classes = _infer_num_classes_from_state_dict(state)
        class_names = [str(i) for i in range(num_classes)]
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, class_names

# ------------------------ 推論ユーティリティ ------------------------
def iter_images(path: Path):
    if path.is_file():
        if path.suffix.lower() in VALID_EXTS:
            yield path
    else:
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                yield p

# ------------------------ メイン ------------------------
def main():
    ap = argparse.ArgumentParser("Predict EfficientNet-V2 (with speed test)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--input", required=True)
    ap.add_argument("--resize-mode", choices=["pad", "crop", "stretch"], default="pad")
    ap.add_argument("--bg", choices=["black", "white", "mean"], default="black")
    ap.add_argument("--speed", action="store_true", help="速度を測定")
    ap.add_argument("--speed-csv", default=None, help="速度結果をCSV保存")
    ap.add_argument("--warmup", type=int, default=5, help="ウォームアップ除外枚数")
    ap.add_argument("--no-sync", action="store_true", help="CUDA同期なし")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu if hasattr(args, 'cpu') else not torch.cuda.is_available() else "cuda")

    # 🔧 修正済み：resize-mode → resize_mode
    tf = build_transform(args.img_size, args.resize_mode, args.bg)

    model, class_names = load_model(args.ckpt, args.model, device)
    inp_root = Path(args.input)

    prep_ms_list, infer_ms_list, total_ms_list = []

    for img_path in iter_images(inp_root):
        with open(img_path, "rb") as fh:
            img = Image.open(fh).convert("RGB")

        # 前処理・推論時間計測
        t0 = time.perf_counter()
        x = tf(img).unsqueeze(0).to(device)
        t1 = time.perf_counter()
        with torch.no_grad():
            logits = model(x)
        if device.type == "cuda" and not args.no_sync:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        prep_ms = (t1 - t0) * 1000
        infer_ms = (t2 - t1) * 1000
        total_ms = (t2 - t0) * 1000

        if len(total_ms_list) >= args.warmup:
            prep_ms_list.append(prep_ms)
            infer_ms_list.append(infer_ms)
            total_ms_list.append(total_ms)

    # 結果出力
    def stats(x):
        x = np.array(x)
        return dict(mean=float(x.mean()), p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)))

    s_prep = stats(prep_ms_list)
    s_infer = stats(infer_ms_list)
    s_total = stats(total_ms_list)
    imgs = len(total_ms_list)
    sec = sum(total_ms_list) / 1000
    ips = imgs / sec if sec > 0 else 0

    print("\n=== Inference Speed ===")
    print(f"Samples: {imgs}, Total Time: {sec:.3f}s, Throughput: {ips:.2f} img/s")
    print(f"Preprocess (ms): mean={s_prep['mean']:.2f}, p95={s_prep['p95']:.2f}")
    print(f"Inference (ms): mean={s_infer['mean']:.2f}, p95={s_infer['p95']:.2f}")
    print(f"Total (ms): mean={s_total['mean']:.2f}, p95={s_total['p95']:.2f}")

if __name__ == "__main__":
    main()
