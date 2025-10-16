# predict.py  (complete, fixed)
import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import torch
import timm
from PIL import Image, ImageOps
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ------------------------ 画像のサイズ変更（3モード） ------------------------
def letterbox_pad(img: Image.Image, dst_size: int, bg: str = "black") -> Image.Image:
    """
    縦横比を保持したまま最長辺を dst_size に縮小し、余白で正方形キャンバスに配置。
    bg: "black" | "white" | "mean"（画像の平均色）
    """
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
    """
    推論前処理を構築。
    - pad: レターボックス（縦横比保持＋正方形化）→ ToTensor → Normalize（推奨）
    - crop: 短辺を合わせてから CenterCrop（中央に対象がある想定で強い）
    - stretch: 強制正方形（従来互換、最も単純）
    """
    if resize_mode == "pad":
        tf = transforms.Compose([
            transforms.Lambda(lambda im: letterbox_pad(im, img_size, bg)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif resize_mode == "crop":
        tf = transforms.Compose([
            transforms.Resize(img_size),          # 短辺=img_size にスケール
            transforms.CenterCrop(img_size),     # 正方形中央トリム
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # "stretch"
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return tf


# ------------------------ モデル＆クラス名の読み込み ------------------------
def _infer_num_classes_from_state_dict(state_dict: dict) -> int:
    """
    timm のモデルは最後の線形層名が一定でないため、
    state_dict から「形が [C, D] の weight（=2次元）で、対応する bias があるもの」を探して
    C を num_classes とみなす。
    """
    candidates: List[Tuple[str, int]] = []
    for k, v in state_dict.items():
        if not (k.endswith("weight") and v.ndim == 2):
            continue
        bias_key = k.replace("weight", "bias")
        if bias_key in state_dict and state_dict[bias_key].ndim == 1:
            out_dim = v.shape[0]
            candidates.append((k, out_dim))
    if not candidates:
        # フォールバック：もっとも小さい出力次元を採用（多くの場合クラス数が最小）
        lin_dims = [v.shape[0] for k, v in state_dict.items() if k.endswith("weight") and v.ndim == 2]
        if lin_dims:
            return int(min(lin_dims))
        raise RuntimeError("Could not infer num_classes from state_dict.")
    # 一般に最後段（classifier/head）が候補の最後になることが多い
    return int(candidates[-1][1])


def load_model(ckpt_path: str, model_name: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)  # 互換：直に state_dict が保存されている場合も考慮
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


# ------------------------ データ反復・推論ユーティリティ ------------------------
def iter_images(path: Path):
    if path.is_file():
        if path.suffix.lower() in VALID_EXTS:
            yield path
    else:
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                yield p


def predict_probs(model, img: Image.Image, tf, device, tta: bool = False) -> torch.Tensor:
    """
    1枚の画像に対して確率ベクトルを返す。tta=True で水平反転の平均。
    """
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    if tta:
        img_hf = ImageOps.mirror(img)
        x2 = tf(img_hf).unsqueeze(0).to(device)
        with torch.no_grad():
            logits2 = model(x2)
            probs2 = torch.softmax(logits2, dim=1)
        probs = (probs + probs2) / 2.0

    return probs.squeeze(0).cpu()


def topk_from_probs(probs: torch.Tensor, class_names: List[str], k: int):
    k = min(k, probs.numel())
    conf, idx = torch.topk(probs, k)
    conf = conf.tolist()
    idx = idx.tolist()
    names = [class_names[i] for i in idx]
    return list(zip(names, conf))


# ------------------------ メイン ------------------------
def main():
    ap = argparse.ArgumentParser("Predict with EfficientNet-V2 (enhanced)")
    ap.add_argument("--ckpt", required=True, help="best.ckpt のパス")
    ap.add_argument("--model", required=True, help="学習時と同じ timm モデル名（例: tf_efficientnetv2_s_in21k_ft_in1k）")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--input", required=True, help="画像ファイル or ディレクトリ")
    ap.add_argument("--resize-mode", choices=["pad", "crop", "stretch"], default="pad",
                    help="pad=レターボックス(推奨) / crop=短辺合わせ+CenterCrop / stretch=強制正方形")
    ap.add_argument("--bg", choices=["black", "white", "mean"], default="black",
                    help="--resize-mode pad の背景色")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Top-1 確率がこの値未満なら 'unknown' とする (例: 0.6)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tta", action="store_true", help="水平反転 TTA を有効化")
    ap.add_argument("--save-csv", default=None, help="予測結果をCSV保存するパス（例: runs/preds.csv）")
    ap.add_argument("--save-per-class", default=None,
                    help="予測クラスごとに画像をコピーするディレクトリ（例: runs/preds_split）")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tf = build_transform(args.img_size, args.resize_mode, args.bg)

    model, class_names = load_model(args.ckpt, args.model, device)

    inp = Path(args.input)
    rows: List[dict] = []
    if args.save_per_class:
        Path(args.save_per_class).mkdir(parents=True, exist_ok=True)

    for img_path in iter_images(inp):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] open failed: {img_path} ({e})")
            continue

        probs = predict_probs(model, img, tf, device, tta=args.tta)
        topk = topk_from_probs(probs, class_names, args.topk)

        top1_name, top1_conf = topk[0]
        pred_name = top1_name
        if args.threshold > 0.0 and top1_conf < args.threshold:
            pred_name = "unknown"

        # コンソール出力
        line = ", ".join([f"{n}:{c:.3f}" for n, c in topk])
        suffix = f"  (pred='{pred_name}')" if pred_name != top1_name else ""
        print(f"{img_path.name} -> {line}{suffix}")

        # CSV 行
        row = {
            "path": str(img_path),
            "pred": pred_name,
            "top1": f"{topk[0][0]}:{topk[0][1]:.6f}",
        }
        for i in range(1, min(3, len(topk))):
            row[f"top{i+1}"] = f"{topk[i][0]}:{topk[i][1]:.6f}"
        rows.append(row)

        # クラス別仕分け保存（ファイル名衝突回避付き）
        if args.save_per_class:
            dst_dir = Path(args.save_per_class) / pred_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / img_path.name
            if dst_path.exists():
                stem, suf = dst_path.stem, dst_path.suffix
                i = 1
                while (dst_dir / f"{stem}_{i}{suf}").exists():
                    i += 1
                dst_path = dst_dir / f"{stem}_{i}{suf}"
            try:
                shutil.copy2(img_path, dst_path)
            except Exception:
                # 失敗時はPIL経由で保存（フォーマット変換のリスクあり）
                img.save(dst_path)

    # CSV 保存
    if args.save_csv:
        outp = Path(args.save_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fields = ["path", "pred", "top1", "top2", "top3"]
        with open(outp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"Saved CSV -> {outp}")


if __name__ == "__main__":
    main()
