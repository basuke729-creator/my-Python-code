#!/usr/bin/env python
# yolo2cls.py
# YOLO形式のアノテーションからクラス分類用クロップ画像を作る
# - 黒キャンバスは使わず、元画像から背景ごとクロップ
# - クロップ範囲はBBoxをscale倍した正方形を基本とし、画像外はクリップ

import argparse
from pathlib import Path
import os

from PIL import Image

def parse_args():
    ap = argparse.ArgumentParser("YOLO -> cls crops (no black padding)")
    ap.add_argument("--img-dir", required=True,
                    help="元画像のディレクトリ（例: images/）")
    ap.add_argument("--label-dir", required=True,
                    help="YOLOラベル(.txt)のディレクトリ（例: labels/）")
    ap.add_argument("--names", required=True,
                    help="クラス名一覧の .names ファイル")
    ap.add_argument("--out-dir", required=True,
                    help="切り出し結果を保存するルートディレクトリ")
    ap.add_argument("--out-size", type=int, default=384,
                    help="出力画像の一辺のピクセル数（正方形にリサイズ）")
    ap.add_argument("--scale", type=float, default=1.25,
                    help="BBoxをどれくらい拡大してクロップするか（1.25 = 25%拡大）")
    ap.add_argument("--img-exts", nargs="+",
                    default=[".jpg", ".jpeg", ".png", ".bmp"],
                    help="対象とする画像拡張子")
    return ap.parse_args()


def load_class_names(names_path: Path):
    with open(names_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    return [l for l in lines if l != ""]


def yolo_to_pixel_box(xc, yc, w, h, W, H, scale=1.0):
    """
    YOLO正規化座標 -> 画像座標の正方形BBox（スケール拡大込み）
    xc, yc, w, h: 正規化(0-1)
    W, H: 画像サイズ
    """
    cx = xc * W
    cy = yc * H
    bw = w * W * scale
    bh = h * H * scale

    # 正方形にする：長い方に合わせる
    side = max(bw, bh)
    x1 = cx - side / 2.0
    y1 = cy - side / 2.0
    x2 = cx + side / 2.0
    y2 = cy + side / 2.0

    # 画像範囲にクリップ（はみ出した部分は捨てるだけで黒塗りはしない）
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(1, min(W, x2))
    y2 = max(1, min(H, y2))

    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def process_one_image(img_path: Path, label_path: Path,
                      out_root: Path, class_names, out_size: int, scale: float):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    base_name = img_path.stem
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        if cls_id < 0 or cls_id >= len(class_names):
            continue

        x1, y1, x2, y2 = yolo_to_pixel_box(xc, yc, w, h, W, H, scale=scale)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img.crop((x1, y1, x2, y2))  # 背景ごとそのまま切り出し
        crop = crop.resize((out_size, out_size), Image.BICUBIC)

        cls_name = class_names[cls_id]
        out_dir = out_root / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{base_name}_{idx}.jpg"
        crop.save(out_path, quality=95)


def main():
    args = parse_args()

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(Path(args.names))

    img_exts = {e.lower() for e in args.img_exts}

    img_files = sorted(
        [p for p in img_dir.iterdir()
         if p.is_file() and p.suffix.lower() in img_exts]
    )

    print(f"[INFO] images: {len(img_files)}")
    for img_path in img_files:
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        process_one_image(
            img_path, label_path,
            out_root, class_names,
            out_size=args.out_size,
            scale=args.scale
        )

    print("[DONE] all crops written to:", out_root)


if __name__ == "__main__":
    main()
