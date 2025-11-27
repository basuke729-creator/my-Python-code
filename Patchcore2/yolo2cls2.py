#!/usr/bin/env python
# yolo2cls_pad_dirs.py
# - 複数データセット (IMG_DIR:LABEL_DIR) 対応
# - .names からクラス名読み込み
# - クラスごとフォルダ分け出力
# - 左右・上下で別倍率 (--scale-left/right/top/bottom)
# - 画像端はクリップ
# - パディング用の黒キャンバスは使わず、元画像からそのままcropしてリサイズ

import argparse
from pathlib import Path
import os
from datetime import datetime

from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    ap = argparse.ArgumentParser("YOLO -> Classification crops (dir, no black padding)")
    # IMG_DIR:LABEL_DIR を複数指定可能
    ap.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="形式 'IMG_DIR:LABEL_DIR' をスペース区切りで複数指定可"
    )
    ap.add_argument(
        "--names",
        required=True,
        help=".names ファイルのパス（クラス名を1行ずつ）"
    )
    ap.add_argument(
        "--out",
        required=True,
        help="出力ルートディレクトリ（クラスごとにサブフォルダ作成）"
    )
    ap.add_argument(
        "--size",
        type=int,
        default=384,
        help="出力画像の一辺のサイズ（正方形にリサイズ）"
    )

    # 拡大倍率（左右・上下で個別に）
    ap.add_argument(
        "--scale-left", type=float, default=1.25,
        help="左方向にどれだけ広げるかの倍率（1.0でそのまま）"
    )
    ap.add_argument(
        "--scale-right", type=float, default=1.25,
        help="右方向にどれだけ広げるかの倍率"
    )
    ap.add_argument(
        "--scale-top", type=float, default=1.25,
        help="上方向にどれだけ広げるかの倍率"
    )
    ap.add_argument(
        "--scale-bottom", type=float, default=1.25,
        help="下方向にどれだけ広げるかの倍率"
    )

    return ap.parse_args()


def load_class_names(names_path: Path):
    with open(names_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    # 空行を除外
    return [l for l in lines if l != ""]


def yolo_to_pixel_box(
    xc: float,
    yc: float,
    w: float,
    h: float,
    W: int,
    H: int,
    scale_left: float,
    scale_right: float,
    scale_top: float,
    scale_bottom: float,
):
    """
    YOLO 正規化 bbox (xc,yc,w,h in [0,1]) を元に、
    左右・上下それぞれ別倍率で拡大した矩形を画像座標で返す。

    - まずピクセル単位の bbox (cx,cy,bw,bh) を計算
    - 左右で半幅 * scale_left/right、上下で半高 * scale_top/bottom
    - 画像端でクリップ（はみ出し部分は捨てる。黒で埋めない）
    """
    cx = xc * W
    cy = yc * H
    bw = w * W
    bh = h * H

    half_w = bw / 2.0
    half_h = bh / 2.0

    left   = cx - half_w * scale_left
    right  = cx + half_w * scale_right
    top    = cy - half_h * scale_top
    bottom = cy + half_h * scale_bottom

    # クリップ
    left   = max(0.0, min(W - 1.0, left))
    right  = max(1.0, min(W * 1.0, right))
    top    = max(0.0, min(H - 1.0, top))
    bottom = max(1.0, min(H * 1.0, bottom))

    if right <= left or bottom <= top:
        return None

    return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))


def process_one_image(
    img_path: Path,
    label_path: Path,
    out_root: Path,
    class_names,
    out_size: int,
    scale_left: float,
    scale_right: float,
    scale_top: float,
    scale_bottom: float,
    log_f,
):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        log_f.write(f"[WARN] open failed: {img_path} ({e})\n")
        return 0

    W, H = img.size

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
    except FileNotFoundError:
        # ラベルがない画像はスキップ
        return 0

    base_name = img_path.stem
    saved = 0

    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            log_f.write(f"[WARN] bad line in {label_path}: '{line}'\n")
            continue

        try:
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
        except Exception:
            log_f.write(f"[WARN] parse error in {label_path}: '{line}'\n")
            continue

        if cls_id < 0 or cls_id >= len(class_names):
            log_f.write(f"[WARN] invalid class id {cls_id} in {label_path}\n")
            continue

        box = yolo_to_pixel_box(
            xc, yc, w, h,
            W, H,
            scale_left, scale_right,
            scale_top, scale_bottom
        )
        if box is None:
            log_f.write(f"[WARN] zero-size box in {img_path} line='{line}'\n")
            continue

        x1, y1, x2, y2 = box
        # 元画像から背景ごと crop
        crop = img.crop((x1, y1, x2, y2))
        # そのまま正方形にリサイズ（歪みは出るが黒ベタは作らない）
        crop = crop.resize((out_size, out_size), Image.BICUBIC)

        cls_name = class_names[cls_id]
        out_dir = out_root / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{base_name}_{idx}.jpg"
        # 同名衝突を避ける（ほぼ無いと思うが念のため）
        k = 1
        while out_path.exists():
            out_path = out_dir / f"{base_name}_{idx}_{k}.jpg"
            k += 1

        try:
            crop.save(out_path, quality=95)
            saved += 1
        except Exception as e:
            log_f.write(f"[WARN] save failed {out_path}: {e}\n")

    return saved


def main():
    args = parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(Path(args.names))
    print(f"[INFO] class names: {class_names}")

    # ログファイル
    log_path = out_root / "_convert.log"
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n=== run at {datetime.now().isoformat()} ===\n")
        log_f.write(f"datasets   : {args.dataset}\n")
        log_f.write(f"names      : {args.names}\n")
        log_f.write(f"size       : {args.size}\n")
        log_f.write(
            f"scale L/R/T/B = {args.scale_left}, {args.scale_right}, "
            f"{args.scale_top}, {args.scale_bottom}\n"
        )

        total_saved = 0
        total_imgs = 0

        for pair in args.dataset:
            # "IMG_DIR:LABEL_DIR"
            if ":" not in pair:
                log_f.write(f"[WARN] bad --dataset entry: '{pair}' (skip)\n")
                continue
            img_dir_str, label_dir_str = pair.split(":", 1)
            img_dir = Path(img_dir_str)
            label_dir = Path(label_dir_str)

            if not img_dir.is_dir():
                log_f.write(f"[WARN] img_dir not found: {img_dir}\n")
                continue
            if not label_dir.is_dir():
                log_f.write(f"[WARN] label_dir not found: {label_dir}\n")
                continue

            log_f.write(f"[INFO] process dataset IMG={img_dir} LABEL={label_dir}\n")

            img_files = sorted(
                p for p in img_dir.rglob("*")
                if p.suffix.lower() in VALID_EXTS and p.is_file()
            )
            log_f.write(f"[INFO] images found: {len(img_files)}\n")

            for img_path in img_files:
                label_path = label_dir / (img_path.stem + ".txt")
                if not label_path.exists():
                    # ラベルがない画像はスキップ
                    continue
                saved = process_one_image(
                    img_path,
                    label_path,
                    out_root,
                    class_names,
                    out_size=args.size,
                    scale_left=args.scale_left,
                    scale_right=args.scale_right,
                    scale_top=args.scale_top,
                    scale_bottom=args.scale_bottom,
                    log_f=log_f,
                )
                total_saved += saved
                total_imgs += 1

        log_f.write(f"[INFO] processed images: {total_imgs}, crops saved: {total_saved}\n")

    print(f"[DONE] crops saved: {total_saved}  (log: {log_path})")


if __name__ == "__main__":
    main()
