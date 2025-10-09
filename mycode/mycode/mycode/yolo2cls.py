#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolo2cls_pad.py
YOLO(txt) → 画像分類用フォルダ（ImageFolder互換）変換スクリプト
・上端固定で左右/下方向を倍率拡張（端ではclip）
・切り出し後、アスペクト比を保持したまま正方形へパディング（ImageOps.pad）
出力:
  out_root/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg   # --val-split > 0 のとき
"""

import argparse
from pathlib import Path
from PIL import Image, ImageOps
import random
from typing import List, Tuple, Optional

# ---------- utilities ----------
def load_names(names_path: Optional[str]) -> Optional[List[str]]:
    if not names_path:
        return None
    p = Path(names_path)
    if not p.exists():
        raise FileNotFoundError(f".names file not found: {names_path}")
    names = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return names

def yolo_norm_to_xyxy(img_w: int, img_h: int, cx: float, cy: float, w: float, h: float):
    x = cx * img_w; y = cy * img_h
    bw = w * img_w; bh = h * img_h
    x1 = int(round(x - bw / 2)); y1 = int(round(y - bh / 2))
    x2 = int(round(x + bw / 2)); y2 = int(round(y + bh / 2))
    return x1, y1, x2, y2

def expand_keep_top_clip_edges(x1, y1, x2, y2, W, H, scale_lr=1.25, scale_bottom=1.25):
    """
    上端(y1)固定のまま、左右(幅)と下端(高さ)を拡張。
    画像端に当たった部分は clip（平行移動しない）。
    """
    # 幅拡張（中心基準）
    w = max(1, x2 - x1)
    cx = (x1 + x2) / 2.0
    new_w = max(1, int(round(w * scale_lr)))
    nx1 = int(round(cx - new_w / 2))
    nx2 = int(round(cx + new_w / 2))

    # 高さ拡張（上固定で下へ）
    h = max(1, y2 - y1)
    new_h = max(1, int(round(h * scale_bottom)))
    ny1 = y1
    ny2 = y1 + new_h

    # 端でクリップ
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(W, nx2)
    ny2 = min(H, ny2)

    # 退化防止
    if nx2 <= nx1: nx2 = min(W, nx1 + 1)
    if ny2 <= ny1: ny2 = min(H, ny1 + 1)
    return nx1, ny1, nx2, ny2

def list_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])

def save_crop(img_path: Path, box_xyxy, out_dir: Path, stem: str, idx: int,
              resize_size: int, pad_color=(0, 0, 0)):
    """
    切り出し→アスペクト比維持パディング→正方形resize保存
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        crop = im.crop(box_xyxy)  # (left, upper, right, lower)
        # アスペクト比を維持しながら正方形にパディングしてリサイズ
        crop = ImageOps.pad(crop, (resize_size, resize_size), color=pad_color, centering=(0.5, 0.5))
        crop.save(out_dir / f"{stem}_{idx}.jpg", quality=95)

def precreate_class_dirs(out_root: Path, classes: List[str], create_val: bool):
    for split in (["train", "val"] if create_val else ["train"]):
        for cname in classes:
            (out_root / split / cname).mkdir(parents=True, exist_ok=True)

# ---------- core ----------
def process_one_pair(images_dir: Path, labels_dir: Path, out_root: Path, names: Optional[List[str]],
                     val_split: float, scale_lr: float, scale_bottom: float, min_side: int,
                     resize_size: int, pad_color):
    images = list_images(images_dir)
    if not images:
        print(f"[WARN] No images under: {images_dir}")
        return 0

    saved = 0
    for img_path in images:
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            continue

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size
        except Exception as e:
            print(f"[WARN] Skip broken image: {img_path} ({e})")
            continue

        # ラベル読み込み（UTF-8優先→cp932フォールバック）
        try:
            text = label_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = label_path.read_text(encoding="cp932", errors="ignore")

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        idx = 0
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
            except Exception:
                continue

            x1, y1, x2, y2 = yolo_norm_to_xyxy(W, H, cx, cy, bw, bh)
            x1, y1, x2, y2 = expand_keep_top_clip_edges(x1, y1, x2, y2, W, H,
                                                        scale_lr=scale_lr, scale_bottom=scale_bottom)
            if (x2 - x1) < min_side or (y2 - y1) < min_side:
                continue

            cname = names[cid] if (names and 0 <= cid < len(names)) else str(cid)
            split = "val" if (val_split > 0 and random.random() < val_split) else "train"
            out_dir = out_root / split / cname
            save_crop(img_path, (x1, y1, x2, y2), out_dir, stem, idx, resize_size, pad_color)
            saved += 1
            idx += 1
    return saved

def parse_args():
    ap = argparse.ArgumentParser(description="YOLO→分類フォルダ（余白パディングで正方形化）")
    # 入力指定（どちらか/併用可）
    ap.add_argument("--dataset", action="append",
                    help="画像:ラベル のペアを 'IMAGES_DIR:LABELS_DIR' 形式で（複数可）。Windowsは --images/--labels 推奨。")
    ap.add_argument("--images", action="append", help="画像ディレクトリ（複数可・--labels と同数）")
    ap.add_argument("--labels", action="append", help="ラベルディレクトリ（複数可・--images と同数）")

    ap.add_argument("--out", required=True, help="出力ルート（ImageFolder互換構成）")
    ap.add_argument("--names", default="", help=".names（1行1クラス名）。省略可（数値IDフォルダになる）")
    ap.add_argument("--val-split", type=float, default=0.0, help="検証に回す割合（例 0.1）")

    # 拡大倍率（コマンドから即変更可）
    ap.add_argument("--scale-lr", type=float, default=1.25, help="左右(幅)の拡大倍率")
    ap.add_argument("--scale-bottom", type=float, default=1.25, help="下方向(高さ)の拡大倍率")

    ap.add_argument("--min-side", type=int, default=32, help="切り出し後の最小辺ピクセル")

    # パディング付きリサイズ設定
    ap.add_argument("--resize", type=int, default=384, help="正方形の出力サイズ（例: 384 → 384x384）")
    ap.add_argument("--pad-color", type=str, default="0,0,0",
                    help="パディング色を 'R,G,B' で指定（例: '0,0,0' 黒 / '255,255,255' 白 / '128,128,128' グレー）")

    ap.add_argument("--seed", type=int, default=42, help="val split の乱数シード")
    ap.add_argument("--precreate-dirs", action="store_true",
                    help="namesがある場合に train/val の全クラスサブフォルダを事前作成")
    return ap.parse_args()

def parse_color(s: str):
    try:
        parts = [int(v) for v in s.split(",")]
        if len(parts) != 3:
            raise ValueError
        return tuple(max(0, min(255, v)) for v in parts)
    except Exception:
        raise SystemExit(f"--pad-color は 'R,G,B'（0-255）で指定してください。例: 0,0,0")

def main():
    args = parse_args()
    random.seed(args.seed)

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    names = load_names(args.names) if args.names else None
    pad_color = parse_color(args.pad-color)

    # 入力ペアの解決
    pairs: List[Tuple[Path, Path]] = []
    if args.dataset:
        for ds in args.dataset:
            if ":" not in ds:
                raise SystemExit(f"--dataset は 'IMAGES_DIR:LABELS_DIR' 形式で指定してください: {ds}")
            img_dir_str, lbl_dir_str = ds.rsplit(":", 1)  # Windowsドライブ文字対策
            img_dir, lbl_dir = Path(img_dir_str), Path(lbl_dir_str)
            if not img_dir.exists(): raise SystemExit(f"images dir not found: {img_dir}")
            if not lbl_dir.exists(): raise SystemExit(f"labels dir not found: {lbl_dir}")
            pairs.append((img_dir, lbl_dir))
    if args.images or args.labels:
        if not (args.images and args.labels) or len(args.images) != len(args.labels):
            raise SystemExit("--images と --labels は同数指定してください。")
        for img_dir_str, lbl_dir_str in zip(args.images, args.labels):
            img_dir, lbl_dir = Path(img_dir_str), Path(lbl_dir_str)
            if not img_dir.exists(): raise SystemExit(f"images dir not found: {img_dir}")
            if not lbl_dir.exists(): raise SystemExit(f"labels dir not found: {lbl_dir}")
            pairs.append((img_dir, lbl_dir))

    if not pairs:
        raise SystemExit("入力データセットが指定されていません。--dataset か --images/--labels を使ってください。")

    if args.precreate_dirs and names:
        precreate_class_dirs(out_root, names, create_val=(args.val_split > 0))

    total_saved = 0
    for i, (img_dir, lbl_dir) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] images={img_dir}  labels={lbl_dir}")
        saved = process_one_pair(
            images_dir=img_dir, labels_dir=lbl_dir, out_root=out_root, names=names,
            val_split=args.val_split, scale_lr=args.scale_lr, scale_bottom=args.scale_bottom,
            min_side=args.min_side, resize_size=args.resize, pad_color=pad_color
        )
        print(f"  -> saved crops: {saved}")
        total_saved += saved

    print(f"Done. Total saved crops: {total_saved}")
    print(f"Output root: {out_root.resolve()}")

if __name__ == "__main__":
    main()
