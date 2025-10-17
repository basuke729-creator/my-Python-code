#!/usr/bin/env python3
# yolo2cls_pad_dirs.py  (robust, logged)
import argparse, sys, re, json
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict
from PIL import Image, ImageOps
from tqdm import tqdm
import math
import time

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def parse_args():
    ap = argparse.ArgumentParser("YOLO -> classification crops (with per-side scaling, padding, robust logging)")
    ap.add_argument("--dataset", action="append", required=True,
                    help="画像とYOLOラベルのペア 'IMAGES_DIR:LABELS_DIR'（複数可、--dataset を繰り返し）")
    ap.add_argument("--out", required=True, help="出力ルート（クラスごとのサブフォルダを作成）")
    ap.add_argument("--names", required=False, help=".names ファイル（各行がクラス名）。省略時は 0..N-1 を逐次発見")
    ap.add_argument("--scale-left", type=float, default=1.0, help="左方向倍率（例 1.25）")
    ap.add_argument("--scale-right", type=float, default=1.25, help="右方向倍率")
    ap.add_argument("--scale-top", type=float, default=1.0, help="上方向倍率")
    ap.add_argument("--scale-bottom", type=float, default=1.25, help="下方向倍率")
    ap.add_argument("--pad-bg", default="black",
                    help="パディング背景（black|white|mean|#RRGGBB）")
    ap.add_argument("--img-size", type=int, default=384, help="出力正方形サイズ")
    ap.add_argument("--min-side", type=int, default=2, help="切り出し後の最小幅/高さ(px)。未満はスキップ")
    ap.add_argument("--max-per-image", type=int, default=9999, help="1画像からの最大切り出し数")
    ap.add_argument("--only-classes", default="", help="このクラスID/名のみ対象（カンマ区切り）例: '0,4,5' or '脚立1段目の人,脚立2段目の人'")
    ap.add_argument("--skip-classes", default="", help="このクラスID/名を除外")
    ap.add_argument("--keep-empty-dirs", action="store_true", help="画像が無くても全クラスの空フォルダを作成する")
    ap.add_argument("--log", default="", help="詳細ログ保存パス（指定すると原因が追える）")
    return ap.parse_args()

def load_names(names_path: Optional[str]) -> Optional[List[str]]:
    if not names_path:
        return None
    p = Path(names_path)
    if not p.exists():
        raise FileNotFoundError(f".names が見つかりません: {p}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines

def color_from_bg(bg: str, img: Optional[Image.Image]=None) -> Tuple[int,int,int]:
    bg = bg.lower()
    if bg == "black":
        return (0,0,0)
    if bg == "white":
        return (255,255,255)
    if bg == "mean" and img is not None:
        thumb = img.resize((1,1), Image.BILINEAR)
        return tuple(thumb.getpixel((0,0)))
    # #RRGGBB
    if re.fullmatch(r"#?[0-9a-f]{6}", bg):
        hexv = bg[1:] if bg.startswith("#") else bg
        return tuple(int(hexv[i:i+2], 16) for i in (0,2,4))
    # デフォルト黒
    return (0,0,0)

def safe_crop(img: Image.Image, box: Tuple[int,int,int,int]) -> Image.Image:
    # PIL は box=(left, top, right, bottom) で右下は排他的。ここではすべてクリップ済み前提
    return img.crop(box)

def letterbox_square(img: Image.Image, dst: int, bg_color: Tuple[int,int,int]) -> Image.Image:
    w, h = img.size
    if w==dst and h==dst:
        return img
    scale = dst / max(w,h)
    nw, nh = max(1,int(round(w*scale))), max(1,int(round(h*scale)))
    im2 = img.resize((nw,nh), Image.BICUBIC)
    canvas = Image.new("RGB", (dst,dst), bg_color)
    x = (dst - nw) // 2
    y = (dst - nh) // 2
    canvas.paste(im2, (x,y))
    return canvas

def parse_line(line: str) -> Optional[Tuple[int,float,float,float,float]]:
    # YOLO: <class> <cx> <cy> <w> <h>  （正規化）
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        return cls, cx, cy, w, h
    except Exception:
        return None

def yolo_to_xyxy(cx, cy, w, h, W, H):
    # 正規化→絶対座標
    x = (cx - w/2) * W
    y = (cy - h/2) * H
    x2 = (cx + w/2) * W
    y2 = (cy + h/2) * H
    return x, y, x2, y2

def scale_box_sides(x1, y1, x2, y2, W, H, sL, sR, sT, sB):
    # 各辺を独立に拡大。中心を固定せず “辺を外側へ” 拡げるイメージ
    # 例：sR=1.25 → 右側を 1.25 倍へ（増分だけ右へ拡張）
    # 現サイズ
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    # 増分
    dxL = (sL - 1.0) * w
    dxR = (sR - 1.0) * w
    dyT = (sT - 1.0) * h
    dyB = (sB - 1.0) * h
    x1_ = x1 - dxL
    x2_ = x2 + dxR
    y1_ = y1 - dyT
    y2_ = y2 + dyB
    # 画面端でクリップ
    x1_ = max(0.0, x1_)
    y1_ = max(0.0, y1_)
    x2_ = min(float(W), x2_)
    y2_ = min(float(H), y2_)
    # 最低1px確保（丸め時ゼロにならないよう）
    if x2_ - x1_ < 1.0:
        midx = (x1_ + x2_) / 2.0
        x1_ = max(0.0, midx - 0.5)
        x2_ = min(float(W), midx + 0.5)
    if y2_ - y1_ < 1.0:
        midy = (y1_ + y2_) / 2.0
        y1_ = max(0.0, midy - 0.5)
        y2_ = min(float(H), midy + 0.5)
    return x1_, y1_, x2_, y2_

def round_box(x1, y1, x2, y2):
    # PIL の crop は int が基本。off-by-one で 0px にならないよう ceil/floor を工夫
    left   = int(math.floor(x1))
    top    = int(math.floor(y1))
    right  = int(math.ceil(x2))
    bottom = int(math.ceil(y2))
    # 念のため right>left, bottom>top を保証
    if right <= left:
        right = left + 1
    if bottom <= top:
        bottom = top + 1
    return left, top, right, bottom

def ensure_dirs(root: Path, class_names: List[str], keep_empty: bool):
    root.mkdir(parents=True, exist_ok=True)
    if keep_empty:
        for cname in class_names:
            (root / cname).mkdir(parents=True, exist_ok=True)

def class_filter_sets(only: str, skip: str, class_names: List[str]) -> Tuple[Set[str], Set[str]]:
    # 入力はID/名が混在可。名を優先し、IDは class_names に解決する
    def parse_list(s):
        return [x.strip() for x in s.split(",") if x.strip()]
    only_set: Set[str] = set()
    skip_set: Set[str] = set()
    for token in parse_list(only):
        if token.isdigit() and int(token) < len(class_names):
            only_set.add(class_names[int(token)])
        else:
            only_set.add(token)
    for token in parse_list(skip):
        if token.isdigit() and int(token) < len(class_names):
            skip_set.add(class_names[int(token)])
        else:
            skip_set.add(token)
    return only_set, skip_set

def main():
    args = parse_args()
    out_root = Path(args.out)
    log_path = Path(args.log) if args.log else None
    logger = open(log_path, "w", encoding="utf-8") if log_path else None
    def log(msg):
        if logger:
            logger.write(msg.rstrip() + "\n")
        # 進捗は静かめに。必要なら print(msg)

    # クラス名
    names = load_names(args.names)
    discovered_max_cls = -1

    datasets: List[Tuple[Path,Path]] = []
    for pair in args.dataset:
        if ":" not in pair:
            raise ValueError(f"--dataset は 'IMAGES_DIR:LABELS_DIR' 形式です: {pair}")
        im_dir, lb_dir = pair.split(":", 1)
        datasets.append((Path(im_dir), Path(lb_dir)))

    # 画像・ラベルのインデックス作成
    items: List[Tuple[Path, Path]] = []
    for img_dir, lb_dir in datasets:
        if not img_dir.exists() or not lb_dir.exists():
            raise FileNotFoundError(f"パス確認: images={img_dir} labels={lb_dir}")
        # ラベル側を基準に、対応する画像を探す（拡張子違いに対応）
        for lb in lb_dir.rglob("*.txt"):
            stem = lb.stem
            found_img = None
            for ext in VALID_EXTS:
                cand = img_dir / f"{stem}{ext}"
                if cand.exists():
                    found_img = cand
                    break
            if not found_img:
                # サブフォルダ構成が違う時も試す（labels と images が同じ相対構造の場合）
                # 例: labels/a/b/xxx.txt -> images/a/b/xxx.jpg
                rel = lb.relative_to(lb_dir).with_suffix("")
                for ext in VALID_EXTS:
                    cand = img_dir / rel.with_suffix(ext)
                    if cand.exists():
                        found_img = cand
                        break
            if not found_img:
                log(f"[MISS_IMG] 対応画像なし: {lb}")
                continue
            items.append((found_img, lb))

    if not items:
        print("処理対象が見つかりませんでした。--dataset の指定を確認してください。")
        if logger: logger.close()
        sys.exit(1)

    # 一旦、全ラベルを走査して最大クラスIDを確認（.names 省略時の上限用）
    if names is None:
        for _, lb in items:
            try:
                for ln in lb.read_text(encoding="utf-8").splitlines():
                    parsed = parse_line(ln)
                    if not parsed: 
                        continue
                    cls, cx, cy, w, h = parsed
                    discovered_max_cls = max(discovered_max_cls, cls)
            except Exception as e:
                log(f"[READ_ERR] {lb} {e}")
        if discovered_max_cls >= 0:
            names = [str(i) for i in range(discovered_max_cls+1)]
        else:
            print("有効なラベルが見つかりませんでした。")
            if logger: logger.close()
            sys.exit(1)

    class_names: List[str] = names
    class_count: Dict[str,int] = {c:0 for c in class_names}
    ensure_dirs(out_root, class_names, keep_empty=args.keep_empty_dirs)

    only_set, skip_set = class_filter_sets(args.only_classes, args.skip_classes, class_names)

    pad_color_cache: Dict[str, Tuple[int,int,int]] = {}

    total_boxes = 0
    saved_boxes  = 0

    for img_path, lb_path in tqdm(items, desc="Cropping"):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            log(f"[OPEN_ERR] {img_path} {e}")
            continue

        W, H = img.size
        try:
            lines = lb_path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            log(f"[READ_ERR] {lb_path} {e}")
            continue

        out_count = 0
        box_idx = 0
        for ln in lines:
            parsed = parse_line(ln)
            if not parsed:
                if ln.strip(): 
                    log(f"[BAD_LINE] {lb_path} :: '{ln}'")
                continue
            cls_id, cx, cy, w, h = parsed
            total_boxes += 1

            if not (0 <= cls_id < len(class_names)):
                log(f"[CLS_OOB] {lb_path} クラスID={cls_id} 範囲外(0..{len(class_names)-1})")
                continue

            cls_name = class_names[cls_id]
            if only_set and cls_name not in only_set: 
                continue
            if skip_set and cls_name in skip_set: 
                continue

            # 正規化 → 座標
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, W, H)

            # 各辺倍率で拡大＋画面端クリップ＋最小1px保証
            x1s, y1s, x2s, y2s = scale_box_sides(
                x1, y1, x2, y2, W, H,
                args.scale_left, args.scale_right, args.scale_top, args.scale_bottom
            )
            left, top, right, bottom = round_box(x1s, y1s, x2s, y2s)

            bw, bh = right - left, bottom - top
            if bw < args.min_side or bh < args.min_side:
                log(f"[TOO_SMALL] {img_path} box#{box_idx} -> {bw}x{bh}px (min {args.min_side})")
                box_idx += 1
                continue

            # クロップ
            try:
                crop = safe_crop(img, (left, top, right, bottom))
            except Exception as e:
                log(f"[CROP_ERR] {img_path} box#{box_idx} {e}")
                box_idx += 1
                continue

            # パディング正方形化
            key = f"{args.pad_bg}:{img_path.suffix.lower()}"
            bg_col = pad_color_cache.get(key)
            if bg_col is None:
                bg_col = color_from_bg(args.pad_bg, crop if args.pad_bg=="mean" else None)
                pad_color_cache[key] = bg_col
            out_img = letterbox_square(crop, args.img_size, bg_col)

            # 保存先
            cls_dir = out_root / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            stem = img_path.stem
            # 画像ごとにボックス連番、衝突回避
            out_name = f"{stem}_b{box_idx}.jpg"
            dst = cls_dir / out_name
            k = 1
            while dst.exists():
                dst = cls_dir / f"{stem}_b{box_idx}_{k}.jpg"
                k += 1

            try:
                out_img.save(dst, quality=95)
            except Exception as e:
                log(f"[SAVE_ERR] {dst} {e}")
                box_idx += 1
                continue

            class_count[cls_name] += 1
            saved_boxes += 1
            out_count += 1
            box_idx += 1

            if out_count >= args.max_per_image:
                log(f"[MAX_PER_IMAGE] {img_path} reached {args.max_per_image}")
                break

        if out_count == 0:
            log(f"[NO_CROP] {img_path} 画像から切り出し0件（ラベル無効/フィルタ/極小/範囲外の可能性）")

    # 要約
    summary = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out": str(out_root),
        "total_label_boxes": total_boxes,
        "saved_crops": saved_boxes,
        "per_class_counts": class_count,
        "scales": {
            "left": args.scale_left, "right": args.scale_right,
            "top": args.scale_top, "bottom": args.scale_bottom
        },
        "img_size": args.img_size,
        "min_side": args.min_side,
        "pad_bg": args.pad_bg
    }
    (out_root / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if logger:
        logger.write(json.dumps(summary, ensure_ascii=False) + "\n")
        logger.close()
    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

