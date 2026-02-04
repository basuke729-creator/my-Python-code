#!/usr/bin/env python3
# rotate_augment_dirs_full.py
# - Input: root/(split)/class/(optional subdirs)/images
# - Output: output_root/(split)/class_{angle}/(optional subdirs)/name_{angle}.ext
# - Rotates images by angles (default: 90,180,270)
# - Optional: copy original as angle=0 with --copy-original
# - Handles EXIF orientation
# - Keeps extensions, supports common formats

import argparse
from pathlib import Path
from typing import List, Set
from PIL import Image, ImageOps, UnidentifiedImageError

VALID_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def parse_angles(s: str) -> List[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a = int(part)
        if a % 90 != 0:
            raise ValueError(f"Angle must be multiple of 90: {a}")
        a = a % 360
        if a == 0:
            continue
        out.append(a)
    # preserve order, unique
    uniq = []
    seen = set()
    for a in out:
        if a in seen:
            continue
        seen.add(a)
        uniq.append(a)
    return uniq

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VALID_EXTS

def iter_images(root: Path):
    # yield all image files under root (recursive)
    for p in root.rglob("*"):
        if is_image_file(p):
            yield p

def split_mode_children(input_root: Path, splits: List[str], recursive_splits: bool) -> List[Path]:
    """
    If --splits is specified and found, we treat input_root/split as a unit.
    Otherwise, treat input_root itself as a unit.
    """
    found = []
    for sp in splits:
        d = input_root / sp
        if d.exists() and d.is_dir():
            found.append(d)
    if found:
        return found

    # if recursive_splits, try to find split folders anywhere under root (rarely needed)
    if recursive_splits:
        for sp in splits:
            for d in input_root.rglob(sp):
                if d.is_dir() and d.name == sp:
                    found.append(d)
        if found:
            return found

    return [input_root]

def get_class_dir(img_path: Path, base_unit: Path) -> str:
    """
    Determine class name as the first directory under base_unit.
    base_unit could be:
      - input_root/train  (then first dir is class)
      - input_root        (then first dir could be class or split)
    """
    rel = img_path.relative_to(base_unit)
    parts = rel.parts
    # Need at least: class / file
    if len(parts) < 2:
        return ""
    return parts[0]

def relative_inside_class(img_path: Path, base_unit: Path) -> Path:
    """
    Returns relative path inside the class folder:
      base_unit/class/sub/xxx.jpg -> sub/xxx.jpg
      base_unit/class/xxx.jpg     -> xxx.jpg
    """
    rel = img_path.relative_to(base_unit)
    parts = rel.parts
    if len(parts) < 2:
        return Path(img_path.name)
    return Path(*parts[1:])  # drop class

def safe_open_rgb(img_path: Path) -> Image.Image:
    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im)
        # keep mode mostly, but ensure rotate+save works consistently
        if im.mode in ("RGBA", "LA"):
            # keep alpha if png/webp supports, but for simplicity convert to RGB
            # (必要ならここをRGBA保持に変更できます)
            im = im.convert("RGB")
        elif im.mode == "P":
            im = im.convert("RGB")
        return im.copy()

def save_image(im: Image.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

def main():
    ap = argparse.ArgumentParser(
        description="Rotate augmentation: output class folders with suffix _90/_180/_270 and filenames with _angle."
    )
    ap.add_argument("--input_dir", required=True, help="Input root dir (e.g., dataset/train or dataset)")
    ap.add_argument("--output_dir", required=True, help="Output root dir (e.g., dataset_aug/train or dataset_aug)")
    ap.add_argument("--angles", default="90,180,270", help="Comma-separated angles (multiples of 90). default=90,180,270")
    ap.add_argument("--splits", default="train,val,test", help="Split folder names to detect under input_dir")
    ap.add_argument("--recursive-splits", action="store_true", help="Find split folders anywhere under input_dir (rare)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output files if exist")
    ap.add_argument("--copy-original", action="store_true", help="Also copy original images as angle=0 into class_0")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files, just print planned outputs")
    args = ap.parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)

    if not in_root.exists():
        raise FileNotFoundError(f"input_dir not found: {in_root}")

    angles = parse_angles(args.angles)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    base_units = split_mode_children(in_root, splits, args.recursive_splits)

    total_in = 0
    total_written = 0
    total_skipped = 0
    total_failed = 0

    for base_unit in base_units:
        # Determine output base (keep split name if base_unit is a split folder)
        if base_unit != in_root and base_unit.parent == in_root:
            # input_root/train -> output_root/train
            out_base = out_root / base_unit.name
        else:
            out_base = out_root

        for img_path in iter_images(base_unit):
            class_name = get_class_dir(img_path, base_unit)
            if not class_name:
                continue

            rel_inside = relative_inside_class(img_path, base_unit)  # subdirs/file.ext
            stem = rel_inside.stem
            ext = rel_inside.suffix

            total_in += 1

            # copy original as _0
            if args.copy_original:
                out_class_dir = out_base / f"{class_name}_0" / rel_inside.parent
                out_path = out_class_dir / f"{stem}_0{ext}"
                if out_path.exists() and not args.overwrite:
                    total_skipped += 1
                else:
                    if args.dry_run:
                        print(f"[DRY] {img_path} -> {out_path}")
                        total_written += 1
                    else:
                        try:
                            im = safe_open_rgb(img_path)
                            save_image(im, out_path)
                            total_written += 1
                        except (UnidentifiedImageError, OSError) as e:
                            print(f"[WARN] open failed: {img_path} ({e})")
                            total_failed += 1

            # rotated outputs
            try:
                im0 = safe_open_rgb(img_path)
            except (UnidentifiedImageError, OSError) as e:
                print(f"[WARN] open failed: {img_path} ({e})")
                total_failed += 1
                continue

            for ang in angles:
                out_class_dir = out_base / f"{class_name}_{ang}" / rel_inside.parent
                out_path = out_class_dir / f"{stem}_{ang}{ext}"

                if out_path.exists() and not args.overwrite:
                    total_skipped += 1
                    continue

                if args.dry_run:
                    print(f"[DRY] {img_path} -> {out_path}")
                    total_written += 1
                    continue

                try:
                    rotated = im0.rotate(ang, expand=True)
                    save_image(rotated, out_path)
                    total_written += 1
                except Exception as e:
                    print(f"[WARN] save failed: {out_path} ({e})")
                    total_failed += 1

    print("\n=== Summary ===")
    print(f"input_images_seen : {total_in}")
    print(f"written_files     : {total_written}")
    print(f"skipped_existing  : {total_skipped}")
    print(f"failed            : {total_failed}")
    print(f"output_dir        : {out_root}")
    print("[OK] done")

if __name__ == "__main__":
    main()
