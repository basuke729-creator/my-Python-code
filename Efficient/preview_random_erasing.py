# preview_random_erasing.py
# ランダムイレースの「マスク面積」を調整して確認できるツール

import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def build_random_erasing_transform(img_size: int, p: float, scale_min: float, scale_max: float):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(
            p=p,
            scale=(scale_min, scale_max),  # 👈 隠す面積をここで調整
            ratio=(0.3, 3.3),
            value="random"
        )
    ])

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    inv = transforms.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD]
    )(tensor)
    inv = inv.clamp(0, 1)
    return transforms.ToPILImage()(inv)

def main(input_dir, output_dir, img_size, p, scale_min, scale_max, repeats):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tf = build_random_erasing_transform(img_size, p, scale_min, scale_max)
    img_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    print(f"[INFO] {len(img_paths)} images found in {input_dir}")

    for img_path in img_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] cannot open {img_path}: {e}")
            continue

        for i in range(repeats):
            out = tf(img)
            out_img = tensor_to_image(out)
            out_name = f"{img_path.stem}_erased{i+1}_scale{scale_min}-{scale_max}{img_path.suffix}"
            out_img.save(output_dir / out_name)

    print(f"[DONE] Saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("RandomErasing Preview Tool (Adjustable scale)")
    ap.add_argument("--input", required=True, help="入力フォルダ")
    ap.add_argument("--output", default="./erasing_preview", help="出力フォルダ")
    ap.add_argument("--img-size", type=int, default=384, help="リサイズ後のサイズ")
    ap.add_argument("--p", type=float, default=1.0, help="ランダムイレース発動確率 (0〜1)")
    ap.add_argument("--scale-min", type=float, default=0.02, help="マスク面積の最小値 (相対値)")
    ap.add_argument("--scale-max", type=float, default=0.2, help="マスク面積の最大値 (相対値)")
    ap.add_argument("--repeats", type=int, default=3, help="各画像の試行回数")
    args = ap.parse_args()

    main(args.input, args.output, args.img_size, args.p, args.scale_min, args.scale_max, args.repeats)
