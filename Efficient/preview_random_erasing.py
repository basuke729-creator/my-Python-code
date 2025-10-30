# preview_random_erasing.py
# 指定フォルダ内の画像に RandomErasing を適用して出力
# (train_effnetv2.py と同じ正規化・イレース設定)

import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import torchvision.utils as vutils

# === 定数 ===
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# === Augment定義（train_effnetv2.pyと同じ構造） ===
def build_random_erasing_transform(img_size: int = 384, p: float = 0.25):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(
            p=p,                # 発動確率
            scale=(0.02, 0.2),  # 消す領域の相対面積
            ratio=(0.3, 3.3),   # アスペクト比の範囲
            value="random"      # 塗りつぶし色をランダムに
        )
    ])

# === テンソルを画像に戻すための逆変換 ===
def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    inv = transforms.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD]
    )(tensor)
    inv = inv.clamp(0, 1)
    img = transforms.ToPILImage()(inv)
    return img

# === メイン処理 ===
def main(input_dir: str, output_dir: str, img_size: int = 384, p: float = 0.25, repeats: int = 3):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tf = build_random_erasing_transform(img_size, p)

    img_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    print(f"[INFO] {len(img_paths)} images found in {input_dir}")

    for img_path in img_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            print(f"[WARN] cannot open: {img_path}")
            continue

        for i in range(repeats):
            out = tf(img)
            out_img = tensor_to_image(out)
            out_name = f"{img_path.stem}_erased{i+1}{img_path.suffix}"
            out_img.save(output_dir / out_name)

    print(f"[DONE] All processed images saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("RandomErasing Preview Tool")
    ap.add_argument("--input", required=True, help="入力フォルダ（画像が入っている）")
    ap.add_argument("--output", default="./erasing_preview", help="出力フォルダ")
    ap.add_argument("--img-size", type=int, default=384, help="リサイズ後の画像サイズ")
    ap.add_argument("--p", type=float, default=0.25, help="ランダムイレース発動確率 (0〜1)")
    ap.add_argument("--repeats", type=int, default=3, help="各画像に対して何回試すか")
    args = ap.parse_args()

    main(args.input, args.output, args.img_size, args.p, args.repeats)
