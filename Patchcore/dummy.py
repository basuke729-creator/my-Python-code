from pathlib import Path
from PIL import Image
import os

# 異常画像フォルダとマスク出力フォルダ
img_dir = Path("test/broken")           # 危険姿勢
mask_dir = Path("ground_truth/broken")  # マスク出力先
mask_dir.mkdir(parents=True, exist_ok=True)

# broken 内の画像をソートして取得
img_files = sorted(
    [p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
)

if not img_files:
    print("test/broken に画像が見つかりませんでした。")
    exit(1)

print(f"{len(img_files)} 枚の異常画像に対してダミーマスクを作成します。")

for img_path in img_files:
    # 画像サイズを取得（384x384のはずだが一応読み込み）
    with Image.open(img_path) as im:
        w, h = im.size

    # 真っ黒のマスク画像を作成（8bitグレースケール）
    mask = Image.new("L", (w, h), 0)

    # MVTec っぽくファイル名を *_mask.png にする
    mask_name = img_path.stem + "_mask.png"
    mask_path = mask_dir / mask_name
    mask.save(mask_path)

    print(f"  -> {mask_path}")

print("ダミーマスク作成完了！")
