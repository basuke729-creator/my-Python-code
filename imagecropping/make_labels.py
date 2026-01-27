from pathlib import Path

# ====== 確定パス（あなたの環境） ======
IMAGES_DIR = Path("/src/annotation/dataset/images/aisan/1_bottom_part_fps6_3")
LABELS_DIR = Path("/src/annotation/dataset/labels/aisan/1_bottom_part_fps6_3")

# template.txt（固定ボックスのYOLOラベルtxt）
TEMPLATE_LABEL = Path("/src/annotation/dataset/images/aisan/1_bottom_part_490.txt")

# classes.txt と同じ順・同じ名前（番号付きのままでOK）
CLASSES = [
    "3.プレートAスレット有り",
    "4.エンドプレートE",
    "5.金属SP A",
    "6.ガスケットB",
    "7.金属SP B",
    "8.シートA",
    "9.シートB",
    "10.シートC",
    "11.金属SP C",
    "12.コネクタ",
    "13.エンドプレートG",
]

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# =====================================

# テンプレから座標取得（class_idは無視）
line = TEMPLATE_LABEL.read_text(encoding="utf-8").strip()
parts = line.split()
if len(parts) != 5:
    raise ValueError(f"テンプレの形式が不正です（5要素である必要）: {line}")

_, x, y, w, h = parts

name_to_id = {name: i for i, name in enumerate(CLASSES)}

for class_name, class_id in name_to_id.items():
    img_dir = IMAGES_DIR / class_name
    if not img_dir.exists():
        print(f"[SKIP] フォルダなし: {img_dir}")
        continue

    out_dir = LABELS_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    if not imgs:
        print(f"[SKIP] 画像なし: {img_dir}")
        continue

    for img_path in imgs:
        label_path = out_dir / f"{img_path.stem}.txt"
        label_path.write_text(f"{class_id} {x} {y} {w} {h}\n", encoding="utf-8")

    print(f"[OK] {class_name}: {len(imgs)}枚 -> class_id={class_id}")

print("\n=== 完了：全画像に固定ボックス＋クラス別ラベルを自動生成しました ===")
