from pathlib import Path

# ====== 設定 ======
BASE_DIR = Path("dataset")

IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"

TEMPLATE_LABEL = Path("template.txt")   # 1枚だけ作った固定ボックス

# classes.txt と同じ順・同じ名前（番号付きのままでOK）
CLASSES = [
    "3.プレートAスレット有り",
    "4.エンドプレートE",
    "5.金属sp A",
    "6.ガスケットB",
    "7.金属sp B",
    "8.シートA",
    "9.シートB",
    "10.シートC",
    "11.金属sp C",
    "12.コネクタ",
    "13.エンドプレートC",
]

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ===================

# テンプレから座標取得
line = TEMPLATE_LABEL.read_text(encoding="utf-8").strip()
parts = line.split()
if len(parts) != 5:
    raise ValueError(f"template.txtの形式が不正です: {line}")

_, x, y, w, h = parts

# クラス名 -> class_id（0始まり）
name_to_id = {name: i for i, name in enumerate(CLASSES)}

# 各クラスフォルダを走査
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
