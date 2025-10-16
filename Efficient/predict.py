# predict.py
import argparse, os, csv
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import timm
from PIL import Image, ImageOps
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VALID_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

# --- リサイズモード実装 ------------------------------------------------------
def letterbox_pad(img: Image.Image, dst_size: int, bg: str = "black") -> Image.Image:
    """縦横比を保持しつつ最長辺を dst_size に、余白を塗って正方形化（推奨）"""
    w, h = img.size
    scale = dst_size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BICUBIC)
    if bg == "white":
        canvas = Image.new("RGB", (dst_size, dst_size), (255, 255, 255))
    elif bg == "mean":
        # 画像の平均色を背景に
        thumb = img.resize((1,1), Image.BILINEAR)
        canvas = Image.new("RGB", (dst_size, dst_size), tuple(thumb.getpixel((0,0))))
    else:
        canvas = Image.new("RGB", (dst_size, dst_size), (0, 0, 0))
    # 中央貼り付け
    x = (dst_size - nw) // 2
    y = (dst_size - nh) // 2
    canvas.paste(img, (x, y))
    return canvas

def build_transform(img_size: int, resize_mode: str, bg: str):
    """推論前処理を構築"""
    if resize_mode == "pad":
        # Letterbox を transforms で包む（LambdaでPIL処理→Tensor化）
        tf = transforms.Compose([
            transforms.Lambda(lambda im: letterbox_pad(im, img_size, bg)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif resize_mode == "crop":
        # 短辺=img_size に合わせてから CenterCrop
        tf = transforms.Compose([
            transforms.Resize(img_size),          # 短辺に合わせる
            transforms.CenterCrop(img_size),     # 中央正方形
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # "stretch"
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return tf

# --- モデル読み込み ----------------------------------------------------------
def load_model(ckpt_path: str, model_name: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt.get("class_names", None)
    if class_names is None:
        # ヘッド形状からクラス数推定（timm の全モデルで classifier 名は一定ではないため ckpt から取得）
        num_classes = ckpt["model"]["classifier.weight"].shape[0]
        class_names = [str(i) for i in range(num_classes)]
    num_classes = len(class_names)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, class_names

# --- 画像列挙 ----------------------------------------------------------------
def iter_images(path: Path):
    if path.is_file():
        if path.suffix.lower() in VALID_EXTS:
            yield path
    else:
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in VALID_EXTS:
                yield p

# --- 推論（TTA対応） ----------------------------------------------------------
def predict_probs(model, img: Image.Image, tf, device, tta: bool = False) -> torch.Tensor:
    """1枚の画像に対して確率ベクトルを返す（TTA=水平反転平均）"""
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    if tta:
        # 水平反転の平均
        img_hf = ImageOps.mirror(img)
        x2 = tf(img_hf).unsqueeze(0).to(device)
        with torch.no_grad():
            logits2 = model(x2)
            probs2 = torch.softmax(logits2, dim=1)
        probs = (probs + probs2) / 2.0
    return probs.squeeze(0).cpu()

def topk_from_probs(probs: torch.Tensor, class_names: List[str], k: int):
    k = min(k, probs.numel())
    conf, idx = torch.topk(probs, k)
    conf = conf.tolist()
    idx = idx.tolist()
    names = [class_names[i] for i in idx]
    return list(zip(names, conf))

# --- メイン -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("Predict with EfficientNet-V2 (enhanced)")
    ap.add_argument("--ckpt", required=True, help="best.ckpt のパス")
    ap.add_argument("--model", required=True, help="学習時と同じ timm モデル名")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--input", required=True, help="画像ファイル or ディレクトリ")
    ap.add_argument("--resize-mode", choices=["pad","crop","stretch"], default="pad",
                    help="pad=レターボックス(推奨) / crop=短辺合わせ+CenterCrop / stretch=強制正方形")
    ap.add_argument("--bg", choices=["black","white","mean"], default="black",
                    help="--resize-mode pad の背景色")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Top-1 確率がこの値未満なら 'unknown' とする (例: 0.6)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tta", action="store_true", help="水平反転 TTA を有効化")
    ap.add_argument("--save-csv", default=None, help="予測結果をCSV保存するパス（例: runs/preds.csv）")
    ap.add_argument("--save-per-class", default=None,
                    help="予測クラスごとに画像をコピーするディレクトリ（例: runs/preds_split）")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tf = build_transform(args.img_size, args.resize-mode if hasattr(args, "resize-mode") else args.resize_mode, args.bg)
    # ↑ argparse の属性名はハイフンがアンダースコアになるので安全に参照
    tf = build_transform(args.img_size, args.resize_mode, args.bg)

    model, class_names = load_model(args.ckpt, args.model, device)

    inp = Path(args.input)
    rows = []
    if args.save_per_class:
        Path(args.save_per_class).mkdir(parents=True, exist_ok=True)

    for img_path in iter_images(inp):
        img = Image.open(img_path).convert("RGB")
        probs = predict_probs(model, img, tf, device, tta=args.tta)
        topk = topk_from_probs(probs, class_names, args.topk)
        top1_name, top1_conf = topk[0]

        # しきい値で unknown
        pred_name = top1_name
        if args.threshold > 0.0 and top1_conf < args.threshold:
            pred_name = "unknown"

        # 表示
        line = ", ".join([f"{n}:{c:.3f}" for n,c in topk])
        print(f"{img_path.name} -> {line}" + (f"  (pred='{pred_name}')" if pred_name != top1_name else ""))

        # CSV 行
        row = {
            "path": str(img_path),
            "pred": pred_name,
            "top1": f"{topk[0][0]}:{topk[0][1]:.6f}",
        }
        for i in range(1, min(3, len(topk))):
            row[f"top{i+1}"] = f"{topk[i][0]}:{topk[i][1]:.6f}"
        rows.append(row)

        # クラスごと仕分け保存
        if args.save_per_class:
            dst_dir = Path(args.save_per_class) / pred_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            # 同名衝突を避ける
            dst_path = dst_dir / img_path.name
            if dst_path.exists():
                stem, suf = dst_path.stem, dst_path.suffix
                i = 1
                while (dst_dir / f"{stem}_{i}{suf}").exists():
                    i += 1
                dst_path = dst_dir / f"{stem}_{i}{suf}"
            # コピーロスを避けたい場合は shutil.copy2 でもOK
            img.save(dst_path)

    # CSV 保存
    if args.save_csv:
        outp = Path(args.save_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fields = ["path","pred","top1","top2","top3"]
        with open(outp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"Saved CSV -> {outp}")

if __name__ == "__main__":
    main()
