# predict.py  (inference + evaluation, robust)
import argparse, csv, os, shutil, json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import timm
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ---------- 画像サイズ変更 ----------
def letterbox_pad(img: Image.Image, dst_size: int, bg: str = "black") -> Image.Image:
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (dst_size, dst_size), (0, 0, 0))
    scale = dst_size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img_resized = img.resize((nw, nh), Image.BICUBIC)

    if bg == "white":
        canvas = Image.new("RGB", (dst_size, dst_size), (255, 255, 255))
    elif bg == "mean":
        thumb = img.resize((1, 1), Image.BILINEAR)
        canvas = Image.new("RGB", (dst_size, dst_size), tuple(thumb.getpixel((0, 0))))
    else:
        canvas = Image.new("RGB", (dst_size, dst_size), (0, 0, 0))

    x = (dst_size - nw) // 2
    y = (dst_size - nh) // 2
    canvas.paste(img_resized, (x, y))
    return canvas

def build_transform(img_size: int, resize_mode: str, bg: str):
    if resize_mode == "pad":
        tf = transforms.Compose([
            transforms.Lambda(lambda im: letterbox_pad(im, img_size, bg)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif resize_mode == "crop":
        tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # stretch
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return tf

# ---------- モデル読み込み ----------
def _infer_num_classes_from_state_dict(state_dict: dict) -> int:
    candidates = []
    for k, v in state_dict.items():
        if not (k.endswith("weight") and getattr(v, "ndim", 0) == 2):
            continue
        bias_key = k.replace("weight", "bias")
        if bias_key in state_dict and getattr(state_dict[bias_key], "ndim", 0) == 1:
            out_dim = v.shape[0]
            candidates.append((k, out_dim))
    if not candidates:
        lin_dims = [v.shape[0] for k, v in state_dict.items() if k.endswith("weight") and getattr(v, "ndim", 0) == 2]
        if lin_dims:
            return int(min(lin_dims))
        raise RuntimeError("Could not infer num_classes from state_dict.")
    return int(candidates[-1][1])

def load_model(ckpt_path: str, model_name: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    class_names = ckpt.get("class_names", None)
    if class_names is not None:
        num_classes = len(class_names)
    else:
        num_classes = _infer_num_classes_from_state_dict(state)
        class_names = [str(i) for i in range(num_classes)]

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, class_names

# ---------- ユーティリティ ----------
def iter_images(path: Path):
    if path.is_file():
        if path.suffix.lower() in VALID_EXTS:
            yield path
    else:
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                yield p

def predict_probs(model, img: Image.Image, tf, device, tta: bool = False) -> torch.Tensor:
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    if tta:
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

def infer_true_label(img_path: Path, input_root: Path) -> Optional[str]:
    """
    input_root/クラス名/画像 のとき、そのクラス名を返す。
    深い階層でも最上位のサブフォルダ名を真値とみなす（日本語名OK）。
    """
    try:
        rel = img_path.resolve().relative_to(input_root.resolve())
    except Exception:
        return None
    parts = rel.parts
    if len(parts) >= 2:   # 少なくとも class/filename
        return parts[0]
    return None

# ---------- 評価補助 ----------
def build_label_space(y_true: List[str], y_pred: List[str], class_names: List[str], include_unknown: bool):
    labels = list(class_names)
    extra = sorted({*(y_true or []), *(y_pred or [])} - set(labels))
    labels += [l for l in extra if (include_unknown or l != "unknown")]
    if include_unknown and "unknown" not in labels:
        labels.append("unknown")
    return labels

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def save_report_csv(report: Dict, labels: List[str], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for c in labels:
            d = report.get(c, {})
            w.writerow([c, d.get("precision", 0), d.get("recall",0), d.get("f1-score",0), int(d.get("support",0))])
        ov = report.get("_overall", {})
        w.writerow([])
        w.writerow(["_overall", ov.get("accuracy",0), ov.get("micro_f1",0), ov.get("macro_f1",0), ov.get("num_samples",0)])

def save_confusion_matrix_csv(cm: np.ndarray, labels: List[str], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for i, r in enumerate(cm):
            w.writerow([labels[i]] + list(r))

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_png: Path, normalize: bool = False):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    m = cm.astype(np.float32)
    if normalize:
        row_sums = m.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        m = m / row_sums
    plt.figure(figsize=(max(6, len(labels)*0.6), max(5, len(labels)*0.5)))
    plt.imshow(m, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=60, ha="right")
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_bars(report: Dict, labels: List[str], metric: str, out_png: Path):
    vals = [report.get(c, {}).get(metric, 0.0) for c in labels]
    plt.figure(figsize=(max(6, len(labels)*0.6), 4.5))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    plt.ylabel(metric)
    plt.title(f"Per-class {metric}")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()

# ---------- メイン ----------
def main():
    ap = argparse.ArgumentParser("Predict with EfficientNet-V2 (inference + evaluation)")
    ap.add_argument("--ckpt", required=True, help="best.ckpt のパス")
    ap.add_argument("--model", required=True, help="学習時と同じ timm モデル名（例: tf_efficientnetv2_s_in21k_ft_in1k）")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--input", required=True, help="画像ファイル or ディレクトリ（親/クラス/画像 なら評価可）")
    ap.add_argument("--resize-mode", choices=["pad", "crop", "stretch"], default="pad")
    ap.add_argument("--bg", choices=["black", "white", "mean"], default="black")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.0, help="Top-1 確率が未満なら 'unknown'")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tta", action="store_true", help="水平反転 TTA")
    # 保存
    ap.add_argument("--save-csv", default=None, help="予測をCSV保存（例: runs/preds.csv）")
    ap.add_argument("--save-per-class", default=None, help="予測クラスごとに画像コピーする先")
    # 評価
    ap.add_argument("--eval", action="store_true", help="親/クラス/画像 構造として評価を保存")
    ap.add_argument("--eval-out", default=None, help="評価レポート出力先（例: runs/eval_from_predict）")
    ap.add_argument("--include-unknown", action="store_true", help="unknown を評価に含める")
    ap.add_argument("--export-miscls", action="store_true", help="誤分類を True/Pred 別にコピー")
    ap.add_argument("--miscls-limit", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tf = build_transform(args.img_size, args.resize_mode, args.bg)
    model, class_names = load_model(args.ckpt, args.model, device)

    inp_root = Path(args.input)
    rows = []
    if args.save_per_class:
        Path(args.save_per_class).mkdir(parents=True, exist_ok=True)

    y_true, y_pred, paths_for_eval = [], [], []

    for img_path in iter_images(inp_root):
        # --- 画像読込：ここでの例外を丁寧に処理（以前のバグ修正） ---
        try:
            img = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"[WARN] open failed (not an image or corrupted): {img_path} ({e})")
            continue
        except Exception as e:
            print(f"[WARN] open failed: {img_path} ({e})")
            continue

        probs = predict_probs(model, img, tf, device, tta=args.tta)
        topk = topk_from_probs(probs, class_names, args.topk)
        top1_name, top1_conf = topk[0]
        pred_name = top1_name
        if args.threshold > 0.0 and top1_conf < args.threshold:
            pred_name = "unknown"

        # 表示
        line = ", ".join([f"{n}:{c:.3f}" for n, c in topk])
        suffix = f"  (pred='{pred_name}')" if pred_name != top1_name else ""
        print(f"{img_path.name} -> {line}{suffix}")

        # CSV 行
        row = {"path": str(img_path), "pred": pred_name, "top1": f"{topk[0][0]}:{topk[0][1]:.6f}"}
        for i in range(1, min(3, len(topk))):
            row[f"top{i+1}"] = f"{topk[i][0]}:{topk[i][1]:.6f}"
        rows.append(row)

        # 予測クラスごとに保存（任意）
        if args.save_per_class:
            dst_dir = Path(args.save_per_class) / pred_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / img_path.name
            k = 1
            while dst_path.exists():
                dst_path = dst_dir / f"{img_path.stem}_{k}{img_path.suffix}"
                k += 1
            try:
                shutil.copy2(img_path, dst_path)
            except Exception:
                img.save(dst_path)

        # 真値（親/クラス/画像 構造なら）
        if args.eval:
            gt = infer_true_label(img_path, inp_root)
            if gt is not None:
                y_true.append(gt)
                y_pred.append(pred_name)
                paths_for_eval.append(str(img_path))

    # CSV 保存
    if args.save_csv:
        outp = Path(args.save_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fields = ["path", "pred", "top1", "top2", "top3"]
        with open(outp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"Saved CSV -> {outp}")

    # ---------- 評価 ----------
    if args.eval:
        if not y_true:
            print("[INFO] 評価対象が見つかりませんでした。--input が '親/クラス/画像' 構造であるか確認してください。")
            return

        labels = build_label_space(y_true, y_pred, class_names, include_unknown=args.include_unknown)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        report["_overall"] = {"accuracy": acc, "micro_f1": micro_f1, "macro_f1": macro_f1, "num_samples": len(y_true)}

        eval_dir = Path(args.eval_out or "runs/eval_from_predict")
        eval_dir.mkdir(parents=True, exist_ok=True)

        # 保存
        def _save_json(o, p): p.write_text(json.dumps(o, indent=2, ensure_ascii=False), encoding="utf-8")
        _save_json(report, eval_dir / "report.json")
        save_report_csv(report, labels, eval_dir / "report.csv")
        save_confusion_matrix_csv(cm, labels, eval_dir / "confusion_matrix.csv")
        plot_confusion_matrix(cm, labels, eval_dir / "confusion_matrix.png", normalize=False)
        plot_confusion_matrix(cm, labels, eval_dir / "confusion_matrix_norm.png", normalize=True)
        plot_bars(report, labels, "precision", eval_dir / "precision_per_class.png")
        plot_bars(report, labels, "recall",    eval_dir / "recall_per_class.png")
        plot_bars(report, labels, "f1-score",  eval_dir / "f1_per_class.png")
        plot_bars(report, labels, "support",   eval_dir / "support_per_class.png")

        # 誤分類コピー（任意）
        if args.export_miscls:
            out_dir = eval_dir / "misclassified"
            out_dir.mkdir(parents=True, exist_ok=True)
            counter: Dict[Tuple[str,str], int] = {}
            for p, t, pr in zip(paths_for_eval, y_true, y_pred):
                if t == pr: 
                    continue
                key = (t, pr)
                counter[key] = counter.get(key, 0) + 1
                if counter[key] > args.miscls_limit: 
                    continue
                try:
                    im = Image.open(p).convert("RGB")
                    pair_dir = out_dir / f"true_{t}__pred_{pr}"
                    pair_dir.mkdir(parents=True, exist_ok=True)
                    dst = pair_dir / Path(p).name
                    i = 1
                    while dst.exists():
                        dst = pair_dir / f"{Path(p).stem}_{i}{Path(p).suffix}"
                        i += 1
                    im.save(dst)
                except Exception:
                    pass

        print(f"[OK] Evaluation saved to: {eval_dir}")

if __name__ == "__main__":
    main()

