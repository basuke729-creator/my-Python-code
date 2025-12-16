# predict.py  (inference + evaluation + speed)
# v1.7.1 (fixed: resize_mode bug, keep all features)
import argparse, csv, os, shutil, json, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import timm
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

print("predict.py v1.7.1 (speed + per-class summary + eval + CSV)")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VALID_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

# ------------------------ ★追加：混同行列などで表示したいクラス順（固定） ------------------------
# ユーザー指定順。重複している「脚立上の安全な人」は順序を保って1回にまとめて扱います。
PREFERRED_LABEL_ORDER = [
    "脚立",
    "立馬",
    "その他の人",
    "脚立上の安全な人",
    "脚立1段目の人",
    "脚立2段目の人",
    "脚立を跨ぐ人",
    "脚立上の不安定姿勢の人",
    "立馬上の安全な人",
    "立馬上の不安定姿勢な人",
]

def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def apply_preferred_label_order(labels_current: List[str], include_unknown: bool) -> List[str]:
    """
    labels_current: build_label_space(...) が返す、実際に存在するラベル集合
    返り値: 指定順を優先しつつ、未指定ラベルは末尾へ追加。unknown は必要なら末尾へ。
    """
    preferred = _unique_preserve_order(PREFERRED_LABEL_ORDER)

    # まず指定順のうち、実際に存在するものだけ採用（順序は固定）
    ordered = [l for l in preferred if l in labels_current]

    # 指定順に無いが存在するラベルは、従来通り後ろに足す（機能維持の安全策）
    for l in labels_current:
        if l not in ordered:
            ordered.append(l)

    # include_unknown が有効で、unknown が必要なら最後に置く（指定順に無い場合）
    if include_unknown and "unknown" in labels_current and "unknown" not in ordered:
        ordered.append("unknown")

    return ordered

# ------------------------ 日本語フォント ------------------------
def set_japanese_font(font_path: str = ""):
    try:
        if font_path and Path(font_path).exists():
            fp = font_manager.FontProperties(fname=font_path)
            matplotlib.rcParams["font.family"] = fp.get_name()
            return
        candidates = [
            "Noto Sans CJK JP","Noto Sans JP","Noto Sans CJK JP Regular",
            "IPAexGothic","IPAGothic","TakaoPGothic","VL PGothic"
        ]
        avail = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in avail:
                matplotlib.rcParams["font.family"] = name
                return
    except Exception:
        pass

# ------------------------ リサイズ ------------------------
def letterbox_pad(img: Image.Image, dst_size: int, bg: str = "black") -> Image.Image:
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (dst_size, dst_size), (0,0,0))
    scale = dst_size / max(w, h)
    nw, nh = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
    img_resized = img.resize((nw, nh), Image.BICUBIC)

    if bg == "white":
        canvas = Image.new("RGB", (dst_size, dst_size), (255,255,255))
    elif bg == "mean":
        thumb = img.resize((1,1), Image.BILINEAR)
        canvas = Image.new("RGB", (dst_size, dst_size), tuple(thumb.getpixel((0,0))))
    else:
        canvas = Image.new("RGB", (dst_size, dst_size), (0,0,0))

    x = (dst_size - nw)//2
    y = (dst_size - nh)//2
    canvas.paste(img_resized, (x, y))
    return canvas

def build_transform(img_size: int, resize_mode: str, bg: str):
    if resize_mode == "pad":
        return transforms.Compose([
            transforms.Lambda(lambda im: letterbox_pad(im, img_size, bg)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif resize_mode == "crop":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # stretch
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

# ------------------------ モデル読み込み ------------------------
def _infer_num_classes_from_state_dict(state_dict: dict) -> int:
    candidates = []
    for k, v in state_dict.items():
        if not (k.endswith("weight") and getattr(v, "ndim", 0) == 2):
            continue
        bias_key = k.replace("weight", "bias")
        if bias_key in state_dict and getattr(state_dict[bias_key], "ndim", 0) == 1:
            candidates.append((k, v.shape[0]))
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

# ------------------------ 画像列挙 ------------------------
def iter_images(path: Path):
    if path.is_file():
        if path.suffix.lower() in VALID_EXTS:
            yield path
    else:
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                yield p

# ------------------------ 小ユーティリティ ------------------------
def topk_from_probs(probs: torch.Tensor, class_names: List[str], k: int):
    k = min(k, probs.numel())
    conf, idx = torch.topk(probs, k)
    conf = conf.tolist()
    idx = idx.tolist()
    names = [class_names[i] for i in idx]
    return list(zip(names, conf))

def infer_true_label(img_path: Path, input_root: Path) -> Optional[str]:
    try:
        rel = img_path.resolve().relative_to(input_root.resolve())
    except Exception:
        return None
    parts = rel.parts
    return parts[0] if len(parts) >= 2 else None

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
        w.writerow(["class","precision","recall","f1","support"])
        for c in labels:
            d = report.get(c, {})
            w.writerow([c, d.get("precision",0), d.get("recall",0), d.get("f1-score",0), int(d.get("support",0))])
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

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_png: Path,
                          normalize: bool = False, cmap: str = "Blues"):
    m = cm.astype(np.float32)
    ann = m.copy()
    title = "Confusion Matrix"
    if normalize:
        row_sums = m.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        m = m / row_sums
        ann = m * 100.0
        title += " (normalized, %) "

    fig_w = max(6, len(labels) * 0.7)
    fig_h = max(5, len(labels) * 0.6)
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(m, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_idx = np.arange(len(labels))
    plt.xticks(tick_idx, labels, rotation=60, ha="right")
    plt.yticks(tick_idx, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    thresh = m.max() / 2.0 if m.size > 0 else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{ann[i, j]:.1f}" if normalize else f"( {int(ann[i, j])} )"
            color = "white" if m[i, j] > thresh else "black"
            plt.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

# ------------------------ メイン ------------------------
def main():
    ap = argparse.ArgumentParser("Predict with EfficientNet-V2 (inference + evaluation + speed)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--input", required=True)
    ap.add_argument("--resize-mode", choices=["pad","crop","stretch"], default="pad")
    ap.add_argument("--bg", choices=["black","white","mean"], default="black")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tta", action="store_true")
    # 出力
    ap.add_argument("--save-csv", default=None)
    ap.add_argument("--save-per-class", default=None)
    # 評価
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-out", default=None)
    ap.add_argument("--include-unknown", action="store_true")
    ap.add_argument("--export-miscls", action="store_true")
    ap.add_argument("--miscls-limit", type=int, default=50)
    ap.add_argument("--font", default="")
    # 1枚ごとのログ表示
    ap.add_argument("--per-image", action="store_true")
    # 速度計測
    ap.add_argument("--speed", action="store_true", help="推論速度を計測して要約を表示")
    ap.add_argument("--speed-csv", default=None, help="速度を per-image でCSV保存するパス")
    ap.add_argument("--warmup", type=int, default=5, help="最初のN枚は計測から除外（GPU起動やキャッシュの安定化）")
    ap.add_argument("--no-sync", action="store_true", help="CUDA同期を行わない（粗い計測）。精密計測は未指定が推奨。")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tf = build_transform(args.img_size, args.resize_mode, args.bg)
    model, class_names = load_model(args.ckpt, args.model, device)

    inp_root = Path(args.input)
    rows = []
    if args.save_per_class:
        Path(args.save_per_class).mkdir(parents=True, exist_ok=True)

    y_true, y_pred, paths_for_eval = [], [], []
    total_seen = 0

    # 速度計測のバッファ
    prep_ms_list, infer_ms_list, total_ms_list = [], [], []
    if args.speed and args.speed_csv:
        Path(args.speed_csv).parent.mkdir(parents=True, exist_ok=True)
        sp_f = open(args.speed_csv, "w", newline="", encoding="utf-8")
        sp_w = csv.writer(sp_f)
        sp_w.writerow(["path", "prep_ms", "infer_ms", "total_ms"])
    else:
        sp_f = None
        sp_w = None

    for img_path in iter_images(inp_root):
        # 画像読み込み
        try:
            with open(img_path, "rb") as fh:
                img = Image.open(fh).convert("RGB")
        except (UnidentifiedImageError, OSError):
            if args.per_image:
                print(f"[WARN] open failed: {img_path}")
            continue
        except Exception:
            if args.per_image:
                print(f"[WARN] open failed: {img_path}")
            continue

        # ---- 計測: 前処理 + 推論 ----
        t0 = time.perf_counter()
        x = tf(img).unsqueeze(0).to(device)
        t1 = time.perf_counter()

        with torch.no_grad():
            logits = model(x)
        if device.type == "cuda" and not args.no_sync:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        if args.tta:
            img_hf = ImageOps.mirror(img)
            x2 = tf(img_hf).unsqueeze(0).to(device)
            with torch.no_grad():
                logits2 = model(x2)
            if device.type == "cuda" and not args.no_sync:
                torch.cuda.synchronize()
            probs = (probs + torch.softmax(logits2, dim=1).squeeze(0).cpu()) / 2.0
            t2 = time.perf_counter()

        prep_ms  = (t1 - t0) * 1000.0
        infer_ms = (t2 - t1) * 1000.0
        total_ms = (t2 - t0) * 1000.0

        # ウォームアップ分は統計に含めない
        if args.speed and total_seen >= args.warmup:
            prep_ms_list.append(prep_ms)
            infer_ms_list.append(infer_ms)
            total_ms_list.append(total_ms)
            if sp_w:
                sp_w.writerow([str(img_path), f"{prep_ms:.3f}", f"{infer_ms:.3f}", f"{total_ms:.3f}"])

        # 予測
        k = min(args.topk, probs.numel())
        conf, idx = torch.topk(probs, k)
        conf = conf.tolist(); idx = idx.tolist()
        names = [class_names[i] for i in idx]
        topk = list(zip(names, conf))
        top1_name, top1_conf = topk[0]
        pred_name = top1_name if (args.threshold == 0.0 or top1_conf >= args.threshold) else "unknown"

        if args.per_image:
            line = ", ".join([f"{n}:{c:.3f}" for n, c in topk])
            suffix = f"  (pred='{pred_name}')" if pred_name != top1_name else ""
            print(f"{img_path.name} -> {line}{suffix}")

        rows.append({
            "path": str(img_path),
            "pred": pred_name,
            "top1": f"{topk[0][0]}:{topk[0][1]:.6f}",
            "top2": f"{topk[1][0]}:{topk[1][1]:.6f}" if len(topk) > 1 else "",
            "top3": f"{topk[2][0]}:{topk[2][1]:.6f}" if len(topk) > 2 else "",
        })

        if args.save_per_class:
            dst_dir = Path(args.save_per_class) / pred_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / img_path.name
            kdup = 1
            while dst_path.exists():
                dst_path = dst_dir / f"{img_path.stem}_{kdup}{img_path.suffix}"
                kdup += 1
            try:
                shutil.copy2(img_path, dst_path)
            except Exception:
                img.save(dst_path)

        if args.eval:
            gt = infer_true_label(img_path, inp_root)
            if gt is not None:
                y_true.append(gt)
                y_pred.append(pred_name)
                paths_for_eval.append(str(img_path))

        total_seen += 1

    if sp_f:
        sp_f.close()

    # 予測CSV
    if args.save_csv:
        outp = Path(args.save_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path","pred","top1","top2","top3"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Saved CSV -> {outp}")

    # 速度要約
    if args.speed and len(total_ms_list) > 0:
        def stats(x):
            x = np.array(x, dtype=np.float64)
            return dict(
                mean=float(x.mean()),
                p50=float(np.percentile(x, 50)),
                p90=float(np.percentile(x, 90)),
                p95=float(np.percentile(x, 95)),
                p99=float(np.percentile(x, 99)),
            )
        s_prep = stats(prep_ms_list)
        s_infer = stats(infer_ms_list)
        s_total = stats(total_ms_list)
        imgs = len(total_ms_list)
        sec = sum(total_ms_list) / 1000.0
        ips = imgs / sec if sec > 0 else 0.0

        print("\n=== Inference Speed (warmup除外) ===")
        print(f"samples: {imgs}  total_time: {sec:.3f}s  throughput: {ips:.2f} img/s")
        print(f"preprocess ms -> mean {s_prep['mean']:.2f} | p50 {s_prep['p50']:.2f} | p95 {s_prep['p95']:.2f}")
        print(f"inference ms -> mean {s_infer['mean']:.2f} | p50 {s_infer['p50']:.2f} | p95 {s_infer['p95']:.2f}")
        print(f"total ms     -> mean {s_total['mean']:.2f} | p50 {s_total['p50']:.2f} | p95 {s_total['p95']:.2f}")

    # 評価（混同行列など）
    if args.eval:
        if not y_true:
            print("[INFO] 評価対象が見つかりませんでした。--input が '親/クラス/画像' 構造か確認してください。")
            return
        set_japanese_font(args.font)

        # ------------------------ ★ここだけ変更：labels を指定順で固定 ------------------------
        labels_current = build_label_space(y_true, y_pred, class_names, include_unknown=args.include_unknown)
        labels = apply_preferred_label_order(labels_current, include_unknown=args.include_unknown)
        # -------------------------------------------------------------------------------

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        report["_overall"] = {"accuracy": acc, "micro_f1": micro_f1, "macro_f1": macro_f1, "num_samples": len(y_true)}

        eval_dir = Path(args.eval_out or "runs/eval_from_predict")
        eval_dir.mkdir(parents=True, exist_ok=True)
        save_json(report, eval_dir / "report.json")
        save_report_csv(report, labels, eval_dir / "report.csv")
        save_confusion_matrix_csv(cm, labels, eval_dir / "confusion_matrix.csv")
        plot_confusion_matrix(cm, labels, eval_dir / "confusion_matrix.png", normalize=False)
        plot_confusion_matrix(cm, labels, "confusion_matrix_norm.png" if isinstance(eval_dir, str) else eval_dir / "confusion_matrix_norm.png", normalize=True)

        totals = cm.sum(axis=1)
        corrects = np.diag(cm)
        per_class_acc = np.divide(corrects, np.maximum(totals, 1), out=np.zeros_like(corrects, dtype=float), where=totals>0)

        print("\n=== Per-class Accuracy (真値ベース) ===")
        width = max(len(s) for s in labels) if labels else 10
        print(f"{'class'.ljust(width)}  {'acc':>6}  {'correct':>7} / {'total':<7}")
        for i, name in enumerate(labels):
            print(f"{name.ljust(width)}  {per_class_acc[i]*100:6.2f}%  {int(corrects[i]):7d} / {int(totals[i]):<7d}")
        print(f"\nOverall accuracy: {acc*100:.2f}%   (samples: {len(y_true)})")
        print(f"Macro-F1: {macro_f1:.4f}  Micro-F1: {micro_f1:.4f}")

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
    else:
        print(f"[DONE] processed images: {total_seen}")

if __name__ == "__main__":
    main()

    


