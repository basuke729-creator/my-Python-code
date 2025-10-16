# evaluate_predictions.py
import argparse, csv, json, os, shutil
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ------------------------------------------------------------
# 前提:
#  - predict.py で --save-csv したファイルを入力（列: path, pred, top1, top2, top3）
#  - 画像パスは test_crops/<GT_CLASS>/xxx.jpg のように「親フォルダ名 = 正解ラベル」
# 出力:
#  - output_dir に JSON / TXT / CSV / PNG（グラフ） を保存
# ------------------------------------------------------------

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def read_preds_csv(csv_path: Path) -> List[Dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def infer_true_label(p: Path) -> str:
    return p.parent.name

def build_label_space(rows: List[Dict], include_unknown: bool) -> List[str]:
    y_true_labels = set()
    y_pred_labels = set()
    for r in rows:
        p = Path(r["path"])
        if p.suffix.lower() not in VALID_EXTS:
            continue
        y_true_labels.add(infer_true_label(p))
        y_pred_labels.add(r.get("pred", ""))
    labels = sorted(y_true_labels.union(y_pred_labels))
    if not include_unknown and "unknown" in labels:
        labels.remove("unknown")
    # unknown を含める場合は最後尾に
    if include_unknown and "unknown" in labels:
        labels = [x for x in labels if x != "unknown"] + ["unknown"]
    return labels

def filter_rows(rows: List[Dict], include_unknown: bool) -> List[Dict]:
    if include_unknown:
        return rows
    # unknown 予測は評価対象から外す（任意）
    return [r for r in rows if r.get("pred", "") != "unknown"]

def compute_metrics(rows: List[Dict], labels: List[str]):
    y_true, y_pred, used = [], [], []
    for r in rows:
        p = Path(r["path"])
        if p.suffix.lower() not in VALID_EXTS:
            continue
        t = infer_true_label(p)
        pr = r.get("pred", "")
        if t in labels and pr in labels:
            y_true.append(t)
            y_pred.append(pr)
            used.append(r)
    if not y_true:
        raise RuntimeError("評価対象がありません。labels / rows を見直してください。")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    report["_overall"] = {"accuracy": acc, "micro_f1": micro_f1, "macro_f1": macro_f1, "num_samples": len(y_true)}
    return report, cm, labels, y_true, y_pred

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def save_txt_summary(report: Dict, labels: List[str], out_txt: Path):
    lines = []
    ov = report.get("_overall", {})
    lines.append(f"Accuracy : {ov.get('accuracy', 0):.4f}")
    lines.append(f"Micro F1 : {ov.get('micro_f1', 0):.4f}")
    lines.append(f"Macro F1 : {ov.get('macro_f1', 0):.4f}")
    lines.append("")
    lines.append("Per-class metrics:")
    lines.append(f"{'class':30s}  {'precision':>9s}  {'recall':>9s}  {'f1':>9s}  {'support':>7s}")
    for c in labels:
        d = report.get(c, {})
        lines.append(f"{c:30.30s}  {d.get('precision',0):9.4f}  {d.get('recall',0):9.4f}  {d.get('f1-score',0):9.4f}  {int(d.get('support',0)):7d}")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines))

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
    vals = []
    for c in labels:
        v = report.get(c, {}).get(metric, 0.0)
        vals.append(v)
    plt.figure(figsize=(max(6, len(labels)*0.6), 4.5))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    plt.ylabel(metric)
    plt.title(f"Per-class {metric}")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()

def dump_misclassified(paths: List[str], y_true: List[str], y_pred: List[str], out_dir: Path, limit_per_pair: int = 50):
    """
    誤分類画像を True/Pred ごとに仕分けコピー（確認用、任意）
    """
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    counter = {}
    for p, t, pr in zip(paths, y_true, y_pred):
        if t == pr: 
            continue
        key = (t, pr)
        counter[key] = counter.get(key, 0) + 1
        if counter[key] > limit_per_pair:
            continue
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        pair_dir = out_dir / f"true_{t}__pred_{pr}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        dst = pair_dir / Path(p).name
        # 衝突回避
        i = 1
        while dst.exists():
            dst = pair_dir / f"{Path(p).stem}_{i}{Path(p).suffix}"
            i += 1
        try:
            im.save(dst)
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser("Evaluate predictions (per-class metrics, confusion matrix, charts)")
    ap.add_argument("--preds-csv", required=True, help="predict.py の --save-csv で出力したCSV")
    ap.add_argument("--output-dir", required=True, help="評価レポート保存先ディレクトリ")
    ap.add_argument("--include-unknown", action="store_true", help="unknown 予測も評価に含める")
    ap.add_argument("--export-miscls", action="store_true", help="誤分類画像を True/Pred 別にコピー保存")
    ap.add_argument("--miscls-limit", type=int, default=50, help="誤分類コピーの1組み合わせ上限数")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_preds_csv(Path(args.preds_csv))
    labels = build_label_space(rows, include_unknown=args.include_unknown)
    rows_eval = filter_rows(rows, include_unknown=args.include_unknown)

    # メトリクス計算
    report, cm, labels_order, y_true, y_pred = compute_metrics(rows_eval, labels)

    # 保存
    save_json(report, out_dir / "report.json")
    save_report_csv(report, labels_order, out_dir / "report.csv")
    save_confusion_matrix_csv(cm, labels_order, out_dir / "confusion_matrix.csv")
    save_txt_summary(report, labels_order, out_dir / "summary.txt")

    # グラフ保存
    plot_confusion_matrix(cm, labels_order, out_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, labels_order, out_dir / "confusion_matrix_norm.png", normalize=True)
    # per-class bars
    plot_bars(report, labels_order, "precision", out_dir / "precision_per_class.png")
    plot_bars(report, labels_order, "recall",    out_dir / "recall_per_class.png")
    plot_bars(report, labels_order, "f1-score",  out_dir / "f1_per_class.png")
    # support は数なので棒グラフも作る
    plot_bars(report, labels_order, "support",   out_dir / "support_per_class.png")

    # 誤分類ダンプ（任意）
    if args.export_miscls:
        paths = [r["path"] for r in rows_eval if Path(r["path"]).suffix.lower() in VALID_EXTS]
        dump_misclassified(paths, y_true, y_pred, out_dir / "misclassified", limit_per_pair=args.miscls_limit)

    # 完了メッセージ（残るのはファイル群）
    print(f"[OK] Saved evaluation to: {out_dir}")

if __name__ == "__main__":
    main()

