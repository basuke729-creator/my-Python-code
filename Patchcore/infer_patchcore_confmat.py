# infer_patchcore_confmat.py
import argparse, os, glob, subprocess, pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def find_weights(exp_dir: str) -> str:
    # anomalib の出力 ckpt を探索
    candidates = glob.glob(os.path.join(exp_dir, "**", "weights.ckpt"), recursive=True)
    if not candidates:
        raise FileNotFoundError("weights.ckpt が見つかりません。先に学習を完了させてください。")
    # 最新を一つ返す
    return max(candidates, key=os.path.getmtime)

def find_image_scores(exp_dir: str) -> str:
    # anomalib test 実行後に出力されるスコアCSV（バージョンによりパスが異なる場合あり）
    # 代表例: {exp}/results/*/image_scores.csv
    candidates = glob.glob(os.path.join(exp_dir, "**", "image_scores.csv"), recursive=True)
    if not candidates:
        raise FileNotFoundError("image_scores.csv が見つかりません。anomalib test 実行後に生成されます。")
    return max(candidates, key=os.path.getmtime)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root")
    ap.add_argument("--exp",  required=True, help="experiment root (train時 --out と同じ)")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data)
    exp_root  = os.path.abspath(args.exp)

    # 学習済み重みのパス
    ckpt = find_weights(exp_root)
    # 学習時に使った config を取得（そのまま流用）
    cfgs = glob.glob(os.path.join(exp_root, "**", "patchcore_cls.yaml"), recursive=True)
    if not cfgs:
        raise FileNotFoundError("patchcore_cls.yaml が見つかりません。train_patchcore.py で生成されたものを使用します。")
    cfg = max(cfgs, key=os.path.getmtime)

    print(f"[INFO] weights: {ckpt}")
    print(f"[INFO] config : {cfg}")

    # anomalib test を実行（画像レベルのメトリクス出力＆スコアCSV保存）
    # 参考: anomalib docs（Inference/Deploy, CLI） :contentReference[oaicite:3]{index=3}
    cmd = ["anomalib", "test", "--config", cfg, "--weights", ckpt]
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 出力された image_scores.csv を探す
    scores_csv = find_image_scores(exp_root)
    print(f"[INFO] scores csv: {scores_csv}")

    # 読み込み（列名は anomalib 版によって多少異なる場合あり）
    df = pd.read_csv(scores_csv)

    # ラベル列・スコア列の推定（一般的な列名の優先順で取得）
    label_col_candidates = [c for c in ["label", "target", "gt_label"] if c in df.columns]
    score_col_candidates = [c for c in ["score", "anomaly_score", "image_score"] if c in df.columns]
    if not label_col_candidates or not score_col_candidates:
        raise RuntimeError(f"想定列が見つかりません。columns={list(df.columns)}")

    y_true = df[label_col_candidates[0]].astype(int)     # normal=0, abnormal=1 を想定
    scores = df[score_col_candidates[0]].astype(float)

    # しきい値は “安全(=0) 側の上位パーセンタイル” で初期化（要件で微調整）
    th = scores[y_true == 0].quantile(0.99)
    y_pred = (scores > th).astype(int)

    print(f"\n[INFO] threshold (99th % of normal): {th:.6f}")
    print("\n=== Confusion Matrix (TN FP / FN TP) ===")
    print(confusion_matrix(y_true, y_pred))
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, digits=3))

    # 保存
    out_txt = os.path.join(exp_root, "confusion_report.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"threshold: {th:.6f}\n\n")
        f.write("Confusion Matrix (TN FP / FN TP)\n")
        f.write(str(confusion_matrix(y_true, y_pred)) + "\n\n")
        f.write("Classification Report\n")
        f.write(classification_report(y_true, y_pred, digits=3))
    print(f"\n[OK] レポート保存: {out_txt}")

if __name__ == "__main__":
    main()
