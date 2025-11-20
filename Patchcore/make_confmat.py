import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ★ここを results.csv のフルパスに書き換える
CSV_PATH = "/home/yamamao/Patchcore/results/patchcore/ladder_dataset/version_0/results.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    print("==== columns in results.csv ====")
    print(df.columns)

    """
    anomalib のバージョンによって列名が違う可能性があります。
    よくあるパターンの例：

      - "image_path", "label", "pred_label"
      - "image_path", "gt_label", "pred_label"
      - 0/1 のラベルで normal=0, abnormal=1

    下の行の 'label_col', 'pred_col' を
    print(df.columns) の結果に合わせて書き換えてください。
    """

    # 例: 'label' が正解ラベル, 'pred_label' が予測ラベルだった場合
    label_col = "label"        # 正解ラベルの列名
    pred_col = "pred_label"    # 予測ラベルの列名

    if label_col not in df.columns or pred_col not in df.columns:
        raise RuntimeError(
            f"列名が合っていません。現在の列: {list(df.columns)}\n"
            f"label_col={label_col}, pred_col={pred_col} を実データに合わせて直してください。"
        )

    y_true = df[label_col].values
    y_pred = df[pred_col].values

    # ラベルが文字列（"normal"/"abnormal" など）の場合はそのまま使う
    labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Patchcore)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_patchcore.png", dpi=200)
    print("confusion_matrix_patchcore.png を作成しました。")

if __name__ == "__main__":
    main()
