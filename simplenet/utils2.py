import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def save_confusion_matrices(
    y_true,
    y_pred,
    save_dir,
    stem="test",
    labels=("normal", "abnormal"),
    thr=None,
):
    """
    Save confusion matrices:
      - counts (raw)
      - normalized by true label (row-normalized, %)

    Args:
        y_true: array-like of 0/1
        y_pred: array-like of 0/1
        save_dir: output directory
        stem: filename stem
        labels: (label0, label1)
        thr: threshold (for title)
    """
    os.makedirs(save_dir, exist_ok=True)

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # --- counts ---
    title_counts = f"Confusion Matrix ({stem})"
    if thr is not None:
        title_counts += f" thr={thr:.4f}"
    out_counts = os.path.join(save_dir, f"confusion_matrix_counts_{stem}.png")
    _plot_confusion_matrix(
        cm=cm,
        labels=labels,
        normalize=False,
        title=title_counts,
        save_path=out_counts,
        cmap="Blues",
    )

    # --- normalized (%), row-normalized like PatchCore ---
    cm_norm = cm.astype(float)
    row_sum = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sum, out=np.zeros_like(cm_norm), where=row_sum != 0)

    title_norm = f"Confusion Matrix (normalized) ({stem})"
    if thr is not None:
        title_norm += f" thr={thr:.4f}"
    out_norm = os.path.join(save_dir, f"confusion_matrix_norm_{stem}.png")
    _plot_confusion_matrix(
        cm=cm_norm,
        labels=labels,
        normalize=True,
        title=title_norm,
        save_path=out_norm,
        cmap="Blues",
    )


def _plot_confusion_matrix(cm, labels, normalize, title, save_path, cmap="Blues"):
    """
    cm:
      - if normalize=False: counts (int)
      - if normalize=True: 0..1 float (row-normalized)
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # annotate
    for i in range(2):
        for j in range(2):
            if normalize:
                text = f"{cm[i, j]*100:.1f}"
            else:
                text = f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", color="black")

    # for normalized, colorbar 0..1 にしたい場合はそのままでOK
    # counts の時もそのままでOK（PatchCoreと同じく値は枠内に出る）

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
