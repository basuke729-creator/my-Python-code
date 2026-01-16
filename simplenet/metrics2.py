"""Anomaly metrics."""
import cv2
import numpy as np
from sklearn import metrics

import pandas as pd
from skimage import measure


def _as_numpy(x):
    if isinstance(x, list):
        return np.stack(x)
    return np.asarray(x)


def _has_two_classes(y_true_1d: np.ndarray) -> bool:
    y = np.asarray(y_true_1d).astype(int).reshape(-1)
    uniq = np.unique(y)
    return uniq.size >= 2


def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels):
    """
    Computes retrieval statistics (AUROC, FPR, TPR, thresholds, PR-AUC, best F1 threshold).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] anomaly scores (higher = more anomalous)
        anomaly_ground_truth_labels: [np.array or list] [N] binary labels (1=anomaly, 0=normal)
    """
    scores = np.asarray(anomaly_prediction_weights).reshape(-1)
    labels = np.asarray(anomaly_ground_truth_labels).astype(int).reshape(-1)

    if not _has_two_classes(labels):
        return {
            "auroc": np.nan,
            "auc_pr": np.nan,
            "fpr": np.array([]),
            "tpr": np.array([]),
            "threshold": np.array([]),
            "best_threshold_f1": None,
            "skipped": True,
            "reason": "Only one class present in image-level ground truth labels.",
        }

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    auroc = metrics.roc_auc_score(labels, scores)

    precision, recall, thr_pr = metrics.precision_recall_curve(labels, scores)
    auc_pr = metrics.auc(recall, precision)

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    best_threshold_f1 = None
    if len(f1) > 1 and len(thr_pr) > 0:
        best_idx = int(np.argmax(f1[:-1]))  # last point has no threshold
        if best_idx < len(thr_pr):
            best_threshold_f1 = float(thr_pr[best_idx])

    return {
        "auroc": float(auroc),
        "auc_pr": float(auc_pr),
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
        "best_threshold_f1": best_threshold_f1,
        "skipped": False,
        "reason": "",
    }


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics for anomaly maps and GT masks.

    For datasets WITHOUT pixel-level GT masks (common in posture safety),
    this safely skips and returns NaNs instead of crashing.
    """
    anomaly_segmentations = _as_numpy(anomaly_segmentations)

    if ground_truth_masks is None:
        return {
            "auroc": np.nan,
            "fpr": np.array([]),
            "tpr": np.array([]),
            "optimal_threshold": None,
            "optimal_fpr": np.nan,
            "optimal_fnr": np.nan,
            "skipped": True,
            "reason": "ground_truth_masks is None",
        }

    ground_truth_masks = _as_numpy(ground_truth_masks)

    flat_anomaly = np.asarray(anomaly_segmentations).reshape(-1)
    flat_gt = np.asarray(ground_truth_masks).astype(int).reshape(-1)

    if not _has_two_classes(flat_gt):
        return {
            "auroc": np.nan,
            "fpr": np.array([]),
            "tpr": np.array([]),
            "optimal_threshold": None,
            "optimal_fpr": np.nan,
            "optimal_fnr": np.nan,
            "skipped": True,
            "reason": "Only one class present in pixel-level ground truth mask",
        }

    fpr, tpr, _ = metrics.roc_curve(flat_gt, flat_anomaly)
    auroc = metrics.roc_auc_score(flat_gt, flat_anomaly)

    precision, recall, thr_pr = metrics.precision_recall_curve(flat_gt, flat_anomaly)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    best_idx = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    optimal_threshold = thr_pr[best_idx] if len(thr_pr) > 0 else None

    if optimal_threshold is None:
        return {
            "auroc": float(auroc),
            "fpr": fpr,
            "tpr": tpr,
            "optimal_threshold": None,
            "optimal_fpr": np.nan,
            "optimal_fnr": np.nan,
            "skipped": False,
            "reason": "PR thresholds not available",
        }

    predictions = (flat_anomaly >= optimal_threshold).astype(int)
    fp = np.logical_and(predictions == 1, flat_gt == 0).mean()
    fn = np.logical_and(predictions == 0, flat_gt == 1).mean()

    return {
        "auroc": float(auroc),
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": float(optimal_threshold),
        "optimal_fpr": float(fp),
        "optimal_fnr": float(fn),
        "skipped": False,
        "reason": "",
    }


def compute_pro(masks, amaps, num_th=200):
    """
    PRO score (requires pixel-level masks).
    If you don't have pixel GT masks, this returns NaN safely.
    """
    masks = _as_numpy(masks).astype(np.uint8)
    amaps = _as_numpy(amaps).astype(np.float32)

    flat_gt = masks.reshape(-1)
    if not _has_two_classes(flat_gt):
        return np.nan

    if num_th <= 0:
        return np.nan

    min_th = float(amaps.min())
    max_th = float(amaps.max())
    if max_th == min_th:
        return np.nan

    delta = (max_th - min_th) / float(num_th)

    rows = []
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = False
        binary_amaps[amaps > th] = True

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)

            labeled = measure.label(mask)
            regions = measure.regionprops(labeled)
            for region in regions:
                coords = region.coords
                tp_pixels = binary_amap[coords[:, 0], coords[:, 1]].sum()
                pros.append(tp_pixels / float(region.area))

        if len(pros) == 0:
            continue

        inverse_masks = (1 - masks).astype(bool)
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        denom = inverse_masks.sum()
        if denom == 0:
            continue

        fpr = fp_pixels / float(denom)
        rows.append({"pro": float(np.mean(pros)), "fpr": float(fpr), "threshold": float(th)})

    if len(rows) == 0:
        return np.nan

    df = pd.DataFrame(rows)
    df = df[df["fpr"] < 0.3]
    if df.empty:
        return np.nan

    df["fpr"] = df["fpr"] / df["fpr"].max()
    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return float(pro_auc)
