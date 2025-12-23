#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import contextlib
import logging
import os
import sys
import shutil
from pathlib import Path

import click
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


# -----------------------------
# Utils: safe image / mask conversion
# -----------------------------
def _imagesize_to_int(imagesize):
    """imagesize が int または (H,W) のどちらでも安全に int(H) を返す"""
    if isinstance(imagesize, (tuple, list)):
        return int(imagesize[0])
    return int(imagesize)


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_hwc_uint8(img):
    """
    img: torch or np
    Return: HWC uint8 (3ch)
    """
    arr = _to_numpy(img)

    # (C,H,W) -> (H,W,C)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    # (H,W) -> (H,W,1)
    if arr.ndim == 2:
        arr = arr[..., None]

    # channel adjust
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]

    # float -> uint8
    if arr.dtype != np.uint8:
        # 0-1 っぽければ 255 スケール、そうでなければ clip
        arr_min, arr_max = float(arr.min()), float(arr.max())
        if 0.0 <= arr_min and arr_max <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def _ensure_hw_mask(mask):
    """
    mask: torch or np
    Return: HW float32 (0..1 or 0..255 ok)
    """
    arr = _to_numpy(mask)

    # (C,H,W) -> (H,W)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr[0]  # まず1chへ

    # (H,W,C) -> (H,W)
    if arr.ndim == 3 and arr.shape[2] in (1, 3):
        arr = arr[:, :, 0]

    if arr.ndim != 2:
        # 最後の保険
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Unexpected mask shape after squeeze: {arr.shape}")

    return arr.astype(np.float32)


def _resize_image_hwc(img_hwc_uint8, out_size):
    """HWC uint8 -> out_size x out_size HWC uint8"""
    pil = Image.fromarray(img_hwc_uint8)
    pil = pil.resize((out_size, out_size), resample=Image.BILINEAR)
    return np.array(pil)


def _resize_mask_hw(mask_hw, out_size):
    """HW float -> out_size x out_size HW float (nearest)"""
    # 0..1/255 どちらでも OK
    pil = Image.fromarray(mask_hw.astype(np.float32))
    pil = pil.resize((out_size, out_size), resample=Image.NEAREST)
    return np.array(pil).astype(np.float32)


# -----------------------------
# Confusion matrices
# -----------------------------
def plot_confusion_matrices(cm, class_names, save_dir, dataset_name):
    """枚数ベース＋行正規化(%)の混同行列を保存（どちらも Blues）"""
    os.makedirs(save_dir, exist_ok=True)

    # counts
    fig, ax = plt.subplots()
    im = ax.imshow(cm, vmin=0, vmax=max(1, cm.max()), cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("PatchCore Confusion Matrix (counts)")

    vmax = max(1, cm.max())
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > vmax * 0.5 else "black",
            )

    fig.colorbar(im, ax=ax)
    count_path = os.path.join(save_dir, f"confusion_matrix_counts_{dataset_name}.png")
    fig.savefig(count_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    # row-normalized %
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm.astype(np.float32), denom,
            out=np.zeros_like(cm, dtype=np.float32),
            where=(denom != 0),
        )

    fig, ax = plt.subplots()
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("PatchCore Confusion Matrix (row-normalized %)")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j] * 100.0
            ax.text(
                j, i, f"{val:.1f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )

    fig.colorbar(im, ax=ax)
    norm_path = os.path.join(save_dir, f"confusion_matrix_norm_{dataset_name}.png")
    fig.savefig(norm_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    LOGGER.info("Saved confusion matrices to:")
    LOGGER.info("  counts: %s", count_path)
    LOGGER.info("  norm  : %s", norm_path)


# -----------------------------
# Heatmap writer (robust)
# -----------------------------
def save_heatmap_triplets(
    save_dir,
    image_paths,
    segmentations,  # (N,H,W) or (N,1,H,W)
    scores,         # (N,)
    mask_paths=None,
    dataset=None,
):
    """
    右の例みたいに
      [Image] [Image + Anomaly Map] [Image + Pred Mask(輪郭)]
    を1枚のpngにして保存する（shape事故が起きないよう全部防御）
    """
    os.makedirs(save_dir, exist_ok=True)

    # seg shape normalize -> (N,H,W)
    seg = np.asarray(segmentations)
    if seg.ndim == 4 and seg.shape[1] == 1:
        seg = seg[:, 0]
    if seg.ndim != 3:
        raise ValueError(f"Unexpected segmentations shape: {seg.shape}")

    out_size = None
    if dataset is not None and hasattr(dataset, "imagesize"):
        out_size = _imagesize_to_int(dataset.imagesize)

    for idx, img_path in enumerate(image_paths):
        try:
            # --- load original image ---
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)  # HWC uint8

            # resize image to out_size if available (segmentation側と揃える)
            if out_size is not None:
                img_np_rs = _resize_image_hwc(img_np, out_size)
            else:
                img_np_rs = img_np

            # --- anomaly map ---
            amap = seg[idx]
            amap = _ensure_hw_mask(amap)

            # segmentation map sizeに合わせる
            if out_size is not None and (amap.shape[0] != out_size or amap.shape[1] != out_size):
                amap = _resize_mask_hw(amap, out_size)

            # normalize to 0..1 for colormap
            a_min, a_max = float(amap.min()), float(amap.max())
            amap_norm = (amap - a_min) / (a_max - a_min + 1e-12)

            # threshold for predicted mask (簡易: 99 percentile)
            thr = float(np.percentile(amap_norm, 99.0))
            pred_mask = (amap_norm >= thr).astype(np.uint8)  # 0/1

            # --- plot ---
            fig = plt.figure(figsize=(12, 4))
            ax1 = fig.add_subplot(1, 3, 1)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)

            ax1.imshow(img_np_rs)
            ax1.set_title("Image")
            ax1.axis("off")

            ax2.imshow(img_np_rs)
            ax2.imshow(amap_norm, alpha=0.55, cmap="jet")
            ax2.set_title("Image + Anomaly Map")
            ax2.axis("off")

            ax3.imshow(img_np_rs)
            # 輪郭線（赤）
            ax3.contour(pred_mask, levels=[0.5], colors="red", linewidths=2)
            ax3.set_title("Image + Pred Mask")
            ax3.axis("off")

            base = Path(img_path).stem
            out_path = os.path.join(save_dir, f"{base}_score_{scores[idx]:.6f}.png")
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)

        except Exception as e:
            LOGGER.warning("Failed to save heatmap for %s: %s", img_path, e)


# -----------------------------
# Click CLI
# -----------------------------
@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True, help="ヒートマップ画像を保存")
@click.option("--save_patchcore_model", is_flag=True)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device(f"cuda:{device.index}")
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [%s] (%d/%d)...",
            dataloaders["training"].name,
            dataloader_count + 1,
            len(list_of_dataloaders),
        )

        patchcore.utils.fix_seeds(seed, device)
        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()

            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](device)
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)

            if len(PatchCore_list) > 1:
                LOGGER.info("Utilizing PatchCore Ensemble (N=%d).", len(PatchCore_list))

            # train
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info("Training models (%d/%d)", i + 1, len(PatchCore_list))
                PatchCore.fit(dataloaders["training"])

            # infer (test)
            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}

            labels_gt = None
            masks_gt = None
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info("Embedding test data with models (%d/%d)", i + 1, len(PatchCore_list))
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(dataloaders["testing"])
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            # ensemble normalize (official-like)
            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores + 1e-12)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores + 1e-12)
            segmentations = np.mean(segmentations, axis=0)

            # true label: "good" 以外を異常
            anomaly_labels = [x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate]

            # ---- heatmaps ----
            if save_segmentation_images:
                image_paths = [x[2] for x in dataloaders["testing"].dataset.data_to_iterate]
                # mask_paths は無いデータセットもあるので optional
                mask_paths = []
                for x in dataloaders["testing"].dataset.data_to_iterate:
                    if len(x) >= 4:
                        mask_paths.append(x[3])
                    else:
                        mask_paths.append(None)

                image_save_path = os.path.join(run_save_path, "segmentation_images_v2", dataset_name)
                save_heatmap_triplets(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths=mask_paths,
                    dataset=dataloaders["testing"].dataset,
                )
                LOGGER.info("Saved segmentation/heatmap images to: %s", image_save_path)

            # metrics
            LOGGER.info("Computing evaluation metrics.")
            image_metrics = patchcore.metrics.compute_imagewise_retrieval_metrics(scores, anomaly_labels)
            auroc = image_metrics["auroc"]

            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(segmentations, masks_gt)
            full_pixel_auroc = pixel_scores["auroc"]

            sel_idxs = [i for i in range(len(masks_gt)) if np.sum(masks_gt[i]) > 0]
            if len(sel_idxs) > 0:
                pixel_scores_anom = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                    [segmentations[i] for i in sel_idxs],
                    [masks_gt[i] for i in sel_idxs],
                )
                anomaly_pixel_auroc = pixel_scores_anom["auroc"]
            else:
                anomaly_pixel_auroc = float("nan")

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )
            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("%s: %.3f", key, item)

            # confusion matrix
            class_names = ["normal", "abnormal"]
            y_true = np.asarray(anomaly_labels, dtype=int)

            threshold = image_metrics.get("optimal_threshold", float(np.percentile(scores, 95.0)))
            y_pred = (scores >= threshold).astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_save_dir = os.path.join(run_save_path, "confusion_matrices")
            plot_confusion_matrices(cm, class_names, cm_save_dir, dataset_name)

            # sort images copy
            sorted_base = os.path.join(run_save_path, "sorted_images", dataset_name)
            normal_dir = os.path.join(sorted_base, "normal")
            abnormal_dir = os.path.join(sorted_base, "abnormal")
            os.makedirs(normal_dir, exist_ok=True)
            os.makedirs(abnormal_dir, exist_ok=True)

            test_data = dataloaders["testing"].dataset.data_to_iterate
            for idx, score_val in enumerate(scores):
                img_path = test_data[idx][2]
                pred_label = "abnormal" if score_val >= threshold else "normal"
                dst_dir = abnormal_dir if pred_label == "abnormal" else normal_dir
                try:
                    shutil.copy2(img_path, dst_dir)
                except Exception as e:
                    LOGGER.warning("Could not copy image %s: %s", img_path, e)

            LOGGER.info("Saved sorted images to: %s", sorted_base)

            # save model
            if save_patchcore_model:
                patchcore_save_path = os.path.join(run_save_path, "models", dataset_name)
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = f"Ensemble-{i+1}-{len(PatchCore_list)}_" if len(PatchCore_list) > 1 else ""
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # store final csv
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("patch_core")
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(backbone_names, layers_to_extract_from_coll):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])

            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            # FaissNN の引数差異を吸収（バージョン差で keyword が無いケースがあるため positional）
            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)
        else:
            raise click.ClickException(f"Unknown sampler: {name}")

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None

            dataloader_dict = {"training": train_dataloader, "validation": val_dataloader, "testing": test_dataloader}
            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: %s", " ".join(sys.argv))
    main()



