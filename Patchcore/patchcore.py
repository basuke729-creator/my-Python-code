import contextlib
import logging
import os
import sys
import shutil

import click
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


def plot_confusion_matrices(cm, class_names, save_dir, dataset_name):
    """枚数ベース＋行正規化(%)の混同行列を保存する。"""
    os.makedirs(save_dir, exist_ok=True)

    # ===== 枚数ベース =====
    fig, ax = plt.subplots()
    im = ax.imshow(cm, vmin=0, vmax=cm.max())
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("PatchCore Confusion Matrix (counts)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() * 0.5 else "black",
            )

    fig.colorbar(im, ax=ax)
    count_path = os.path.join(save_dir, f"confusion_matrix_counts_{dataset_name}.png")
    fig.savefig(count_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    # ===== 行方向で正規化(%) =====
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)  # 0割り対策

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
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )

    fig.colorbar(im, ax=ax)
    norm_path = os.path.join(save_dir, f"confusion_matrix_norm_{dataset_name}.png")
    fig.savefig(norm_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    LOGGER.info("Saved confusion matrices to:")
    LOGGER.info("  counts: %s", count_path)
    LOGGER.info("  norm  : %s", norm_path)


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
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
    # GPU メモリリーク対策のため context manager を使う
    device_context = (
        torch.cuda.device(f"cuda:{device.index}")
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, device)
        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](device)
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)

            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )

            # ====== 学習 ======
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info("Training models ({}/{})".format(i + 1, len(PatchCore_list)))
                torch.cuda.empty_cache()
                PatchCore.fit(dataloaders["training"])

            # ====== 推論 ======
            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            # Ensemble 正規化
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
            segmentations = (segmentations - min_scores) / (
                max_scores - min_scores + 1e-12
            )
            segmentations = np.mean(segmentations, axis=0)

            # True label（元データに対して "good" 以外を異常とみなす）
            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # ===== (オプション) セグメンテーション画像の保存 =====
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            # ===== 評価指標 =====
            LOGGER.info("Computing evaluation metrics.")
            image_metrics = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )
            auroc = image_metrics["auroc"]

            # Pixel-wise AUROC
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            # 異常ピクセルを含む画像だけに絞った AUROC
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
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
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # ===== 混同行列 (2×2) の作成 =====
            class_names = ["normal", "abnormal"]
            y_true = np.asarray(anomaly_labels, dtype=int)

            # しきい値：metrics に optimal_threshold があればそれを使用
            # なければスコアの 95 パーセンタイル
            threshold = image_metrics.get(
                "optimal_threshold", float(np.percentile(scores, 95.0))
            )
            y_pred = (scores >= threshold).astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_save_dir = os.path.join(run_save_path, "confusion_matrices")
            plot_confusion_matrices(cm, class_names, cm_save_dir, dataset_name)

            # ===== テスト画像を normal / abnormal に仕分け =====
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

            # ===== PatchCore モデルの保存 (オプション) =====
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # ===== 全データセットの結果を CSV に保存 =====
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
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline).
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
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
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

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

            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }
            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()

