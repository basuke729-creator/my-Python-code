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
    """counts と 行正規化(%) の混同行列を保存"""
    os.makedirs(save_dir, exist_ok=True)

    # ===== counts =====
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
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > vmax * 0.5 else "black",
            )

    fig.colorbar(im, ax=ax)
    count_path = os.path.join(save_dir, f"confusion_matrix_counts_{dataset_name}.png")
    fig.savefig(count_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    # ===== row-normalized (%) =====
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm.astype(np.float32),
            denom,
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


def _get_mean_std_from_dataset(ds):
    """
    環境差が大きいので、存在する属性から平均・分散を拾う。
    無ければ (None, None) を返す。
    """
    mean = None
    std = None

    # まずあなたが以前書いていた属性
    if hasattr(ds, "transform_mean"):
        mean = ds.transform_mean
    if hasattr(ds, "transform_std"):
        std = ds.transform_std

    # 他でありがちな名前
    if mean is None and hasattr(ds, "mean"):
        mean = ds.mean
    if std is None and hasattr(ds, "std"):
        std = ds.std

    # torchvision transforms の Normalize を掘る（入ってる場合だけ）
    # ds.transform_img が Compose のとき、.transforms 内に Normalize があることがある
    try:
        t = getattr(ds, "transform_img", None)
        if (mean is None or std is None) and hasattr(t, "transforms"):
            for tr in t.transforms:
                cname = tr.__class__.__name__.lower()
                if "normalize" in cname:
                    if mean is None:
                        mean = tr.mean
                    if std is None:
                        std = tr.std
                    break
    except Exception:
        pass

    if mean is not None:
        mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
    if std is not None:
        std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)

    return mean, std


def _image_transform_for_plot(ds, image):
    """
    patchcore.utils.plot_segmentation_images() は内部で
    image.transpose(1,2,0) をする (= 入力は CHW を想定)
    なので、ここは必ず (C,H,W) uint8 を返す。
    """
    x = ds.transform_img(image)  # torch Tensor を想定

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        # 形が (H,W,C) の場合は CHW に戻す
        if x.ndim == 3 and x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1)
        x = x.numpy()
    else:
        x = np.asarray(x)

    # ここで x は CHW を保証したい
    if x.ndim != 3:
        raise ValueError(f"Unexpected image tensor shape: {x.shape}")
    if x.shape[0] not in (1, 3):
        # もし HWC が紛れ込んでいたら救済
        if x.shape[2] in (1, 3):
            x = np.transpose(x, (2, 0, 1))
        else:
            raise ValueError(f"Cannot interpret image shape as CHW or HWC: {x.shape}")

    mean, std = _get_mean_std_from_dataset(ds)

    # denorm が取れれば denorm、取れなければ 0..1 想定で 255 スケール
    if (mean is not None) and (std is not None) and mean.shape[0] == x.shape[0]:
        x = (x * std + mean) * 255.0
    else:
        x = x * 255.0

    x = np.clip(x, 0, 255).astype(np.uint8)
    return x  # CHW uint8


def _mask_transform_for_plot(ds, mask):
    """
    plot_segmentation_images() は mask を numpy にして使うだけなので
    ここは HW を返せばOK。
    """
    try:
        m = ds.transform_mask(mask)
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        else:
            m = np.asarray(m)
    except Exception:
        m = np.asarray(mask)

    # もし CHW で来たら HW に落とす
    if m.ndim == 3:
        m = m[0]
    return m


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True, help="ヒートマップ(セグメンテーション画像)を保存する")
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
                LOGGER.info("Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list)))

            # ===== 学習 =====
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info("Training models ({}/{})".format(i + 1, len(PatchCore_list)))
                PatchCore.fit(dataloaders["training"])

            # ===== 推論（test）=====
            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}

            # predict は PatchCore 側の実装に依存：最後に回したモデルの labels_gt/masks_gt を使う想定
            labels_gt = None
            masks_gt = None

            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info("Embedding test data with models ({}/{})".format(i + 1, len(PatchCore_list)))
                scores_i, segmentations_i, labels_gt, masks_gt = PatchCore.predict(dataloaders["testing"])
                aggregator["scores"].append(scores_i)
                aggregator["segmentations"].append(segmentations_i)

            # ===== Ensemble 正規化（公式処理に合わせる）=====
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

            # True label（"good" 以外を異常扱い）
            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # ===== ヒートマップ可視化（segmentation_images）=====
            if save_segmentation_images:
                image_paths = [x[2] for x in dataloaders["testing"].dataset.data_to_iterate]
                mask_paths = [x[3] for x in dataloaders["testing"].dataset.data_to_iterate]

                ds = dataloaders["testing"].dataset

                def image_transform(image):
                    # 必ず CHW uint8 を返す（utils.py 内部 transpose に合わせる）
                    return _image_transform_for_plot(ds, image)

                def mask_transform(mask):
                    return _mask_transform_for_plot(ds, mask)

                image_save_path = os.path.join(run_save_path, "segmentation_images", dataset_name)
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
                LOGGER.info("Saved segmentation/heatmap images to: %s", image_save_path)

            # ===== 評価指標 =====
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
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # ===== 混同行列 =====
            class_names = ["normal", "abnormal"]
            y_true = np.asarray(anomaly_labels, dtype=int)

            # しきい値：metrics に optimal_threshold があればそれ、なければ 95%tile
            threshold = image_metrics.get("optimal_threshold", float(np.percentile(scores, 95.0)))
            y_pred = (scores >= threshold).astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_save_dir = os.path.join(run_save_path, "confusion_matrices")
            plot_confusion_matrices(cm, class_names, cm_save_dir, dataset_name)

            # ===== テスト画像を normal / abnormal に仕分けコピー =====
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

            # ===== PatchCore モデル保存 =====
            if save_patchcore_model:
                patchcore_save_path = os.path.join(run_save_path, "models", dataset_name)
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # ===== 全データセット結果をCSVに保存 =====
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

            # FaissNN の引数はこの repo 実装に合わせる
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

            dataloader_dict = {"training": train_dataloader, "validation": val_dataloader, "testing": test_dataloader}
            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()


