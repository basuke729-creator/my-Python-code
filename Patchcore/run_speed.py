import contextlib
import inspect
import logging
import os
import shutil
import sys
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
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
    os.makedirs(save_dir, exist_ok=True)

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


def _cuda_sync_if_needed(device: torch.device):
    if device.type.lower().startswith("cuda"):
        torch.cuda.synchronize(device=device)


def _resolve_model_dir(path: str) -> str:
    if path is None:
        raise click.ClickException("--load_model_path is required when --skip_training is used.")
    p = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(p):
        return p
    if os.path.isfile(p):
        return os.path.dirname(p)
    raise click.ClickException(f"Model path does not exist: {p}")


def _load_from_path_compat(patchcore_obj, model_dir: str, device: torch.device, nn_method):
    fn = patchcore_obj.load_from_path
    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())

    kwargs = {}
    if "device" in param_names:
        kwargs["device"] = device
    if "nn_method" in param_names:
        kwargs["nn_method"] = nn_method

    if kwargs:
        return fn(model_dir, **kwargs)

    try:
        return fn(model_dir, device, nn_method)
    except TypeError:
        return fn(model_dir)


def _benchmark_patchcore_infer_fps(
    PatchCore_list,
    dataloader_single,
    device: torch.device,
    bench_iters: int = 105,
    bench_warmup: int = 5,
):
    if not (bench_iters > bench_warmup):
        raise ValueError("bench_iters must be > bench_warmup")

    torch.backends.cudnn.benchmark = True

    times = []

    for it in range(bench_iters):
        _cuda_sync_if_needed(device)
        t0 = time.perf_counter()

        aggregator = {"scores": [], "segmentations": []}
        labels_gt = None
        masks_gt = None

        for PatchCore in PatchCore_list:
            scores, segmentations, labels_gt, masks_gt = PatchCore.predict(dataloader_single)
            aggregator["scores"].append(scores)
            aggregator["segmentations"].append(segmentations)

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

        _cuda_sync_if_needed(device)
        t1 = time.perf_counter()

        dt = t1 - t0
        times.append(dt)

        if it < bench_warmup:
            LOGGER.info("[BENCH] warmup %d/%d: %.3f ms", it + 1, bench_warmup, dt * 1000.0)

    kept = np.array(times[bench_warmup:], dtype=np.float64)
    mean_s = float(kept.mean())
    fps = 1.0 / mean_s if mean_s > 0 else float("inf")
    p50 = float(np.percentile(kept, 50) * 1000.0)
    p95 = float(np.percentile(kept, 95) * 1000.0)

    return {
        "mean_ms": mean_s * 1000.0,
        "fps": fps,
        "p50_ms": p50,
        "p95_ms": p95,
        "n_kept": int(len(kept)),
    }


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
@click.option("--benchmark_fps", is_flag=True)
@click.option("--bench_iters", type=int, default=105, show_default=True)
@click.option("--bench_warmup", type=int, default=5, show_default=True)
@click.option("--bench_image_index", type=int, default=0, show_default=True)
@click.option("--load_model_path", type=str, default=None)
@click.option("--skip_training", is_flag=True)
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
    benchmark_fps,
    bench_iters,
    bench_warmup,
    bench_image_index,
    load_model_path,
    skip_training,
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

            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()

                if skip_training:
                    model_dir = _resolve_model_dir(load_model_path)
                    nn_method = getattr(PatchCore, "nn_method", None)
                    if nn_method is None:
                        raise click.ClickException(
                            "nn_method not found on PatchCore instance. Ensure get_patchcore sets PatchCore.nn_method."
                        )
                    LOGGER.info("Loading model (%d/%d) from: %s", i + 1, len(PatchCore_list), model_dir)
                    _load_from_path_compat(PatchCore, model_dir, device, nn_method)
                else:
                    if PatchCore.backbone.seed is not None:
                        patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                    LOGGER.info("Training model (%d/%d)", i + 1, len(PatchCore_list))
                    PatchCore.fit(dataloaders["training"])

            if benchmark_fps:
                test_dataset = dataloaders["testing"].dataset
                n_test = len(test_dataset)
                if n_test <= 0:
                    raise RuntimeError("Testing dataset is empty; cannot benchmark.")

                idx = int(bench_image_index) % n_test
                subset = torch.utils.data.Subset(test_dataset, [idx])

                dataloader_single = torch.utils.data.DataLoader(
                    subset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )

                LOGGER.info(
                    "[BENCH] Start: iters=%d warmup=%d image_index=%d test_size=%d ensemble=%d",
                    bench_iters,
                    bench_warmup,
                    idx,
                    n_test,
                    len(PatchCore_list),
                )

                stats = _benchmark_patchcore_infer_fps(
                    PatchCore_list=PatchCore_list,
                    dataloader_single=dataloader_single,
                    device=device,
                    bench_iters=bench_iters,
                    bench_warmup=bench_warmup,
                )

                LOGGER.info("[BENCH] mean: %.3f ms / image", stats["mean_ms"])
                LOGGER.info("[BENCH] FPS : %.2f", stats["fps"])
                LOGGER.info("[BENCH] p50 : %.3f ms, p95 : %.3f ms (n=%d)", stats["p50_ms"], stats["p95_ms"], stats["n_kept"])
                LOGGER.info("[BENCH] Done.")
                LOGGER.info("\n\n-----\n")
                continue

            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}

            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info("Embedding test data with model (%d/%d)", i + 1, len(PatchCore_list))
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(dataloaders["testing"])
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

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

            anomaly_labels = [x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate]

            if save_segmentation_images:
                image_paths = [x[2] for x in dataloaders["testing"].dataset.data_to_iterate]
                mask_paths = [x[3] for x in dataloaders["testing"].dataset.data_to_iterate]

                def image_transform(image):
                    ds = dataloaders["testing"].dataset
                    img = ds.transform_img(image)
                    if hasattr(img, "detach"):
                        img = img.detach()
                    img = img.cpu().numpy()

                    mean = getattr(ds, "transform_mean", None)
                    std = getattr(ds, "transform_std", None)
                    if mean is not None and std is not None:
                        mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
                        std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
                        img = img * std + mean

                    if img.max() > 1.5 or img.min() < -0.5:
                        mn, mx = float(img.min()), float(img.max())
                        if mx > mn:
                            img = (img - mn) / (mx - mn)

                    img = np.clip(img, 0.0, 1.0)
                    img = (img * 255.0).astype(np.uint8)
                    img = np.transpose(img, (1, 2, 0))
                    return img

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

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
                LOGGER.info("Saved segmentation images to: %s", image_save_path)

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
                    LOGGER.info("%s: %3.3f", key, item)

            class_names = ["normal", "abnormal"]
            y_true = np.asarray(anomaly_labels, dtype=int)
            threshold = image_metrics.get("optimal_threshold", float(np.percentile(scores, 95.0)))
            y_pred = (scores >= threshold).astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_save_dir = os.path.join(run_save_path, "confusion_matrices")
            plot_confusion_matrices(cm, class_names, cm_save_dir, dataset_name)

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

            if save_patchcore_model:
                patchcore_save_path = os.path.join(run_save_path, "models", dataset_name)
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        f"Ensemble-{i + 1}-{len(PatchCore_list)}_"
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    if len(result_collect) > 0:
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

            patchcore_instance.nn_method = nn_method
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
                    augment=augment,
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
    LOGGER.info("Command line arguments: %s", " ".join(sys.argv))
    main()


