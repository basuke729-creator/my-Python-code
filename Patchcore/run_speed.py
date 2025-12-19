import contextlib
import logging
import os
import sys
import time

import click
import numpy as np
import torch

import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


def _cuda_sync(device):
    if device.type.lower().startswith("cuda"):
        torch.cuda.synchronize(device=device)


def benchmark_predict(
    PatchCore,
    dataloader,
    device,
    iters=105,
    warmup=5,
):
    n_images = len(dataloader.dataset)
    times = []

    for i in range(iters):
        _cuda_sync(device)
        t0 = time.perf_counter()

        PatchCore.predict(dataloader)

        _cuda_sync(device)
        t1 = time.perf_counter()

        dt = t1 - t0
        times.append(dt)

        if i < warmup:
            LOGGER.info(
                "[BENCH] warmup %d/%d: %.3f s (%.3f ms/img)",
                i + 1,
                warmup,
                dt,
                (dt / n_images) * 1000.0,
            )

    times = np.asarray(times[warmup:], dtype=np.float64)

    mean_run = float(times.mean())
    p50_run = float(np.percentile(times, 50))
    p95_run = float(np.percentile(times, 95))

    sec_per_img = mean_run / n_images
    fps = n_images / mean_run

    LOGGER.info("========== BENCH RESULT ==========")
    LOGGER.info("Images        : %d", n_images)
    LOGGER.info("Runs (valid)  : %d", len(times))
    LOGGER.info("Mean / run    : %.3f s", mean_run)
    LOGGER.info("P50 / run     : %.3f s", p50_run)
    LOGGER.info("P95 / run     : %.3f s", p95_run)
    LOGGER.info("Time / image  : %.3f ms", sec_per_img * 1000.0)
    LOGGER.info("FPS           : %.3f", fps)
    LOGGER.info("==================================")


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True)
@click.option("--seed", type=int, default=0)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
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
):
    methods = {key: item for (key, item) in methods}

    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device(f"cuda:{device.index}")
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Dataset [{}] ({}/{})".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        with device_context:
            torch.cuda.empty_cache()

            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](device)
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)

            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info("Training ({}/{})".format(i + 1, len(PatchCore_list)))
                PatchCore.fit(dataloaders["training"])

            torch.cuda.empty_cache()

            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info("Benchmarking predict ({}/{})".format(i + 1, len(PatchCore_list)))
                benchmark_predict(
                    PatchCore=PatchCore,
                    dataloader=dataloaders["testing"],
                    device=device,
                    iters=105,
                    warmup=5,
                )

        LOGGER.info("-----")


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
        pcs = []
        for backbone_name, layers in zip(backbone_names, layers_to_extract_from_coll):
            backbone = patchcore.backbones.load(backbone_name)
            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            pc = patchcore.patchcore.PatchCore(device)
            pc.load(
                backbone=backbone,
                layers_to_extract_from=layers,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            pcs.append(pc)
        return pcs

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1)
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
@click.option("--train_val_split", type=float, default=1)
@click.option("--batch_size", default=2)
@click.option("--num_workers", default=8)
@click.option("--resize", default=256)
@click.option("--imagesize", default=224)
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
        dls = []
        for sub in subdatasets:
            train_ds = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=sub,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )

            test_ds = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=sub,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dl = torch.utils.data.DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dl = torch.utils.data.DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dl.name = name + "_" + sub
            dls.append({"training": train_dl, "testing": test_dl})

        return dls

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()



