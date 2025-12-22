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


def cuda_sync(device):
    if device.type.lower().startswith("cuda"):
        torch.cuda.synchronize(device=device)


def benchmark_paper_style(PatchCore, dataloader, device):
    num_images = len(dataloader.dataset)

    cuda_sync(device)
    t0 = time.perf_counter()

    PatchCore.predict(dataloader)

    cuda_sync(device)
    t1 = time.perf_counter()

    total_time = t1 - t0
    time_per_image = total_time / num_images
    fps = num_images / total_time

    return {
        "images": num_images,
        "total_time": total_time,
        "time_per_image": time_per_image,
        "fps": fps,
    }


def benchmark_single_image(PatchCore, dataset, device, runs=105, warmup=5):
    times = []

    for i in range(runs):
        subset = torch.utils.data.Subset(dataset, [i])
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        cuda_sync(device)
        t0 = time.perf_counter()

        PatchCore.predict(loader)

        cuda_sync(device)
        t1 = time.perf_counter()

        times.append(t1 - t0)

    valid = times[warmup:]
    mean_time = float(np.mean(valid))
    fps = 1.0 / mean_time

    return {
        "runs": len(valid),
        "mean_time": mean_time,
        "fps": fps,
    }


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True)
@click.option("--seed", type=int, default=0)
def main(**kwargs):
    pass


@main.result_callback()
def run(methods, results_path, gpu, seed):
    methods = {k: v for (k, v) in methods.items()}

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device(f"cuda:{device.index}")
        if device.type.lower().startswith("cuda")
        else contextlib.suppress()
    )

    dataloaders_list = methods["get_dataloaders"](seed)

    for dataloaders in dataloaders_list:
        dataset_name = dataloaders["training"].name
        LOGGER.info("Dataset: %s", dataset_name)

        with device_context:
            torch.cuda.empty_cache()

            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](device)
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)

            for PatchCore in PatchCore_list:
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                PatchCore.fit(dataloaders["training"])

            PatchCore = PatchCore_list[0]

            LOGGER.info("Running paper-style benchmark...")
            paper = benchmark_paper_style(
                PatchCore, dataloaders["testing"], device
            )

            LOGGER.info("Running single-image benchmark...")
            single = benchmark_single_image(
                PatchCore, dataloaders["testing"].dataset, device
            )

        LOGGER.info("===== BENCHMARK RESULT =====")
        LOGGER.info("[Paper-style]")
        LOGGER.info("Images       : %d", paper["images"])
        LOGGER.info("Total time   : %.4f s", paper["total_time"])
        LOGGER.info("Time / image : %.4f s", paper["time_per_image"])
        LOGGER.info("FPS          : %.2f", paper["fps"])

        LOGGER.info("[Single-image]")
        LOGGER.info("Runs         : %d", single["runs"])
        LOGGER.info("Time / image : %.4f s", single["mean_time"])
        LOGGER.info("FPS          : %.2f", single["fps"])

        ratio = single["mean_time"] / paper["time_per_image"]
        LOGGER.info("Slowdown (single / paper): %.2fx", ratio)
        LOGGER.info("============================\n")


@main.command("patch_core")
@click.option("--backbone_names", "-b", type=str, multiple=True)
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True)
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--patchsize", type=int, default=3)
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    patchsize,
    anomaly_scorer_num_nn,
    faiss_on_gpu,
    faiss_num_workers,
):
    def get_patchcore(input_shape, sampler, device):
        pcs = []
        for name in backbone_names:
            backbone = patchcore.backbones.load(name)
            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
            pc = patchcore.patchcore.PatchCore(device)
            pc.load(
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
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--subdatasets", "-d", multiple=True, required=True)
@click.option("--batch_size", default=2)
@click.option("--num_workers", default=8)
@click.option("--resize", default=256)
@click.option("--imagesize", default=224)
def dataset(
    name,
    data_path,
    subdatasets,
    batch_size,
    num_workers,
    resize,
    imagesize,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        loaders = []
        for sd in subdatasets:
            train = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=sd,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
            )
            test = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=sd,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            loaders.append(
                {
                    "training": torch.utils.data.DataLoader(
                        train,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                    ),
                    "testing": torch.utils.data.DataLoader(
                        test,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                    ),
                }
            )
        return loaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line: %s", " ".join(sys.argv))
    main()
