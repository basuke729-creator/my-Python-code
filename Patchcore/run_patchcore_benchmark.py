import contextlib
import logging
import os
import sys
import time
from pathlib import Path

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


def _faiss_gpu_available() -> bool:
    try:
        import faiss
        return hasattr(faiss, "index_cpu_to_gpu") or hasattr(faiss, "index_cpu_to_all_gpus")
    except Exception:
        return False


def _make_nn_method(faiss_on_gpu: bool, faiss_num_workers: int):
    if faiss_on_gpu and not _faiss_gpu_available():
        LOGGER.warning(
            "--faiss_on_gpu is set but faiss GPU API is not available. Falling back to CPU."
        )
        faiss_on_gpu = False
    return patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)


def _resolve_model_path(model_dir: str, dataset_name: str) -> str:
    p = Path(model_dir)

    if p.is_file():
        return str(p)

    if (p / "patchcore_params.pkl").exists():
        return str(p)

    cand = p / dataset_name
    if cand.exists():
        if (cand / "patchcore_params.pkl").exists():
            return str(cand)
        return str(cand)

    return str(p)


def _bench_predict_patchcore(
    patchcore_instance: patchcore.patchcore.PatchCore,
    dataloader,
    iters: int,
    warmup: int,
    image_index: int,
):
    samples = []
    for batch in dataloader:
        samples.append(batch)
        if len(samples) >= max(1, iters):
            break

    if len(samples) == 0:
        raise RuntimeError("No samples obtained from dataloader.")

    if image_index < 0 or image_index >= len(samples):
        image_index = 0

    class _OneBatchLoader:
        def __init__(self, one_batch):
            self._b = one_batch

        def __iter__(self):
            yield self._b

        def __len__(self):
            return 1

    one_loader = _OneBatchLoader(samples[image_index])

    times = []

    for i in range(warmup + iters):
        t0 = time.perf_counter()
        _ = patchcore_instance.predict(one_loader)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup:
            times.append(t1 - t0)

    mean_s = float(np.mean(times)) if times else float("nan")
    mean_ms = mean_s * 1000.0
    fps = (1.0 / mean_s) if mean_s > 0 else float("nan")
    return mean_ms, fps


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True)
@click.option("--seed", type=int, default=0)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--model_dir", type=str, required=True)
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
@click.option("--bench_iters", type=int, default=105)
@click.option("--bench_warmup", type=int, default=5)
@click.option("--bench_image_index", type=int, default=0)
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
    model_dir,
    faiss_on_gpu,
    faiss_num_workers,
    bench_iters,
    bench_warmup,
    bench_image_index,
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

    nn_method = _make_nn_method(faiss_on_gpu, faiss_num_workers)

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        dataset_name = getattr(dataloaders["training"], "name", f"dataset_{dataloader_count}")

        with device_context:
            torch.cuda.empty_cache()

            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](device)
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)

            resolved = _resolve_model_path(model_dir, dataset_name)

            for PatchCore in PatchCore_list:
                torch.cuda.empty_cache()
                PatchCore.load_from_path(resolved, device=device, nn_method=nn_method)

            test_loader = dataloaders["testing"]

            ms_list, fps_list = [], []
            for PatchCore in PatchCore_list:
                mean_ms, fps = _bench_predict_patchcore(
                    PatchCore,
                    test_loader,
                    iters=bench_iters,
                    warmup=bench_warmup,
                    image_index=bench_image_index,
                )
                ms_list.append(mean_ms)
                fps_list.append(fps)

            LOGGER.info("mean_ms: %.3f", float(np.mean(ms_list)))
            LOGGER.info("fps: %.2f", float(np.mean(fps_list)))

    LOGGER.info("Done: %s", run_save_path)


@main.command("patch_core")
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
@click.option("--patchsize", type=int, default=3)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    anomaly_scorer_num_nn,
    patchsize,
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
                nn_method=None,
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
        else:
            raise ValueError(name)

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, required=True)
@click.option("--batch_size", type=int, default=1)
@click.option("--num_workers", type=int, default=8)
@click.option("--resize", type=int, default=256)
@click.option("--imagesize", type=int, default=224)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        loaders = []
        for sub in subdatasets:
            train_ds = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=sub,
                resize=resize,
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

            train_dl.name = f"{name}_{sub}"
            test_dl.name = f"{name}_{sub}"

            loaders.append({"training": train_dl, "testing": test_dl})
        return loaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()


