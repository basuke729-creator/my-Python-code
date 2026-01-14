#!/usr/bin/env python3
import contextlib
import logging
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


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _move_to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _resolve_model_subdirs(model_dir: Path):
    model_dir = Path(model_dir)
    if model_dir.is_file():
        if model_dir.name == "patchcore_params.pkl":
            return [model_dir.parent]
        return [model_dir]

    pkl_files = sorted(model_dir.rglob("patchcore_params.pkl"))
    if pkl_files:
        return [p.parent for p in pkl_files]

    return [model_dir]


def _build_nn_method(faiss_on_gpu: bool, faiss_num_workers: int):
    return patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)


def _load_patchcore_ensemble(model_dir: Path, device: torch.device, nn_method):
    subdirs = _resolve_model_subdirs(model_dir)
    pcs = []
    for sd in subdirs:
        pc = patchcore.patchcore.PatchCore(device)
        try:
            pc.load_from_path(str(sd), device=device, nn_method=nn_method)
        except TypeError:
            pc.load_from_path(str(sd), device, nn_method)
        pcs.append(pc)
    return pcs


def _collect_batches_for_benchmark(dataloader, bench_iters: int, bench_image_index: int, device, exclude_h2d: bool):
    it = iter(dataloader)

    for _ in range(bench_image_index):
        try:
            next(it)
        except StopIteration:
            it = iter(dataloader)
            next(it)

    batches = []
    while len(batches) < bench_iters:
        try:
            b = next(it)
        except StopIteration:
            it = iter(dataloader)
            b = next(it)

        if exclude_h2d:
            b = _move_to_device(b, device)
        batches.append(b)

    return batches


def _predict_one(pc, batch_or_tensor):
    return pc.predict(batch_or_tensor)


def _predict_many(pc, data):
    for b in data:
        _ = _predict_one(pc, b)


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--model_dir", type=str, required=True)
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8, show_default=True)
@click.option("--bench_iters", type=int, default=105, show_default=True)
@click.option("--bench_warmup", type=int, default=5, show_default=True)
@click.option("--bench_image_index", type=int, default=0, show_default=True)
@click.option("--bench_exclude_io", is_flag=True)
@click.option("--bench_exclude_h2d", is_flag=True)
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
    bench_exclude_io,
    bench_exclude_h2d,
):
    methods = {k: v for (k, v) in methods}
    if "get_dataloaders" not in methods:
        raise click.ClickException("Missing required command: dataset (get_dataloaders).")

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device(f"cuda:{device.index}")
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)

    with device_context:
        patchcore.utils.fix_seeds(seed, device)
        torch.cuda.empty_cache()

        nn_method = _build_nn_method(faiss_on_gpu, faiss_num_workers)
        PatchCore_list = _load_patchcore_ensemble(Path(model_dir), device, nn_method)

        if len(PatchCore_list) > 1:
            LOGGER.info("Utilizing PatchCore Ensemble (N=%d).", len(PatchCore_list))

        for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
            dataset_name = dataloaders["testing"].name
            LOGGER.info(
                "Benchmarking dataset [%s] (%d/%d)...",
                dataset_name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )

            test_loader = dataloaders["testing"]

            if bench_exclude_io:
                cached = _collect_batches_for_benchmark(
                    test_loader,
                    bench_iters=bench_iters,
                    bench_image_index=bench_image_index,
                    device=device,
                    exclude_h2d=bench_exclude_h2d,
                )
                warm_data = cached[:bench_warmup]
                meas_data = cached[bench_warmup:]
            else:
                warm_data = test_loader
                meas_data = test_loader

            LOGGER.info(
                "[BENCH] Start: iters=%d warmup=%d image_index=%d exclude_io=%s exclude_h2d=%s",
                bench_iters,
                bench_warmup,
                bench_image_index,
                bench_exclude_io,
                bench_exclude_h2d,
            )

            with torch.no_grad():
                _sync(device)
                t0_w = time.perf_counter()
                if bench_exclude_io:
                    _predict_many(PatchCore_list[0], warm_data)
                    warm_n = len(warm_data)
                else:
                    _ = PatchCore_list[0].predict(warm_data)
                    warm_n = max(1, bench_warmup)
                _sync(device)
                t1_w = time.perf_counter()
                warm_ms = (t1_w - t0_w) * 1000.0 / max(1, warm_n)
                LOGGER.info("[BENCH] warmup mean: %.3f ms / image (n=%d)", warm_ms, warm_n)

                _sync(device)
                t0 = time.perf_counter()

                if bench_exclude_io:
                    for pc in PatchCore_list:
                        _predict_many(pc, meas_data)
                    n_meas = len(meas_data)
                else:
                    for pc in PatchCore_list:
                        _ = pc.predict(meas_data)
                    n_meas = len(meas_data)

                _sync(device)
                t1 = time.perf_counter()

            total_s = (t1 - t0)
            mean_ms = (total_s * 1000.0) / max(1, n_meas)
            fps = max(1e-12, n_meas / max(1e-12, total_s))

            LOGGER.info("[BENCH] mean: %.3f ms / image (n=%d)", mean_ms, n_meas)
            LOGGER.info("[BENCH] FPS : %.2f", fps)
            LOGGER.info("[BENCH] Done.")

            out_txt = Path(run_save_path) / f"bench_{dataset_name}.txt"
            out_txt.write_text(
                "\n".join(
                    [
                        f"dataset={dataset_name}",
                        f"exclude_io={bench_exclude_io}",
                        f"exclude_h2d={bench_exclude_h2d}",
                        f"warmup_n={warm_n}",
                        f"measured_n={n_meas}",
                        f"warmup_ms_per_image={warm_ms:.6f}",
                        f"mean_ms_per_image={mean_ms:.6f}",
                        f"fps={fps:.6f}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )


@main.command("patch_core")
@click.option("-b", "--backbone", type=str, default="wideresnet50")
@click.option("-le", "--layers_to_extract_from", multiple=True, type=str)
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--anomaly_scorer_num_nn", type=int, default=10)
@click.option("--patchsize", type=int, default=3)
def patch_core(
    backbone,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    anomaly_scorer_num_nn,
    patchsize,
):
    return (
        "patch_core_args",
        dict(
            backbone=backbone,
            layers_to_extract_from=list(layers_to_extract_from),
            pretrain_embed_dimension=pretrain_embed_dimension,
            target_embed_dimension=target_embed_dimension,
            anomaly_scorer_num_nn=anomaly_scorer_num_nn,
            patchsize=patchsize,
        ),
    )


@main.command("sampler")
@click.argument("name", type=str)
@click.option("-p", "--percentage", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    return ("sampler_args", dict(name=name, percentage=percentage))


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=1, type=int, show_default=True)
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
            test_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset
                test_dataloader.name += "_" + subdataset

            dataloader_dict = {"training": train_dataloader, "validation": None, "testing": test_dataloader}
            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: %s", " ".join(sys.argv))
    main()



