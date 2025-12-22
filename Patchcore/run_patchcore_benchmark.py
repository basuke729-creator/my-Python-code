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


def _sync_if_cuda(device):
    if device is not None and "cuda" in device.type.lower():
        torch.cuda.synchronize(device=device)


def _subset_dataloader(base_dataset, indices, batch_size=1, num_workers=0, pin_memory=True):
    subset = torch.utils.data.Subset(base_dataset, indices)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--model_dir", type=str, default="", show_default=True)
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8, show_default=True)
@click.option("--bench_iters", type=int, default=105, show_default=True)
@click.option("--bench_warmup", type=int, default=5, show_default=True)
@click.option("--bench_image_index", type=int, default=0, show_default=True)
@click.option("--bench_num_workers", type=int, default=0, show_default=True)
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
    bench_num_workers,
):
    methods = {k: v for (k, v) in methods}

    if bench_warmup < 0 or bench_iters <= 0 or bench_warmup >= bench_iters:
        raise click.ClickException("bench_warmup must be >=0 and < bench_iters.")

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

    nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

    with device_context:
        for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
            train_loader = dataloaders["training"]
            test_loader = dataloaders["testing"]

            dataset_name = getattr(train_loader, "name", None)
            if dataset_name is None:
                dataset_name = "dataset_{}".format(dataloader_count)

            LOGGER.info(
                "Benchmarking dataset [{}] ({}/{})...".format(
                    dataset_name, dataloader_count + 1, len(list_of_dataloaders)
                )
            )

            patchcore.utils.fix_seeds(seed, device)
            torch.cuda.empty_cache() if "cuda" in device.type.lower() else None

            imagesize = train_loader.dataset.imagesize
            sampler = methods["get_sampler"](device)
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)

            if model_dir:
                load_dir = model_dir
            else:
                load_dir = os.path.join(run_save_path, "models", dataset_name)

            LOGGER.info("Loading model(s) from: %s", load_dir)

            loaded_list = []
            for _ in PatchCore_list:
                pc = patchcore.patchcore.PatchCore(device)
                pc.load_from_path(load_dir, device=device, nn_method=nn_method)
                loaded_list.append(pc)
            PatchCore_list = loaded_list

            base_test_ds = test_loader.dataset
            n_test = len(base_test_ds)

            start = bench_image_index
            end = bench_image_index + bench_iters
            if end > n_test:
                raise click.ClickException(
                    f"Not enough test images. Need {bench_iters} from index {bench_image_index}, "
                    f"but test size is {n_test}."
                )

            warmup_indices = list(range(start, start + bench_warmup))
            meas_indices = list(range(start + bench_warmup, start + bench_iters))

            warmup_loader = _subset_dataloader(
                base_test_ds,
                warmup_indices,
                batch_size=1,
                num_workers=bench_num_workers,
                pin_memory=True,
            )
            meas_loader = _subset_dataloader(
                base_test_ds,
                meas_indices,
                batch_size=1,
                num_workers=bench_num_workers,
                pin_memory=True,
            )

            LOGGER.info(
                "[BENCH] Start: iters=%d warmup=%d image_index=%d test_size=%d ensemble=%d",
                bench_iters,
                bench_warmup,
                bench_image_index,
                n_test,
                len(PatchCore_list),
            )

            if bench_warmup > 0:
                _sync_if_cuda(device)
                t0 = time.perf_counter()
                for pc in PatchCore_list:
                    pc.predict(warmup_loader)
                _sync_if_cuda(device)
                t1 = time.perf_counter()
                warmup_ms = (t1 - t0) * 1000.0 / float(bench_warmup)
                LOGGER.info("[BENCH] warmup mean: %.3f ms / image (n=%d)", warmup_ms, bench_warmup)

            _sync_if_cuda(device)
            t0 = time.perf_counter()
            for pc in PatchCore_list:
                pc.predict(meas_loader)
            _sync_if_cuda(device)
            t1 = time.perf_counter()

            elapsed = (t1 - t0)
            n = len(meas_indices)
            mean_ms = (elapsed * 1000.0) / float(n)
            fps = float(n) / elapsed if elapsed > 0 else float("inf")

            LOGGER.info("[BENCH] mean: %.3f ms / image", mean_ms)
            LOGGER.info("[BENCH] FPS : %.2f", fps)
            LOGGER.info("[BENCH] Done.")
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
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name = backbone_name

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
                nn_method=None,
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

            train_dataloader.name = name + (f"_{subdataset}" if subdataset is not None else "")
            test_dataloader.name = train_dataloader.name

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
                val_dataloader.name = train_dataloader.name
            else:
                val_dataloader = None

            dataloaders.append(
                {"training": train_dataloader, "validation": val_dataloader, "testing": test_dataloader}
            )
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: %s", " ".join(sys.argv))
    main()


