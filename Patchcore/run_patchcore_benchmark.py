import contextlib
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

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


def _is_cuda(device: torch.device) -> bool:
    return device is not None and "cuda" in device.type.lower()


def _sync_if_cuda(device: torch.device) -> None:
    if _is_cuda(device):
        torch.cuda.synchronize(device=device)


def _move_tensors_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_move_tensors_to_device(v, device) for v in obj]
        return type(obj)(moved)
    return obj


class _PreloadedDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Any]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Any:
        return self.samples[idx]


def _make_preloaded_loader(
    original_dataset: torch.utils.data.Dataset,
    indices: List[int],
    device: torch.device,
    exclude_h2d: bool,
    batch_size: int = 1,
) -> torch.utils.data.DataLoader:
    samples: List[Any] = []
    for i in indices:
        sample = original_dataset[i]
        if exclude_h2d and _is_cuda(device):
            sample = _move_tensors_to_device(sample, device)
        samples.append(sample)

    ds = _PreloadedDataset(samples)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    loader.name = getattr(original_dataset, "name", "dataset")
    return loader


def _subset_loader_from_dataset(
    original_dataset: torch.utils.data.Dataset,
    indices: List[int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> torch.utils.data.DataLoader:
    subset = torch.utils.data.Subset(original_dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    loader.name = getattr(original_dataset, "name", "dataset")
    return loader


def _load_patchcore_from_dir(patchcore_instance, model_dir: str) -> None:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"--model_dir must be a directory: {model_dir}")

    if hasattr(patchcore_instance, "load_from_path"):
        patchcore_instance.load_from_path(model_dir)
        return

    if hasattr(patchcore_instance, "load_from_checkpoint"):
        patchcore_instance.load_from_checkpoint(model_dir)
        return

    if hasattr(patchcore_instance, "load_from_path"):
        patchcore_instance.load_from_path(model_dir)
        return

    raise RuntimeError(
        "PatchCore instance has no supported load method. "
        "Expected load_from_path() or load_from_checkpoint()."
    )


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--model_dir", type=str, default=None, help="Directory containing saved PatchCore model(s).")
@click.option("--bench_only", is_flag=True, help="Benchmark only (no training, no metrics, no outputs).")
@click.option("--bench_n", type=int, default=105, show_default=True)
@click.option("--bench_warmup", type=int, default=5, show_default=True)
@click.option("--bench_image_index", type=int, default=0, show_default=True)
@click.option("--bench_batch_size", type=int, default=1, show_default=True)
@click.option("--bench_num_workers", type=int, default=0, show_default=True)
@click.option("--bench_exclude_io", is_flag=True, help="Exclude disk I/O by preloading samples before timing.")
@click.option("--bench_exclude_h2d", is_flag=True, help="Exclude CPU->GPU transfer by moving tensors to GPU before timing.")
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods: List[Tuple[str, Any]],
    results_path: str,
    gpu: Tuple[int, ...],
    seed: int,
    log_group: str,
    log_project: str,
    model_dir: str,
    bench_only: bool,
    bench_n: int,
    bench_warmup: int,
    bench_image_index: int,
    bench_batch_size: int,
    bench_num_workers: int,
    bench_exclude_io: bool,
    bench_exclude_h2d: bool,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )
    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device(f"cuda:{device.index}")
        if _is_cuda(device)
        else contextlib.suppress()
    )

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        dataset_name = dataloaders["training"].name
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataset_name,
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

            if len(PatchCore_list) > 1:
                LOGGER.info("Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list)))

            if bench_only:
                if not model_dir:
                    raise click.ClickException("--bench_only requires --model_dir")

                test_loader = dataloaders["testing"]
                base_ds = test_loader.dataset

                total_needed = int(bench_n)
                warm = int(bench_warmup)
                meas = max(0, total_needed - warm)

                start = int(bench_image_index)
                warm_indices = list(range(start, min(start + warm, len(base_ds))))
                meas_indices = list(range(start + warm, min(start + warm + meas, len(base_ds))))

                warm_count = len(warm_indices)
                meas_count = len(meas_indices)

                if warm_count < warm or meas_count < meas:
                    LOGGER.warning(
                        "Requested bench_n=%d warmup=%d starting_at=%d, but dataset size=%d. Using warm=%d meas=%d.",
                        total_needed, warm, start, len(base_ds), warm_count, meas_count
                    )

                pin_memory = True

                if bench_exclude_io:
                    warm_loader = _make_preloaded_loader(
                        base_ds, warm_indices, device, exclude_h2d=bench_exclude_h2d, batch_size=bench_batch_size
                    )
                    meas_loader = _make_preloaded_loader(
                        base_ds, meas_indices, device, exclude_h2d=bench_exclude_h2d, batch_size=bench_batch_size
                    )
                else:
                    warm_loader = _subset_loader_from_dataset(
                        base_ds, warm_indices, batch_size=bench_batch_size, num_workers=bench_num_workers, pin_memory=pin_memory
                    )
                    meas_loader = _subset_loader_from_dataset(
                        base_ds, meas_indices, batch_size=bench_batch_size, num_workers=bench_num_workers, pin_memory=pin_memory
                    )

                torch.set_grad_enabled(False)

                for i, PatchCore in enumerate(PatchCore_list):
                    torch.cuda.empty_cache()

                    model_subdir = model_dir
                    if len(PatchCore_list) > 1:
                        candidates = [
                            os.path.join(model_dir, f"Ensemble-{i+1}-{len(PatchCore_list)}_"),
                            os.path.join(model_dir, f"Ensemble-{i+1}-{len(PatchCore_list)}"),
                        ]
                        model_subdir = model_dir
                        for c in candidates:
                            if os.path.exists(c):
                                model_subdir = c
                                break

                    _load_patchcore_from_dir(PatchCore, model_subdir)

                    LOGGER.info("Warmup ({}/{})".format(i + 1, len(PatchCore_list)))
                    _sync_if_cuda(device)
                    PatchCore.predict(warm_loader)
                    _sync_if_cuda(device)

                times = []
                for i, PatchCore in enumerate(PatchCore_list):
                    LOGGER.info("Measured ({}/{})".format(i + 1, len(PatchCore_list)))
                    _sync_if_cuda(device)
                    t0 = time.perf_counter()
                    PatchCore.predict(meas_loader)
                    _sync_if_cuda(device)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

                elapsed = float(np.mean(times)) if times else float("nan")
                fps = (float(meas_count) / elapsed) if elapsed > 0 else float("nan")

                LOGGER.info("=== PatchCore Benchmark Result ===")
                LOGGER.info("dataset=%s", dataset_name)
                LOGGER.info("bench_image_index=%d", start)
                LOGGER.info("bench_total=%d warmup=%d measured=%d", warm_count + meas_count, warm_count, meas_count)
                LOGGER.info("exclude_io=%s exclude_h2d=%s", str(bench_exclude_io), str(bench_exclude_h2d))
                LOGGER.info("elapsed_sec_mean_over_models=%.6f", elapsed)
                LOGGER.info("fps=%.3f", fps)

                continue

            raise click.ClickException("This script is benchmark-only in this version. Use --bench_only.")


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

            dataloader_dict = {
                "training": train_dataloader,
                "validation": None,
                "testing": test_dataloader,
            }
            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()


