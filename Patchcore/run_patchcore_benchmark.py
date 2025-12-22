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


def cuda_sync(device: torch.device):
    if device.type.lower().startswith("cuda"):
        torch.cuda.synchronize(device=device)


def _glob_latest_model_dir(results_path: str, log_project: str, log_group: str, dataset_name: str) -> str:
    """
    Find latest:
      {results_path}/{log_project}/{log_group}_*/models/{dataset_name}/
    (dataset_name is usually "mvtec_<subdataset>")
    """
    base = Path(results_path) / log_project
    if not base.exists():
        raise FileNotFoundError(f"results base not found: {base}")

    candidates = []
    for p in base.glob(f"{log_group}*/models/{dataset_name}"):
        if (p / "patchcore_params.pkl").exists():
            candidates.append(p)

        # ensemble case: any *_patchcore_params.pkl
        if list(p.glob("*patchcore_params.pkl")):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            "No model dir found.\n"
            f"searched: {base}/{log_group}*/models/{dataset_name}/(patchcore_params.pkl)\n"
            "Hint: check your saved model path and pass --model_dir explicitly."
        )

    # pick newest by mtime of patchcore_params.pkl (or newest pkl)
    def mtime_key(d: Path):
        pkls = list(d.glob("*patchcore_params.pkl"))
        if not pkls:
            pkls = [d / "patchcore_params.pkl"]
        return max(x.stat().st_mtime for x in pkls if x.exists())

    latest = max(candidates, key=mtime_key)
    return str(latest)


def _load_patchcores_from_dir(model_dir: str, device: torch.device, nn_method):
    """
    Supports:
      - single: patchcore_params.pkl
      - ensemble: *patchcore_params.pkl
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    pkls = sorted(model_dir.glob("*patchcore_params.pkl"))
    if not pkls:
        # fallback exact name
        p = model_dir / "patchcore_params.pkl"
        if p.exists():
            pkls = [p]

    if not pkls:
        raise FileNotFoundError(f"No *patchcore_params.pkl found in: {model_dir}")

    pcs = []
    for pkl in pkls:
        pc = patchcore.patchcore.PatchCore(device)
        # load_from_path expects directory path
        pc.load_from_path(str(model_dir), device=device, nn_method=nn_method)
        pcs.append(pc)
        break  # IMPORTANT: repo実装によってはdir単位でロードするので、まず1個だけにする（必要なら後で拡張）

    return pcs


def _benchmark_105_images(PatchCore, test_dataset, device: torch.device):
    n = len(test_dataset)
    if n < 105:
        raise RuntimeError(f"test dataset has only {n} images (<105).")

    warm_idx = list(range(0, 5))
    test_idx = list(range(5, 105))

    warm_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_dataset, warm_idx),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_dataset, test_idx),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    PatchCore.model.eval() if hasattr(PatchCore, "model") else None

    with torch.inference_mode():
        cuda_sync(device)
        _ = PatchCore.predict(warm_loader)
        cuda_sync(device)

        cuda_sync(device)
        t0 = time.perf_counter()
        _ = PatchCore.predict(test_loader)
        cuda_sync(device)
        t1 = time.perf_counter()

    total = t1 - t0
    per_img = total / 100.0
    fps = 100.0 / total if total > 0 else float("inf")

    return {"total_s": total, "sec_per_img": per_img, "fps": fps}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group", show_default=True)
@click.option("--log_project", type=str, default="project", show_default=True)
@click.option("--model_dir", type=str, default="", show_default=True)
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8, show_default=True)
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
):
    methods = {k: v for (k, v) in methods}

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device(f"cuda:{device.index}")
        if device.type.lower().startswith("cuda")
        else contextlib.suppress()
    )

    patchcore.utils.fix_seeds(seed, device)

    dataloaders_list = methods["get_dataloaders"](seed)

    with device_context:
        torch.cuda.empty_cache()

        # FAISS NN
        nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

        for dataloaders in dataloaders_list:
            # dataset_name must match your training save folder name: "mvtec_<subdataset>"
            dataset_name = getattr(dataloaders["training"], "name", None)
            if dataset_name is None:
                # fallback: build like original
                ds = dataloaders["training"].dataset
                classname = getattr(ds, "classname", None) or getattr(ds, "class_name", None)
                if classname is None:
                    raise RuntimeError("Cannot infer dataset name. Please ensure train_loader.name is set.")
                dataset_name = f"mvtec_{classname}"

            # locate model dir
            if model_dir:
                use_model_dir = model_dir
            else:
                use_model_dir = _glob_latest_model_dir(results_path, log_project, log_group, dataset_name)

            LOGGER.info("Loading model from: %s", use_model_dir)

            pcs = _load_patchcores_from_dir(use_model_dir, device=device, nn_method=nn_method)
            PatchCore = pcs[0]

            test_dataset = dataloaders["testing"].dataset

            LOGGER.info("[BENCH] 105 images -> drop first 5 -> measure 100 (batch=1)")
            r = _benchmark_105_images(PatchCore, test_dataset, device)

            LOGGER.info("===== RESULT =====")
            LOGGER.info("sec/image : %.6f", r["sec_per_img"])
            LOGGER.info("FPS       : %.3f", r["fps"])
            LOGGER.info("total(100): %.6f s", r["total_s"])
            LOGGER.info("==================\n")


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
        layers_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_coll[idx].append(layer)
    else:
        layers_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, device):
        pcs = []
        for backbone_name, layers in zip(backbone_names, layers_coll):
            backbone = patchcore.backbones.load(backbone_name)
            nn_method = patchcore.common.FaissNN(False, 8)

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

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            # keep compatibility with your original run_patchcore.py
            train_loader.name = name + (f"_{subdataset}" if subdataset is not None else "")
            test_loader.name = train_loader.name

            dataloaders.append({"training": train_loader, "testing": test_loader, "validation": None})
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line: %s", " ".join(sys.argv))
    main()

