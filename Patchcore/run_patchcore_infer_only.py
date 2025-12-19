# bin/run_patchcore_infer_only.py
import time
import torch
import numpy as np
import patchcore.patchcore
import patchcore.utils

def benchmark_inference(
    patchcore_model,
    dataloader,
    warmup=5,
    runs=100,
):
    times = []

    with torch.no_grad():
        for i in range(warmup + runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            patchcore_model.predict(dataloader)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            if i >= warmup:
                times.append(t1 - t0)

    times = np.array(times)
    print("Inference benchmark")
    print(f"  runs        : {runs}")
    print(f"  mean [ms]   : {times.mean()*1000:.3f}")
    print(f"  std  [ms]   : {times.std()*1000:.3f}")
    print(f"  min  [ms]   : {times.min()*1000:.3f}")
    print(f"  max  [ms]   : {times.max()*1000:.3f}")
