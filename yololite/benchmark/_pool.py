"""GPU-aware process pool utilities.

Torch is intentionally NOT imported at module level.  Spawned worker processes
re-import this module, and any module-level torch import would initialize CUDA
before the pool initializer can set CUDA_VISIBLE_DEVICES.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

SPAWN_CTX = multiprocessing.get_context("spawn")
NUM_CPUS = os.cpu_count() or 120


def get_num_gpus() -> int:
    """Return the number of available CUDA GPUs (imports torch lazily)."""
    import torch
    return torch.cuda.device_count()


def cap_threads(max_concurrent: int) -> None:
    """Cap PyTorch intra-op thread pool to avoid oversubscription."""
    import torch
    torch.set_num_threads(max(1, NUM_CPUS // max_concurrent))


def pool_initializer(gpu_queue: multiprocessing.Queue, max_concurrent: int) -> None:
    """Grab a GPU ID from the queue, restrict CUDA_VISIBLE_DEVICES, cap threads.

    Called once per worker process before any task runs.
    """
    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    threads = str(max(1, NUM_CPUS // max_concurrent))
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads


def run_variant_pool(
    jobs: list[tuple],
    worker_fn: Callable[[tuple], dict],
    num_gpus: int,
    jobs_per_gpu: int,
    on_result: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Run jobs through a GPU-aware process pool.

    Creates a ProcessPoolExecutor with ``num_gpus * jobs_per_gpu`` workers.
    Each worker is assigned one GPU via a shared queue.  Results are collected
    as they complete; *on_result* is called for each (e.g. for incremental CSV
    saves).

    Returns all result dicts.
    """
    max_concurrent = num_gpus * jobs_per_gpu

    gpu_queue = SPAWN_CTX.Queue()
    for _ in range(jobs_per_gpu):
        for gpu_id in range(num_gpus):
            gpu_queue.put(gpu_id)

    results: list[dict] = []
    pool = ProcessPoolExecutor(
        max_workers=max_concurrent,
        mp_context=SPAWN_CTX,
        initializer=pool_initializer,
        initargs=(gpu_queue, max_concurrent),
    )
    try:
        futures = {pool.submit(worker_fn, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if on_result is not None:
                on_result(result)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return results
