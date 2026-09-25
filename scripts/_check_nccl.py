"""Does multi-GPU NCCL work on this node at all?

The 122B teacher needs TP=4 and hung at NCCL init for 14 minutes. Every run so far
has been single-GPU, so multi-GPU NCCL is completely untested here. This isolates
it: a few MB and a couple of collectives, with a hard timeout so a hang is a fast
failure instead of a silent one.

    CUDA_VISIBLE_DEVICES=5,6,7 uv run python scripts/_check_nccl.py
"""

import datetime
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world: int, result):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29555",
            rank=rank,
            world_size=world,
            # Without this a broken fabric hangs forever rather than raising.
            timeout=datetime.timedelta(seconds=90),
        )
        x = torch.ones(1024, 1024, device=f"cuda:{rank}") * (rank + 1)
        dist.all_reduce(x)
        expected = sum(range(1, world + 1))
        ok = abs(x[0, 0].item() - expected) < 1e-3

        y = torch.zeros(world, device=f"cuda:{rank}")
        y[rank] = rank
        dist.all_gather_into_tensor(torch.empty(world, device=f"cuda:{rank}"), y[rank : rank + 1])

        dist.barrier()
        result[rank] = 1 if ok else 0
        dist.destroy_process_group()
    except Exception as error:
        print(f"[rank {rank}] FAILED: {type(error).__name__}: {str(error)[:200]}", flush=True)
        result[rank] = 0


if __name__ == "__main__":
    world = torch.cuda.device_count()
    print(f"visible GPUs: {world} ({os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})")
    if world < 2:
        print("need at least 2 GPUs")
        sys.exit(1)
    for key in ("NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_SOCKET_IFNAME"):
        print(f"  {key}={os.environ.get(key, 'unset')}")

    # Must be a spawn-context Array: mp.spawn launches spawn-context children, and
    # a fork-context SemLock cannot be shared across that boundary.
    result = mp.get_context("spawn").Array("i", world)
    mp.spawn(worker, args=(world, result), nprocs=world, join=True)
    good = sum(result[:])
    print(f"\n{good}/{world} ranks completed all-reduce + all-gather + barrier")
    print("NCCL multi-GPU OK" if good == world else "NCCL multi-GPU BROKEN")
    sys.exit(0 if good == world else 1)
