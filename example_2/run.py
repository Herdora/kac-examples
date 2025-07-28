import os
import time
import uuid
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

# Constants
MATRIX_SIZE = 2048
BATCH_SIZE = 64
HIDDEN_SIZE = 1024


def main():
    # MANDATORY: Get trace_dir from environment variable (being passed from backend)
    trace_dir = os.getenv("TRACE_DIR", "traces")
    steps = 5

    # Multi-GPU setup - count available GPUs
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        # Also check if CUDA_VISIBLE_DEVICES is set to limit GPUs
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            gpu_count = len(visible_devices.split(","))
        print(f"Found {gpu_count} GPUs available")
    else:
        gpu_count = 0
        print("No CUDA GPUs available, falling back to CPU")

    if gpu_count > 1:
        print(f"Using multi-GPU setup with {gpu_count} GPUs")
        # Use spawn method for multi-GPU
        mp.spawn(
            run_multi_gpu_profiling, args=(gpu_count, trace_dir, steps), nprocs=gpu_count, join=True
        )
    else:
        print("Using single GPU/CPU setup")
        run_single_gpu_profiling(0, trace_dir, steps)


def run_multi_gpu_profiling(rank, world_size, trace_dir, steps):
    """Run profiling on multiple GPUs using DDP"""
    try:
        # Initialize distributed training
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        
        # Use gloo backend as fallback if nccl fails in container
        try:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        except Exception as e:
            print(f"NCCL failed, falling back to gloo: {e}")
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    run_id = uuid.uuid4().hex[:8]
    out_dir = Path(trace_dir).expanduser().resolve() / f"{run_id}_gpu_{rank}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple model for multi-GPU workload
    model = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    ).to(device)

    # Wrap model with DDP
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Create input data
    input_data = torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    def workload():
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(input_data)
        target = torch.randn_like(output)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Synchronize across GPUs
        torch.cuda.synchronize()

    # Profiler setup
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=steps - 2)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for s in range(steps):
            workload()
            prof.step()
            time.sleep(0.05)

    trace_files = list(out_dir.glob("*.pt.trace.json"))
    print(f"GPU {rank}: Generated {len(trace_files)} trace files in: {out_dir}")

    # Clean up distributed training
    dist.destroy_process_group()


def run_single_gpu_profiling(device_id, trace_dir, steps):
    """Run profiling on single GPU or CPU"""
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    run_id = uuid.uuid4().hex[:8]
    out_dir = Path(trace_dir).expanduser().resolve() / f"{run_id}_single"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    ).to(device)

    input_data = torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    def workload():
        model.train()
        optimizer.zero_grad()

        output = model(input_data)
        target = torch.randn_like(output)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Profiler setup
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=steps - 2)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for s in range(steps):
            workload()
            prof.step()
            time.sleep(0.05)

    trace_files = list(out_dir.glob("*.pt.trace.json"))
    print(f"Generated {len(trace_files)} trace files in: {out_dir}")


if __name__ == "__main__":
    main()
