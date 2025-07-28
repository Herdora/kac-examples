# sample_profiler.py
import argparse
import os
import time
import uuid
from pathlib import Path
import torch

# Constants (pulled from args)
MATRIX_SIZE = 4096
ITERS_PER_STEP = 5


def main():
    # MANDATORY: Get trace_dir from environment variable (being passed from backend)
    trace_dir = os.getenv("TRACE_DIR", "traces")

    steps = 5

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_id = uuid.uuid4().hex[:8]
    out_dir = Path(trace_dir).expanduser().resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    size = MATRIX_SIZE
    a = torch.randn(size, size, device=device, dtype=torch.float16)
    b = torch.randn(size, size, device=device, dtype=torch.float16)

    def workload(n_iters: int):
        print("TESTING TESTING LOGS")
        with torch.no_grad():
            c = a
            for _ in range(n_iters):
                c = c @ b
                c = torch.relu(c)
            if cuda_available:
                torch.cuda.synchronize()
            else:
                _ = c.sum()

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=steps - 2)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda_available:
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
            workload(ITERS_PER_STEP)
            prof.step()
            time.sleep(0.05)

    trace_files = list(out_dir.glob("*.pt.trace.json"))
    print(f"Generated {len(trace_files)} trace files in: {out_dir}")


if __name__ == "__main__":
    main()
