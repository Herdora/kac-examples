import argparse
import os
import time
import uuid
from pathlib import Path
import torch

# RIGHT NOW, ARGS TO FILES ARE NOT SUPPORTED.
CONSTANT_1 = 1000
CONSTANT_2 = "default_value"


def main():
    # MANDATORY: Get trace_dir from environment variable (being passed from backend)
    trace_dir = os.getenv("TRACE_DIR", "traces")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup output directory
    run_id = uuid.uuid4().hex[:8]
    out_dir = Path(trace_dir).expanduser().resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = 5

    # ========================================
    # WORKLOAD SETUP
    # ========================================
    # 1. Create your model/tensors here
    # Example:
    # model = YourModel().to(device)
    # input_data = torch.randn(batch_size, input_size).to(device)
    # target = torch.randn(batch_size, output_size).to(device)

    # 2. Define your workload function
    def workload():
        # This function should contain the operations you want to profile
        # Example:
        # with torch.no_grad():
        #     output = model(input_data)
        #     loss = criterion(output, target)
        #     loss.backward()
        #     optimizer.step()
        pass

    # ========================================
    # PROFILER SETUP
    # ========================================
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=steps - 2)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # ========================================
    # MAIN PROFILING LOOP. MANDATORY.
    # ========================================
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for s in range(steps):
            workload()  # Execute your workload
            prof.step()
            time.sleep(0.05)  # Small gap for visualization

    # Final output
    trace_files = list(out_dir.glob("*.pt.trace.json"))
    print(f"Generated {len(trace_files)} trace files in: {out_dir}")


if __name__ == "__main__":
    main()
