"""
gpu_burn.py -- keep a GPU busy for N seconds so you can demonstrate
nvidia-smi. Runs large matrix multiplications in a loop on the GPU,
no DataLoader, no training, no dataset -- self-contained.

Usage:
    python gpu_burn.py                  # default: 60 seconds, GPU 0
    python gpu_burn.py --seconds 120
    python gpu_burn.py --size 8192      # bigger matmul = higher power draw
    python gpu_burn.py --device cuda:1  # pick a specific GPU on multi-GPU node
"""

import argparse
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="how long to keep the GPU busy (default: 60s)")
    parser.add_argument("--size", type=int, default=4096,
                        help="matmul size NxN (default: 4096; try 8192 for more power)")
    parser.add_argument("--device", default="cuda:0",
                        help="device to run on (default: cuda:0)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available -- nothing to demo")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # Allocate two large matrices once; the loop just multiplies them.
    # bf16 keeps Tensor Cores busy on Ampere+/Hopper; switch to float32
    # if you want to demonstrate a non-Tensor-Core baseline.
    a = torch.randn(args.size, args.size, device=device, dtype=torch.bfloat16)
    b = torch.randn(args.size, args.size, device=device, dtype=torch.bfloat16)

    print(f"Running {args.size}x{args.size} matmuls on {device} for {args.seconds}s")
    print("Watch with:  watch -n 0.5 nvidia-smi")

    start = time.perf_counter()
    iterations = 0
    while time.perf_counter() - start < args.seconds:
        c = a @ b
        # Use the result so the compiler can't optimise it away
        a = c * 0.9999 + a * 0.0001
        iterations += 1

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Done: {iterations} iterations in {elapsed:.2f}s "
          f"({iterations/elapsed:.1f} matmuls/sec)")


if __name__ == "__main__":
    main()
