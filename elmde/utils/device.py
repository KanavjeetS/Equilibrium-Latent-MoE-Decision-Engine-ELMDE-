"""
Hardware and Device Management Utilities for ELMDE.
Supports automatic device detection (CUDA, MPS, CPU), mixed precision selection,
and torch.compile optimization.
"""

from typing import Tuple
import torch


def get_optimal_device() -> torch.device:
    """Returns the best available torch.device."""
    device, _ = get_optimal_device_and_dtype()
    return device


def get_optimal_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    """
    Selects the best available hardware accelerator and floating point precision:
    - CUDA with bfloat16 (Ampere+) or float16
    - MPS (Apple Silicon) with float32/float16
    - CPU with float32 or bfloat16
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Check if bfloat16 is supported (Compute capability >= 8.0)
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    return device, dtype


def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Applies inference optimizations (eval mode, freeze weights, optional torch.compile)."""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Check if torch.compile is available and supported on this platform
    if hasattr(torch, "compile") and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass  # Fallback to standard execution if triton/CUDA tools not present
    return model
