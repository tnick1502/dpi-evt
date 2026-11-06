"""Device selection for training and evaluation: CUDA, MPS, then CPU.

The requested argument and LIQ_DEVICE override automatic selection. MPS sets
PYTORCH_ENABLE_MPS_FALLBACK=1. A kernel probe checks the selected accelerator;
failure selects CPU.
"""
from __future__ import annotations

import os

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    """Select a torch device from requested, LIQ_DEVICE or CUDA/MPS/CPU availability.

    :param requested: device name, such as 'cuda', 'mps', 'cpu' or 'cuda:1'; None
        uses LIQ_DEVICE or automatic selection
    :return: torch.device; MPS also enables PYTORCH_ENABLE_MPS_FALLBACK
    """
    req = requested or os.environ.get("LIQ_DEVICE")
    if req:
        device = torch.device(req)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Probe the accelerator before training, because an available device may still be
    # incompatible with the installed PyTorch build. Use CPU if the probe fails.
    if device.type in ("cuda", "mps") and not _device_can_run_kernels(device):
        print(f"[accel] device {device} failed the kernel check; using CPU. "
              f"Check the PyTorch build for this GPU or set LIQ_DEVICE=cpu.")
        device = torch.device("cpu")
    return device


def _device_can_run_kernels(device: torch.device) -> bool:
    """Check kernel execution and synchronization on the device."""
    try:
        x = torch.ones(8, device=device)
        _ = float((x + 1.0).sum().item())  # Force kernel execution and synchronization.
        return True
    except Exception:
        return False


def describe_device(device: torch.device | None = None) -> str:
    """Return a readable device description, including the CUDA GPU name."""
    device = device or resolve_device()
    if device.type == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(device)})"
        except Exception:
            return "cuda"
    if device.type == "mps":
        return "mps (Apple Metal)"
    return "cpu"


def configure_performance(device: torch.device | None = None, cpu_threads: int | None = None) -> torch.device:
    """Configure computation and return the selected device.

    CUDA enables TF32 and float32_matmul_precision='high'. MPS enables CPU fallback;
    CPU applies the requested thread count.

    :param device: target device; None calls resolve_device
    :param cpu_threads: CPU thread count, applied only on CPU
    :return: the selected device
    """
    device = device or resolve_device()
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    elif device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    elif device.type == "cpu" and cpu_threads:
        torch.set_num_threads(int(cpu_threads))
    return device
