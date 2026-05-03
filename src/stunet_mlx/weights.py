"""STU-Net weight loading for MLX."""

import os
import pickle
from typing import List, Optional

import mlx.core as mx
import numpy as np

from .model import STUNet


# Variant configs (channel widths and per-stage block depth).
VARIANT_CONFIGS = {
    "small": dict(depth=[1] * 6, dims=[16, 32, 64, 128, 256, 256]),
    "base":  dict(depth=[1] * 6, dims=[32, 64, 128, 256, 512, 512]),
    "large": dict(depth=[2] * 6, dims=[64, 128, 256, 512, 1024, 1024]),
    "huge":  dict(depth=[3] * 6, dims=[96, 192, 384, 768, 1536, 1536]),
}

# Plan published with the TotalSegmentator-pretrained STU-Net checkpoints.
# The last pooling stage is anisotropic ([1,1,2]); the rest are [2,2,2].
# All four size variants share these plans (verified from STU-Net/plan_files/*.pkl).
PRETRAINED_POOL_OP_KERNEL_SIZES: List[List[int]] = (
    [[2, 2, 2]] * 4 + [[1, 1, 2]]
)
PRETRAINED_CONV_KERNEL_SIZES: List[List[int]] = [[3, 3, 3]] * 6


def _read_plans(plan_pkl_path: str):
    """Read pool/conv kernel sizes from an nnU-Net plan pkl."""
    with open(plan_pkl_path, "rb") as f:
        d = pickle.load(f)
    plans = d["plans"]
    stage = plans["plans_per_stage"][0]
    return (
        [list(map(int, k)) for k in stage["pool_op_kernel_sizes"]],
        [list(map(int, k)) for k in stage["conv_kernel_sizes"]],
    )


def load_stunet(
    checkpoint_path: str,
    variant: str = "small",
    dtype: str = "float32",
    plan_pkl: Optional[str] = None,
    pool_op_kernel_sizes: Optional[List[List[int]]] = None,
    conv_kernel_sizes: Optional[List[List[int]]] = None,
) -> STUNet:
    """Load STU-Net from a PyTorch checkpoint.

    Args:
        checkpoint_path: Path to .model file.
        variant: "small", "base", "large", or "huge".
        dtype: "float32" or "float16".
        plan_pkl: Optional path to the matching .model.pkl plan file. If a
            file exists at "{checkpoint_path}.pkl" it is auto-detected.
        pool_op_kernel_sizes: Override; per-stage pool kernel sizes.
        conv_kernel_sizes: Override; per-stage conv kernel sizes.

    Plan resolution order: explicit kwargs → plan_pkl → "{checkpoint}.pkl"
    sibling → published pretrained defaults (anisotropic last stage).
    """
    import torch

    if variant not in VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown variant: {variant}. Available: {list(VARIANT_CONFIGS)}"
        )
    cfg = VARIANT_CONFIGS[variant]

    if pool_op_kernel_sizes is None or conv_kernel_sizes is None:
        sibling_pkl = checkpoint_path + ".pkl"
        candidate = plan_pkl or (sibling_pkl if os.path.exists(sibling_pkl) else None)
        if candidate is not None:
            pool_from_pkl, conv_from_pkl = _read_plans(candidate)
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = pool_from_pkl
            if conv_kernel_sizes is None:
                conv_kernel_sizes = conv_from_pkl
        else:
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = PRETRAINED_POOL_OP_KERNEL_SIZES
            if conv_kernel_sizes is None:
                conv_kernel_sizes = PRETRAINED_CONV_KERNEL_SIZES

    model = STUNet(
        input_channels=1, num_classes=105,
        depth=cfg["depth"], dims=cfg["dims"],
        pool_op_kernel_sizes=pool_op_kernel_sizes,
        conv_kernel_sizes=conv_kernel_sizes,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    target_dtype = mx.float16 if dtype == "float16" else mx.float32

    mlx_weights = {}
    for key, tensor in state_dict.items():
        arr = tensor.cpu().numpy()
        # Transpose Conv3d weights: PyTorch (out, in, D, H, W) → MLX (out, D, H, W, in)
        if arr.ndim == 5:
            arr = arr.transpose(0, 2, 3, 4, 1)
        mlx_weights[key] = mx.array(arr.astype(np.float32)).astype(target_dtype)

    model.load_weights(list(mlx_weights.items()), strict=True)
    return model


def download_and_load(variant: str = "small", dtype: str = "float32") -> STUNet:
    """Download STU-Net weights from HuggingFace and load.

    The HF repo only ships the .model file (no plan pkl), so the published
    pretrained plan (anisotropic last pool stage) is used.
    """
    from huggingface_hub import hf_hub_download

    filenames = {
        "small": "small_ep4k.model",
        "base": "base_ep4k.model",
        "large": "large_ep4k.model",
        "huge": "huge_ep4k.model",
    }
    path = hf_hub_download("ziyanhuang/STU-Net", filenames[variant])
    return load_stunet(path, variant=variant, dtype=dtype)
