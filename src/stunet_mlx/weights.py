"""STU-Net weight loading for MLX."""

import mlx.core as mx
import numpy as np

from .model import STUNet


def load_stunet(checkpoint_path: str, variant: str = "small",
                dtype: str = "float32") -> STUNet:
    """Load STU-Net from a PyTorch checkpoint.

    Args:
        checkpoint_path: Path to .model file
        variant: "small", "base", "large", or "huge"
        dtype: "float32" or "float16"
    """
    import torch

    # Variant configs
    configs = {
        "small": dict(depth=[1]*6, dims=[16, 32, 64, 128, 256, 256]),
        "base":  dict(depth=[1]*6, dims=[32, 64, 128, 256, 512, 512]),
        "large": dict(depth=[2]*6, dims=[64, 128, 256, 512, 1024, 1024]),
        "huge":  dict(depth=[3]*6, dims=[96, 192, 384, 768, 1536, 1536]),
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(configs.keys())}")

    cfg = configs[variant]
    model = STUNet(
        input_channels=1, num_classes=105,
        depth=cfg["depth"], dims=cfg["dims"],
    )

    # Load checkpoint
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
    """Download STU-Net weights from HuggingFace and load."""
    from huggingface_hub import hf_hub_download

    filenames = {
        "small": "small_ep4k.model",
        "base": "base_ep4k.model",
        "large": "large_ep4k.model",
        "huge": "huge_ep4k.model",
    }
    path = hf_hub_download("ziyanhuang/STU-Net", filenames[variant])
    return load_stunet(path, variant=variant, dtype=dtype)
