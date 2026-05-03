"""Equivalence test: MLX STUNet vs. original PyTorch STUNet.

Builds both with the published pretrained plan (anisotropic last pool stage),
copies the random PyTorch weights into MLX, runs the same input through both
and asserts numeric agreement. Also checks the deep-supervision output order
matches PyTorch (finest first, coarser in reverse).
"""

import os
import sys
import importlib.util

import numpy as np
import pytest

import mlx.core as mx

from stunet_mlx.model import STUNet as MLXSTUNet
from stunet_mlx.weights import (
    PRETRAINED_POOL_OP_KERNEL_SIZES,
    PRETRAINED_CONV_KERNEL_SIZES,
)


REF_PY_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "STU-Net", "nnUNet-1.7.1", "nnunet", "network_architecture", "STUNet.py",
))


def _load_reference():
    """Import the original STU-Net PyTorch module without its nnunet package.

    The reference file imports from `nnunet.network_architecture.{...}` which
    we don't want to install. Stub those out before importing.
    """
    import types
    import torch.nn as tnn

    # Stub the nnunet package tree the reference file imports from.
    pkg = types.ModuleType("nnunet")
    pkg.__path__ = []
    sub_arch = types.ModuleType("nnunet.network_architecture")
    sub_arch.__path__ = []
    sub_init = types.ModuleType("nnunet.network_architecture.initialization")
    sub_init.InitWeights_He = lambda neg_slope=1e-2: (lambda m: None)
    sub_nn = types.ModuleType("nnunet.network_architecture.neural_network")
    sub_nn.SegmentationNetwork = tnn.Module
    sys.modules.setdefault("nnunet", pkg)
    sys.modules.setdefault("nnunet.network_architecture", sub_arch)
    sys.modules["nnunet.network_architecture.initialization"] = sub_init
    sys.modules["nnunet.network_architecture.neural_network"] = sub_nn

    spec = importlib.util.spec_from_file_location("ref_stunet", REF_PY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _torch_state_to_mlx(state_dict):
    """Convert a PyTorch STU-Net state_dict into an MLX-loadable list."""
    out = {}
    for k, v in state_dict.items():
        arr = v.detach().cpu().numpy()
        if arr.ndim == 5:  # Conv3d: (out, in, D, H, W) → (out, D, H, W, in)
            arr = arr.transpose(0, 2, 3, 4, 1)
        out[k] = mx.array(arr.astype(np.float32))
    return list(out.items())


@pytest.mark.skipif(
    not os.path.exists(REF_PY_PATH),
    reason=f"reference STU-Net not at {REF_PY_PATH}",
)
def test_mlx_matches_pytorch_pretrained_plan():
    import torch

    ref = _load_reference()

    # Tiny dims so the test runs quickly. Plan is the published pretrained one
    # (anisotropic last stage) — what matters is shape geometry.
    dims = [4, 8, 16, 16, 16, 16]
    depth = [1] * 6

    torch.manual_seed(0)
    py_model = ref.STUNet(
        input_channels=1, num_classes=3,
        depth=depth, dims=dims,
        pool_op_kernel_sizes=PRETRAINED_POOL_OP_KERNEL_SIZES,
        conv_kernel_sizes=PRETRAINED_CONV_KERNEL_SIZES,
    ).eval()
    py_model._deep_supervision = False
    py_model.do_ds = False

    mlx_model = MLXSTUNet(
        input_channels=1, num_classes=3,
        depth=depth, dims=dims,
        pool_op_kernel_sizes=PRETRAINED_POOL_OP_KERNEL_SIZES,
        conv_kernel_sizes=PRETRAINED_CONV_KERNEL_SIZES,
    )
    mlx_model.load_weights(_torch_state_to_mlx(py_model.state_dict()), strict=True)

    # Patch divisible by [16, 16, 32] = prod of pool kernels per axis.
    # Use 2x that so the bottleneck has >1 spatial elements (InstanceNorm).
    np.random.seed(0)
    x_np = np.random.randn(1, 1, 32, 32, 64).astype(np.float32)

    with torch.no_grad():
        y_py = py_model(torch.from_numpy(x_np)).cpu().numpy()  # (B, C, D, H, W)

    x_mx = mx.array(x_np.transpose(0, 2, 3, 4, 1))  # → (B, D, H, W, C)
    y_mx = mlx_model(x_mx)
    mx.eval(y_mx)
    y_mx_np = np.array(y_mx).transpose(0, 4, 1, 2, 3)  # → (B, C, D, H, W)

    assert y_py.shape == y_mx_np.shape, (y_py.shape, y_mx_np.shape)
    max_diff = float(np.max(np.abs(y_py - y_mx_np)))
    assert max_diff < 1e-4, f"max abs diff {max_diff}"


@pytest.mark.skipif(
    not os.path.exists(REF_PY_PATH),
    reason=f"reference STU-Net not at {REF_PY_PATH}",
)
def test_deep_supervision_order_matches_pytorch():
    import torch

    ref = _load_reference()
    dims = [4, 8, 16, 16, 16, 16]
    depth = [1] * 6

    torch.manual_seed(1)
    py_model = ref.STUNet(
        input_channels=1, num_classes=3,
        depth=depth, dims=dims,
        pool_op_kernel_sizes=PRETRAINED_POOL_OP_KERNEL_SIZES,
        conv_kernel_sizes=PRETRAINED_CONV_KERNEL_SIZES,
    ).eval()
    py_model._deep_supervision = True
    py_model.do_ds = True

    mlx_model = MLXSTUNet(
        input_channels=1, num_classes=3,
        depth=depth, dims=dims,
        pool_op_kernel_sizes=PRETRAINED_POOL_OP_KERNEL_SIZES,
        conv_kernel_sizes=PRETRAINED_CONV_KERNEL_SIZES,
    )
    mlx_model.load_weights(_torch_state_to_mlx(py_model.state_dict()), strict=True)

    np.random.seed(1)
    x_np = np.random.randn(1, 1, 32, 32, 64).astype(np.float32)

    with torch.no_grad():
        py_outs = py_model(torch.from_numpy(x_np))

    mx_outs = mlx_model(mx.array(x_np.transpose(0, 2, 3, 4, 1)),
                        deep_supervision=True)
    mx.eval(mx_outs)

    assert len(py_outs) == len(mx_outs)
    # Output i should have matching D,H,W after channel reorder.
    for i, (p, m) in enumerate(zip(py_outs, mx_outs)):
        p_np = p.cpu().numpy()
        m_np = np.array(m).transpose(0, 4, 1, 2, 3)
        assert p_np.shape == m_np.shape, (i, p_np.shape, m_np.shape)
        diff = float(np.max(np.abs(p_np - m_np)))
        assert diff < 1e-4, f"output {i}: max diff {diff}"
