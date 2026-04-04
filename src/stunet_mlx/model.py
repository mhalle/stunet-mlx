"""STU-Net model for MLX.

Scalable and Transferable U-Net — a residual nnU-Net variant.
Mirrors PyTorch naming exactly for zero-remapping weight loading.
"""

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class BasicResBlock(nn.Module):
    """Residual block: Conv→Norm→LeakyReLU→Conv→Norm + skip → LeakyReLU."""

    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: int = 3, padding: int = 1, stride: int = 1,
                 use_1x1conv: bool = False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size,
                               stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm(output_channels, affine=True)
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm(output_channels, affine=True)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def __call__(self, x: mx.array) -> mx.array:
        y = self.conv1(x)
        y = nn.leaky_relu(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        return nn.leaky_relu(y + x)


class UpsampleLayer(nn.Module):
    """Nearest-neighbor upsample + 1x1 conv for channel reduction."""

    def __init__(self, input_channels: int, output_channels: int,
                 scale_factor: Tuple[int, ...]):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.scale_factor = scale_factor

    def __call__(self, x: mx.array) -> mx.array:
        # MLX Upsample: channels-last (B, D, H, W, C)
        up = nn.Upsample(scale_factor=self.scale_factor, mode="nearest")
        x = up(x)
        return self.conv(x)


class STUNet(nn.Module):
    """Scalable and Transferable U-Net.

    Mirrored PyTorch attribute names for direct weight loading.
    """

    def __init__(self, input_channels: int = 1, num_classes: int = 105,
                 depth: List[int] = [1, 1, 1, 1, 1, 1],
                 dims: List[int] = [16, 32, 64, 128, 256, 256],
                 pool_op_kernel_sizes: List[List[int]] = None,
                 conv_kernel_sizes: List[List[int]] = None):
        super().__init__()

        if pool_op_kernel_sizes is None:
            pool_op_kernel_sizes = [[2, 2, 2]] * 5
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [[3, 3, 3]] * 6

        num_pool = len(pool_op_kernel_sizes)
        conv_pad_sizes = [[k // 2 for k in ks] for ks in conv_kernel_sizes]

        # Encoder
        self.conv_blocks_context = []
        # Stage 0: no stride
        stage0 = [BasicResBlock(input_channels, dims[0],
                                conv_kernel_sizes[0][0], conv_pad_sizes[0][0],
                                use_1x1conv=True)]
        stage0 += [BasicResBlock(dims[0], dims[0],
                                 conv_kernel_sizes[0][0], conv_pad_sizes[0][0])
                   for _ in range(depth[0] - 1)]
        self.conv_blocks_context.append(stage0)

        # Stages 1-5: strided first block
        for d in range(1, num_pool + 1):
            stage = [BasicResBlock(dims[d - 1], dims[d],
                                   conv_kernel_sizes[d][0], conv_pad_sizes[d][0],
                                   stride=pool_op_kernel_sizes[d - 1][0],
                                   use_1x1conv=True)]
            stage += [BasicResBlock(dims[d], dims[d],
                                    conv_kernel_sizes[d][0], conv_pad_sizes[d][0])
                      for _ in range(depth[d] - 1)]
            self.conv_blocks_context.append(stage)

        # Upsample layers
        self.upsample_layers = [
            UpsampleLayer(dims[-1 - u], dims[-2 - u],
                          tuple(pool_op_kernel_sizes[-1 - u]))
            for u in range(num_pool)
        ]

        # Decoder
        self.conv_blocks_localization = []
        for u in range(num_pool):
            stage = [BasicResBlock(dims[-2 - u] * 2, dims[-2 - u],
                                   conv_kernel_sizes[-2 - u][0], conv_pad_sizes[-2 - u][0],
                                   use_1x1conv=True)]
            stage += [BasicResBlock(dims[-2 - u], dims[-2 - u],
                                    conv_kernel_sizes[-2 - u][0], conv_pad_sizes[-2 - u][0])
                      for _ in range(depth[-2 - u] - 1)]
            self.conv_blocks_localization.append(stage)

        # Segmentation outputs (deep supervision)
        self.seg_outputs = [
            nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1)
            for ds in range(num_pool)
        ]

    def __call__(self, x: mx.array, deep_supervision: bool = False) -> mx.array:
        # Encoder
        skips = []
        for d in range(len(self.conv_blocks_context) - 1):
            for blk in self.conv_blocks_context[d]:
                x = blk(x)
            skips.append(x)

        # Bottleneck
        for blk in self.conv_blocks_context[-1]:
            x = blk(x)

        # Decoder
        seg_outputs = []
        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = mx.concatenate([x, skips[-(u + 1)]], axis=-1)  # channels-last concat
            for blk in self.conv_blocks_localization[u]:
                x = blk(x)
            seg_outputs.append(self.seg_outputs[u](x))

        if deep_supervision:
            return seg_outputs
        return seg_outputs[-1]  # finest resolution output
