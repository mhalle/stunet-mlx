# stunet-mlx

MLX port of [STU-Net](https://github.com/uni-medical/STU-Net) — scalable universal 3D medical image segmentation on Apple Silicon.

STU-Net is a scaled nnU-Net with residual blocks, pre-trained on TotalSegmentator (104 anatomical classes). The small variant (14.6M parameters) runs 128³ patches comfortably on M2 16GB.

## Quick start

```bash
git clone https://github.com/mhalle/stunet-mlx.git
cd stunet-mlx
uv sync

uv run python -c "
from stunet_mlx.weights import download_and_load
import mlx.core as mx

model = download_and_load(variant='small')
x = mx.random.normal((1, 128, 128, 128, 1))
out = model(x)
mx.eval(out)
print(out.shape)  # (1, 128, 128, 128, 105)
"
```

## Results (M2 16GB)

Single-patch (128³) and sliding window on a real CT:

| Organ | Single patch | Sliding window (8 patches, 10s) |
|---|---|---|
| Stomach | — | 0.78 |
| Lung UL left | 0.76 | 0.77 |
| Liver | 0.69 | 0.38 (label collision with spleen) |
| Colon | 0.56 | 0.58 |
| Lung UR right | 0.56 | 0.58 |

128³ forward pass: **0.96 seconds**. Full sliding window: **10.5 seconds** (8 patches).

## Architecture

```
STU-Net Small (14.6M params)
├── Encoder: 6 stages, dims=[16, 32, 64, 128, 256, 256]
│   └── BasicResBlock: Conv3d→InstanceNorm→LeakyReLU→Conv3d→InstanceNorm + 1x1 skip
│   └── Strided conv for downsampling (stride=2)
├── Decoder: 5 levels
│   └── Nearest upsample + Conv3d(1x1) for channels
│   └── Concatenate skip connections
│   └── BasicResBlock
└── Deep supervision outputs at each decoder level
```

Variants: Small (14.6M), Base (58M), Large (440M), Huge (1.4B).

## Model variants

| Variant | Params | Dims | Depth | Fits 16GB? |
|---|---|---|---|---|
| **Small** | **14.6M** | [16,32,64,128,256,256] | [1,1,1,1,1,1] | **Yes** |
| Base | 58M | [32,64,128,256,512,512] | [1,1,1,1,1,1] | Yes |
| Large | 440M | [64,128,256,512,1024,1024] | [2,2,2,2,2,2] | Tight |
| Huge | 1.4B | [96,192,384,768,1536,1536] | [3,3,3,3,3,3] | No |

```python
model = download_and_load(variant='base')  # 58M params
```

## Related work

- [nnunet-mlx](https://github.com/mhalle/nnunet-mlx) — nnU-Net / TotalSegmentator (production, Dice >0.95)
- [segvol-mlx](https://github.com/mhalle/segvol-mlx) — SegVol (text-prompted, 0.95 Dice with text+box)
- [vista3d-mlx](https://github.com/mhalle/vista3d-mlx) — VISTA3D (needs 32GB)
- [sam-med3d-mlx](https://github.com/mhalle/sam-med3d-mlx) — SAM-Med3D

## License

Apache 2.0
