# Fine-tuning STU-Net on Apple Silicon

Train a custom 3D segmentation model on your Mac in under a minute.

## Quick start

```python
from stunet_mlx.weights import download_and_load
from stunet_mlx.lora import combined_loss
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
import numpy as np

# Load pre-trained STU-Net small (14.6M params, trained on TotalSegmentator)
model = download_and_load(variant="small")

# Add a new output head for your task
class MySegmenter(nn.Module):
    def __init__(self, base, num_classes=1):
        super().__init__()
        self.base = base
        self.head = nn.Conv3d(105, num_classes, kernel_size=1)
    def __call__(self, x):
        return self.head(self.base(x))

model = MySegmenter(model, num_classes=1)

# Load your data: list of (D, H, W) numpy arrays
# patches = [patch1, patch2, ...]  # CT patches, z-normalized
# labels = [label1, label2, ...]   # Binary masks

# Train
optimizer = mlx.optimizers.Adam(learning_rate=1e-4)
loss_and_grad = nn.value_and_grad(model, lambda m, x, y: combined_loss(m(x), y))

for epoch in range(20):
    for patch, label in zip(patches, labels):
        x = mx.array(patch[None, :, :, :, None])           # (1, D, H, W, 1)
        y = mx.array(label[None, :, :, :, None])            # (1, D, H, W, 1)
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(loss)
```

That's it. 20 epochs on 5 patches takes ~42 seconds on M2.

## What you get

A model that segments your custom structure from CT (or MRI) patches. The pre-trained STU-Net backbone already understands 3D medical image features — you're teaching it what your specific target looks like.

## Results

We tested three fine-tuning strategies on organ segmentation:

| Method | Trainable params | Dice | Training time |
|---|---|---|---|
| Head only (1×1 conv) | 106 | 0.86 | 22s |
| Full fine-tune (small lr) | 14.6M | **0.96** | 42s |
| Decoder only | 162K | 0.71 | 22s |

**Full fine-tuning wins.** At 14.6M parameters, STU-Net small is small enough that gradient memory isn't a bottleneck on Apple Silicon. No LoRA, adapters, or parameter-efficient tricks needed.

## Preprocessing

STU-Net expects z-normalized input:

```python
# For CT
ct_norm = (ct_data - ct_data.mean()) / (ct_data.std() + 1e-8)

# Extract patches (64³ or 128³)
patch = ct_norm[d:d+64, h:h+64, w:w+64]
```

No resampling needed — the model handles native resolution.

## Training tips

**Learning rate**: Use 1e-4 for full fine-tuning. Higher rates (1e-3) cause catastrophic forgetting of the pre-trained features.

**Patch size**: 64³ for training (fits many in memory), 128³ for inference.

**Data**: 5-50 annotated patches is enough for a single structure. More data helps for complex or variable structures.

**Epochs**: 20-50 is usually sufficient. Watch the loss — if it plateaus, stop.

**Loss function**: Dice + BCE (the standard nnU-Net combination) works well for binary segmentation:

```python
from stunet_mlx.lora import combined_loss  # dice_loss + bce_loss
```

**Don't freeze the encoder.** Counter-intuitively, freezing the encoder and training only the decoder produces worse results (0.71 vs 0.96). The encoder's features need slight adjustment to align with the new task. A small learning rate prevents catastrophic forgetting while allowing adaptation.

## Saving and loading

```python
# Save the fine-tuned model
import mlx.utils
weights = dict(mlx.utils.tree_flatten(model.parameters()))
mx.savez("my_model.npz", **weights)

# Load later
weights = dict(mx.load("my_model.npz"))
model.load_weights(list(weights.items()))
```

## When you need parameter-efficient fine-tuning

For STU-Net small (14.6M), full fine-tuning is the right approach. Parameter-efficient methods (LoRA, BitFit, adapters) become necessary for:

- **STU-Net Large (440M)** — gradient memory may exceed 16GB
- **STU-Net Huge (1.4B)** — definitely needs LoRA or similar
- **Multi-task adapters** — if you want to keep the base frozen and swap lightweight heads

The `lora.py` module provides `SimpleLoRAHead` (head-only, 106 params) and the infrastructure for future LoRA conv adapter development.

## Practical example: train on TotalSegmentator labels

Use TotalSegmentator to generate labels for structures not in STU-Net's base classes, then fine-tune:

```bash
# Generate labels for liver vessels (not in STU-Net's 104 base classes)
TotalSegmentator -i ct.nii -o labels/ -ta liver_vessels

# Extract patches where vessels are present
# Fine-tune STU-Net to segment vessels
# Result: a lightweight vessel segmentation model
```

## Model variants

| Variant | Params | Full fine-tune on 16GB? | Notes |
|---|---|---|---|
| **Small** | **14.6M** | **Yes (42s)** | Recommended for fine-tuning |
| Base | 58M | Yes (~2 min) | More capacity |
| Large | 440M | Tight — may need LoRA | Much more capacity |
| Huge | 1.4B | No — needs LoRA + 32GB | Research only |
