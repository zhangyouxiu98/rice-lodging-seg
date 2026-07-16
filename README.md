# Rice Lodging & Disease Segmentation

Semantic segmentation for detecting rice lodging and Black Stripe Dwarfism disease in paddy fields, based on DeepLabv3+ with **CGAFusion** (Cross-Gated Attention Fusion) and **LocalAttention**.

## Classes

| Index | Class |
|-------|-------|
| 0 | `_background_` |
| 1 | `Black_stripe_dwarfism` |
| 2 | `lodging` |

## Installation

```bash
pip install -r requirements.txt
pip install einops
```

For ONNX export:

```bash
pip install onnx onnx-simplifier
```

## Dataset

PASCAL VOC format:

```
VOCdevkit/VOC2007/
  JPEGImages/         # .jpg images
  SegmentationClass/  # .png labels (pixel value = class index)
  ImageSets/Segmentation/
    train.txt
    val.txt
```

**Important:** Label pixels must use contiguous class indices starting at 0. Using 255 for objects is a common mistake — it will train but predictions will be wrong.

Generate dataset splits:

```bash
python voc_annotation.py
```

## Model

### CGAFusion + LocalAttention (`nets/cga.py`)

| Backbone | Parameters |
|----------|------------|
| MobileNetV2 | ~6.1M |
| Xception | ~55.1M |

- **Encoder:** Backbone → ASPP with CGAFusion (spatial + channel + pixel attention)
- **Decoder:** LocalAttention on low-level features → skip connection → classification head

### Custom Attention Modules (`nets/attention.py`)

- `CGAFusion` — cross-gated fusion of spatial, channel, and pixel attention in ASPP branches
- `LocalAttention` — soft-pooling → conv → sigmoid gating for local feature importance
- `SpatialAttention` / `ChannelAttention` / `PixelAttention` — building blocks
- `TBFE` — three-branch feature extraction with residual connections
- `SoftPooling2D` — softmax-based pooling

## Usage

### Train

Edit configuration in `train.py` (under `if __name__ == "__main__":`):

```python
num_classes     = 3          # classes + 1
backbone        = "mobilenet"
VOCdevkit_path  = '../datasets/VOCdevkit_mydata'
input_shape     = [512, 512]
```

```bash
python train.py
```

Multi-GPU (Linux):

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### Predict

Configure `deeplab.py` `_defaults` dict, then:

```python
# predict.py
mode = "predict"           # single image
mode = "dir_predict"       # batch directory
mode = "video"             # camera / video file
mode = "fps"               # speed benchmark
mode = "export_onnx"       # export ONNX
```

```bash
python predict.py
```

### Evaluate

```bash
python get_miou.py
```

## Results

| Metric | Value |
|--------|-------|
| Inference | ~270 ms/image (CPU) |
| Model size | 23.5 MB (fp32, MobileNetV2) |

## License

MIT. This project is a modified work based on the DeepLabv3+ architecture with custom attention mechanisms.
