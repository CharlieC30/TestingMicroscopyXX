# TestingMicroscopyXX

A deep learning framework for 3D microscopy image super-resolution and reconstruction.

## Overview

This project performs inference and reconstruction on 3D microscopy images using various neural network architectures including Autoencoders, GANs, and Vector Quantized models. It supports patch-based processing for large volumetric data with seamless assembly.

## Features

- **Multiple Model Architectures**: Autoencoders (AE), GANs, VQ-VAE2, CycleGAN/CUT
- **3D Volume Processing**: Patch-based inference with overlapping regions
- **Test-Time Augmentation**: Multiple augmentation strategies for robust predictions
- **Monte Carlo Inference**: Uncertainty estimation via multiple inference runs
- **FP16 Precision**: Efficient mixed-precision inference
- **Seamless Assembly**: Tapered weighting for smooth patch stitching

## Project Structure

```
TestingMicroscopyXX/
├── test_only.py           # Main inference script
├── test_assemble.py       # Volume assembly script
├── run.sh                 # Execution examples
├── requirements.txt       # Dependencies
├── models/                # Model definitions
│   ├── base.py            # Base classes, VGG losses
│   ├── ae0iso0tc.py       # AutoEncoder model
│   └── CUT.py             # Contrastive Unpaired Translation
├── networks/              # Neural network architectures
│   ├── EncoderDecoder/    # 17 encoder-decoder variants
│   ├── resunet/           # ResUNet architectures
│   ├── DeScarGan/         # GAN components
│   └── cyclegan/          # CycleGAN implementations
├── utils/                 # Utility modules
│   ├── model_utils.py     # ModelProcesser class
│   ├── data_utils.py      # DataNormalization class
│   └── raw_to_patches.py  # Patch extraction
├── ldm/                   # Latent Diffusion Model components
├── taming/                # VQGAN/taming modules
└── test/                  # YAML configuration files
```

## Supported Microscopy Modalities

- Structured Illumination Microscopy (SIM)
- Selective Plane Illumination Microscopy (SPIM)
- Golgi apparatus imaging
- Expansion microscopy (iUExM)
- Blood vessel imaging
- Organoid imaging

## Installation

```bash
pip install -r requirements.txt
```

### Key Dependencies

- PyTorch 1.10.0
- PyTorch Lightning 1.9.5
- tifffile, OpenCV, scikit-image
- albumentations, einops

## Usage

### 1. Inference

Run model inference on a 3D microscopy volume:

```bash
python test_only.py --gpu --config ConfigName --save ori xy \
    --augmentation encode --testcube --option MODELOPTION
```

**Arguments:**
- `--gpu`: Use GPU acceleration
- `--config`: Configuration file name (from `test/` directory)
- `--save`: Output types to save
- `--augmentation`: Augmentation strategy (encode/decode)
- `--testcube`: Enable test cube mode
- `--option`: Model-specific options

### 2. Volume Assembly

Assemble patches into complete 3D volumes:

```bash
python test_assemble.py --config ConfigName --targets xy --option MODELOPTION
```

## Configuration

Configurations are YAML files in the `test/` directory. Key parameters:

```yaml
image_path: /path/to/input.tif
checkpoint: /path/to/model/checkpoints
epoch: 100
patch_size: [32, 256, 256]
stride: [16, 128, 128]
normalization: '01'
```

## Workflow

1. **Configure**: Select/create a YAML config file
2. **Inference**: Run `test_only.py` to process patches
3. **Assemble**: Run `test_assemble.py` to combine patches
4. **Output**: Reconstructed 3D TIFF volumes

## Core Components

### ModelProcesser (`utils/model_utils.py`)

Handles model loading and inference:
- Loads AE/GAN/VQQ2 checkpoints
- Manages encoder/decoder processing
- Supports test-time augmentation

### DataNormalization (`utils/data_utils.py`)

Handles image normalization:
- Multiple methods: '00', '01', '11'
- Percentile-based clipping
- Forward/backward transforms

## Output

- Reconstructed 3D volumes in TIFF format
- Multiple output channels (xy, xz, etc.)
- Optional uncertainty maps (standard deviation)
- Supports uint8, uint16, float32 formats

## License

[Add license information]

## Citation

[Add citation information if applicable]
