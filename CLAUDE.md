# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep learning framework for 3D microscopy image super-resolution and reconstruction. Supports Autoencoders, GANs, VQ-VAE2, and CycleGAN/CUT architectures with patch-based processing for large volumetric data.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference on 3D volume
python test_patch.py --gpu --config <ConfigName> --option <ModelOption> --augmentation encode --save ori xy

# Assemble patches into complete volume
python test_assemble.py --config <ConfigName> --targets xy ori --image_datatype float32 --option <ModelOption>
```

Example with DefaultGolgiNovX2 config:
```bash
python test_patch.py --gpu --config DefaultGolgiNovX2 --option ENC --augmentation encode --save ori xy
python test_assemble.py --config DefaultGolgiNovX2 --targets xy ori --image_datatype float32 --option ENC
```

## Architecture

- **models/** - Model definitions wrapping network architectures (base.py has VGG losses, ae0iso0tc.py for AutoEncoder, CUT.py for Contrastive Unpaired Translation)
- **networks/** - Neural network implementations
  - `EncoderDecoder/` - 17 encoder-decoder variants (ed023e.py, ed023eunet3dres.py, etc.)
  - `resunet/`, `cyclegan/`, `DeScarGan/` - Architecture-specific implementations
- **utils/** - Core utilities
  - `model_utils.py` - ModelProcesser class for loading checkpoints and running inference
  - `data_utils.py` - DataNormalization class with methods '00', '01', '11'
  - `raw_to_patches.py` - Patch extraction from volumes
- **ldm/** - Latent Diffusion Model components
- **test/** - YAML configuration files

## Configuration System

Configs are YAML files in `test/` with two sections:
- `DEFAULT`: Global settings (paths, patch sizes, assembly parameters)
- Named sections (e.g., `ENC`): Model-specific options selected via `--option` flag

Key config parameters:
- `image_path`: Input TIFF file(s)
- `checkpoint`: Model checkpoint directory
- `epoch`: Checkpoint epoch to load
- `model_type`: Architecture type ("VQQ2", "AE", etc.)
- `norm_method`: Normalization per input ('00', '01', '11')
- `assemble_params`: Patch stitching settings (crop, stride, weight method)

## Workflow

1. Create/edit YAML config in `test/`
2. Run `test_patch.py` to process volume as patches
3. Run `test_assemble.py` to stitch patches with tapered weighting
4. Output: Reconstructed 3D TIFF volumes
