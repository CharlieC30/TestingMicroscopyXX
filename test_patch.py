"""
Class-based microscopy inference script.
Refactored from test_simple.py into modular classes.

Usage:
    python test_classes.py --config Default --option AIRYSCAN --gpu
"""

import argparse
import os
import shutil
import threading
import queue
import numpy as np
import torch
import torch.nn.functional as F
import tifffile as tiff
import yaml
from tqdm import tqdm

from utils.model_utils import read_json_to_args, import_model, load_pth, ModelProcesser
from utils.data_utils import DataNormalization


# =============================================================================
# Config
# =============================================================================
class Config:
    """Wraps YAML config loading and access."""

    def __init__(self, config_name, option):
        self.config_name = config_name
        self.option = option
        self.cfg = self._load()

    def _load(self):
        with open(f'test/{self.config_name}.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return {**config['DEFAULT'], **config[self.option]}

    def get(self, key, default=None):
        return self.cfg.get(key, default)

    def __getitem__(self, key):
        return self.cfg[key]


# =============================================================================
# ModelLoader
# =============================================================================
class VQQModel:
    """Container for VQQ2 model components."""

    def __init__(self, ckpt, epoch):
        self.encoder = torch.load(f"{ckpt}encoder_model_epoch_{epoch}.pth", map_location='cpu')
        self.decoder = torch.load(f"{ckpt}decoder_model_epoch_{epoch}.pth", map_location='cpu')
        self.net_g = torch.load(f"{ckpt}net_g_model_epoch_{epoch}.pth", map_location='cpu')
        self.quantize = torch.load(f"{ckpt}quantize_model_epoch_{epoch}.pth", map_location='cpu')
        self.quant_conv = torch.load(f"{ckpt}quant_conv_model_epoch_{epoch}.pth", map_location='cpu')
        self.post_quant_conv = torch.load(f"{ckpt}post_quant_conv_model_epoch_{epoch}.pth", map_location='cpu')

    def cuda(self):
        for attr in ['encoder', 'decoder', 'net_g', 'quantize', 'quant_conv', 'post_quant_conv']:
            setattr(self, attr, getattr(self, attr).cuda())
        return self

    def half(self):
        for attr in ['encoder', 'decoder', 'net_g', 'quantize', 'quant_conv', 'post_quant_conv']:
            setattr(self, attr, getattr(self, attr).half())
        return self

    def parameters(self):
        for attr in ['encoder', 'decoder', 'net_g', 'quantize', 'quant_conv', 'post_quant_conv']:
            for p in getattr(self, attr).parameters():
                yield p


class ModelLoader:
    """Handles model loading for different model types."""

    def __init__(self, cfg, gpu=False, fp16=False, augmentation='encode'):
        self.cfg = cfg
        self.gpu = gpu
        self.fp16 = fp16
        self.augmentation = augmentation

        self.model = self._load_model()
        self.upsample = torch.nn.Upsample(size=cfg['upsample_params']['size'], mode='trilinear')

        if gpu:
            self.upsample = self.upsample.cuda()

        self.model_proc = ModelProcesser(
            cfg.cfg if isinstance(cfg, Config) else cfg,
            self.model,
            gpu=gpu,
            augmentation=augmentation,
            fp16=fp16
        )

    def _load_model(self):
        cfg = self.cfg.cfg if isinstance(self.cfg, Config) else self.cfg
        ckpt_root = f"{cfg['SOURCE']}/logs/{cfg['prj']}"
        epoch = cfg['epoch']
        model_type = cfg['model_type']

        if model_type == 'AE':
            model = self._load_ae(ckpt_root, epoch)
        elif model_type == 'GAN':
            model = self._load_gan(ckpt_root, epoch)
        elif model_type == 'VQQ2':
            model = self._load_vqq2(ckpt_root, epoch)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if self.gpu:
            model = model.cuda()
        for p in model.parameters():
            p.requires_grad = False
        if self.fp16:
            model = model.half()

        return model

    def _load_ae(self, ckpt_root, epoch):
        args = read_json_to_args(f"{ckpt_root}0.json")
        model_module = import_model(ckpt_root, model_name=args.models)
        model = model_module.GAN(args, train_loader=None, eval_loader=None, checkpoints=None)
        model = load_pth(model, root=ckpt_root, epoch=epoch,
                         model_names=['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv'])
        return model

    def _load_gan(self, ckpt_root, epoch):
        model_path = f"{ckpt_root}checkpoints/net_g_model_epoch_{epoch}.pth"
        return torch.load(model_path, map_location='cpu')

    def _load_vqq2(self, ckpt_root, epoch):
        ckpt = f"{ckpt_root}checkpoints/"
        return VQQModel(ckpt, epoch)


# =============================================================================
# ImageLoader
# =============================================================================
class ImageLoader:
    """Handles image loading and preprocessing."""

    def __init__(self, cfg):
        self.cfg = cfg.cfg if isinstance(cfg, Config) else cfg
        self.normalizer = DataNormalization(backward_type='float32')
        self.img = None

    def load(self):
        image_path = self.cfg['root_path'] + self.cfg['image_path'][0]
        print(f"Loading: {image_path}")

        img = tiff.imread(image_path)
        self.img = self.normalizer.forward_normalization(
            img, self.cfg['norm_method'][0], self.cfg['trd'][0]
        )
        return self.img

    def apply_percentile_norm(self):
        if self.img is None:
            raise ValueError("Image not loaded. Call load() first.")

        if self.cfg.get('norm_percentile') is not False:
            p0, p1 = self.cfg.get('norm_percentile', [0.5, 99.5])
            if not isinstance(p0, (int, float)):
                p0, p1 = 0.5, 99.5

            xmin = np.percentile(self.img[:, ::2, ::2].flatten(), p0)
            xmax = np.percentile(self.img[:, ::2, ::2].flatten(), p1)
            print(f"Percentile clipping: [{xmin:.2f}, {xmax:.2f}]")

            self.img = torch.clamp(self.img, xmin, xmax)
            self.img = (self.img - xmin) / (xmax - xmin)
            self.img = self.img * 2 - 1

        return self.img

    def pad(self, patch_shape, crop_margin=None):
        """
        Pad image for patch-based inference.

        Args:
            patch_shape: (dz, dx, dy) patch dimensions
            crop_margin: (Cz, Cx, Cy) pixels cropped from each side during assembly.
                         If provided, pads by C on both sides to preserve original size.
        """
        if self.img is None:
            raise ValueError("Image not loaded. Call load() first.")

        _, _, Dz, Dx, Dy = self.img.shape
        dz, dx, dy = patch_shape

        print(f"Padding image from {self.img.shape}")

        # First, pad by crop margin (C) on both sides to preserve original size after assembly
        if crop_margin is not None:
            Cz, Cx, Cy = crop_margin
            self.img = F.pad(self.img, (Cy, Cy, Cx, Cx, Cz, Cz), mode='constant', value=self.img.min())
            _, _, Dz, Dx, Dy = self.img.shape  # Update dimensions after C padding
            print(f"After C padding: {self.img.shape}")

        # Then, pad to make divisible by patch shape
        Nz = ((Dz // dz) + 1) * dz
        Nx = ((Dx // dx) + 1) * dx
        Ny = ((Dy // dy) + 1) * dy

        Pz, Px, Py = Nz - Dz, Nx - Dx, Ny - Dy
        self.img = F.pad(self.img, (0, Py, 0, Px, 0, Pz), mode='constant', value=self.img.min())
        print(f"Final padded shape: {self.img.shape}")

        return self.img


# =============================================================================
# PatchProcessor
# =============================================================================
class PatchProcessor:
    """Handles inference on patches."""

    def __init__(self, model_proc, upsample, augmentations, fp16=False, gpu=False):
        self.model_proc = model_proc
        self.upsample = upsample
        self.augmentations = augmentations
        self.fp16 = fp16
        self.gpu = gpu

    def run(self, x0_list, d0, dx, ii=None):
        # Extract and upsample patch
        patch = [x[:, :, d0[0]:d0[0]+dx[0], d0[1]:d0[1]+dx[1], d0[2]:d0[2]+dx[2]] for x in x0_list]
        patch = torch.cat([self.upsample(x).squeeze().unsqueeze(1) for x in patch], 1)

        if self.fp16 and self.gpu:
            patch = patch.half()
            with torch.cuda.amp.autocast():
                _, Xup, outall, _ = self.model_proc.get_model_result(patch, self.augmentations, ii=ii)
        else:
            _, Xup, outall, _ = self.model_proc.get_model_result(patch, self.augmentations, ii=ii)

        outall = outall.numpy().astype(np.float32)
        Xup = Xup.numpy().astype(np.float32)

        # Match output range to input range per channel
        for c in range(outall.shape[1]):
            omin, omax = outall[:, c, :, :, :].min(), outall[:, c, :, :, :].max()
            xmin, xmax = Xup[:, c, :, :].min(), Xup[:, c, :, :].max()
            outall[:, c, :, :, :] = (outall[:, c, :, :, :] - omin) / (omax - omin + 1e-6) * (xmax - xmin) + xmin

        return outall, Xup


# =============================================================================
# OutputWriter
# =============================================================================
class OutputWriter:
    """Handles threaded file writing."""

    def __init__(self, dest, save_types, output_dtype='float32'):
        self.dest = dest
        self.save_types = save_types
        self.output_dtype = output_dtype
        self.write_queue = queue.Queue(maxsize=100)
        self.writer_thread = None

    def _convert_dtype(self, data):
        """Convert data from (-1, 1) to target dtype."""
        if self.output_dtype == 'uint8':
            # (-1, 1) -> (0, 255)
            data = np.clip((data + 1) * 127.5, 0, 255).astype(np.uint8)
        elif self.output_dtype == 'uint16':
            # (-1, 1) -> (0, 65535)
            data = np.clip((data + 1) * 32767.5, 0, 65535).astype(np.uint16)
        return data

    def _writer_loop(self):
        while True:
            item = self.write_queue.get()
            if item is None:
                self.write_queue.task_done()
                break

            iz, ix, iy, out_all, patch = item
            try:
                out_all_mean = out_all.mean(axis=-1)
                out_all_mean = np.transpose(out_all_mean, (1, 0, 2, 3))
                patch = np.transpose(patch, (1, 0, 2, 3))

                # Convert to target dtype
                out_all_mean = self._convert_dtype(out_all_mean)
                patch = self._convert_dtype(patch)

                if 'xy' in self.save_types:
                    tiff.imwrite(os.path.join(self.dest, 'xy', f'{iz}_{ix}_{iy}.tif'), out_all_mean)
                if 'ori' in self.save_types:
                    tiff.imwrite(os.path.join(self.dest, 'ori', f'{iz}_{ix}_{iy}.tif'), patch)
            except Exception as e:
                print(f"Error writing {iz}_{ix}_{iy}: {e}")
            finally:
                self.write_queue.task_done()

    def start(self):
        self.writer_thread = threading.Thread(target=self._writer_loop)
        self.writer_thread.start()

    def write(self, iz, ix, iy, out_all, patch):
        self.write_queue.put((iz, ix, iy, out_all, patch))

    def stop(self):
        self.write_queue.put(None)
        self.writer_thread.join()
        self.write_queue.join()


# =============================================================================
# InferencePipeline
# =============================================================================
class InferencePipeline:
    """Orchestrates the full inference pipeline."""

    def __init__(self, args):
        self.args = args
        self.config = Config(args.config, args.option)

        # Setup output directory
        self.dest = os.path.join(self.config['DESTINATION'], self.config['dataset'])
        self._setup_output_dirs()

        # Initialize components
        print("Loading model...")
        self.model_loader = ModelLoader(
            self.config,
            gpu=args.gpu,
            fp16=args.fp16,
            augmentation=args.augmentation
        )

        print("Loading image...")
        self.image_loader = ImageLoader(self.config)

        self.processor = PatchProcessor(
            self.model_loader.model_proc,
            self.model_loader.upsample,
            self.config.get('input_augmentation', [None]),
            fp16=args.fp16,
            gpu=args.gpu
        )

        output_dtype = self.config.get('output_dtype', 'float32')
        self.writer = OutputWriter(self.dest, args.save, output_dtype=output_dtype)

    def _setup_output_dirs(self):
        for folder in self.args.save:
            folder_path = os.path.join(self.dest, folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path, exist_ok=True)

        # Save config
        with open(os.path.join(self.dest, 'config.yaml'), 'w') as f:
            yaml.dump(self.config.cfg, f)

    def _get_patch_grid(self, img_shape):
        cfg = self.config.cfg
        _, _, zz, xx, yy = img_shape

        if cfg['assemble_params'].get('testwhole', True):
            zstep = int(eval(str(cfg['assemble_params']['zrange'][-1])))
            xstep = int(eval(str(cfg['assemble_params']['xrange'][-1])))
            ystep = int(eval(str(cfg['assemble_params']['yrange'][-1])))
            zrange = range(0, zz, zstep)
            xrange = range(0, xx, xstep)
            yrange = range(0, yy, ystep)
        else:
            zrange = range(*[int(eval(str(x))) for x in cfg['assemble_params']['zrange']])
            xrange = range(*[int(eval(str(x))) for x in cfg['assemble_params']['xrange']])
            yrange = range(*[int(eval(str(x))) for x in cfg['assemble_params']['yrange']])

        return zrange, xrange, yrange

    def run(self):
        # Load and preprocess image
        self.image_loader.load()
        self.image_loader.apply_percentile_norm()

        patch_shape = self.config['assemble_params']['dx_shape']
        dz, dx, dy = patch_shape

        # Get crop margin C for padding (to preserve original size after assembly)
        crop_margin = None
        if self.config.cfg['assemble_params'].get('testwhole', True):
            crop_margin = self.config['assemble_params']['C']

        self.image_loader.pad(patch_shape, crop_margin=crop_margin)

        x0 = [self.image_loader.img]

        # Get patch grid
        zrange, xrange, yrange = self._get_patch_grid(self.image_loader.img.shape)
        total = len(list(xrange)) * len(list(zrange)) * len(list(yrange))
        print(f"Processing {len(list(zrange))}x{len(list(xrange))}x{len(list(yrange))} = {total} patches")

        # Start writer
        self.writer.start()

        try:
            print('xrange:', xrange)
            print('zrange:', zrange)
            print('yrange:', yrange)
            with tqdm(total=total, desc="Processing") as pbar:
                for ix in xrange:
                    for iz in zrange:
                        for iy in yrange:
                            out_all, Xup = self.processor.run(
                                x0,
                                d0=[iz, ix, iy],
                                dx=[dz, dx, dy],
                                ii=(iz, ix, iy)
                            )
                            self.writer.write(iz, ix, iy, out_all, Xup)
                            pbar.update(1)
        finally:
            self.writer.stop()

        print(f"Done! Output saved to: {self.dest}")


# =============================================================================
# Main
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Class-based microscopy inference')
    parser.add_argument('--config', type=str, required=True, help='Config file name (without .yaml)')
    parser.add_argument('--option', type=str, required=True, help='Option/dataset name in config')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    parser.add_argument('--augmentation', type=str, default='encode', choices=['encode', 'decode'],
                        help='Augmentation stage: encode or decode')
    parser.add_argument('--save', nargs='+', default=['ori', 'xy'], help='What to save: ori, xy')
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = InferencePipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()
