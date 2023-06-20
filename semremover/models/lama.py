import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import logging

import yaml
import torch
import numpy as np
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo

from .utils import package_path


class LaMa:
    def __init__(self, ckpt, config):
        if not os.path.exists(ckpt):
            log.warning(f"Could not find LaMa checkpoint at specified path.")
            ckpt = self.__download_lama_weights()
        self.model, self.config, self.device = self.__load_config(ckpt, config)

    @staticmethod
    def __download_lama_weights():
        ckpt_dir = package_path("models/weights/big-lama")
        ckpt_file = os.path.join(ckpt_dir, "models", "best.ckpt")
        config_file = os.path.join(ckpt_dir, "config.yaml")
        if os.path.exists(ckpt_file) and os.path.exists(config_file):
            log.info("Using default Big-LaMa weights.")
            return ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        log.info("Downloading Big-LaMa model weights, this might take a minute...")
        url = "https://github.com/Jessseee/lama/releases/latest/download/big-lama.zip"
        with urlopen(url) as zip_repr:
            with ZipFile(BytesIO(zip_repr.read())) as zip_file:
                zip_file.extractall(ckpt_dir)
        return ckpt_dir

    @staticmethod
    def __load_config(ckpt, config):
        predict_config = OmegaConf.load(config)
        predict_config.model.input_path = ckpt
        device = torch.device("cuda")
        train_config_path = os.path.join(predict_config.model.input_path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        checkpoint_path = os.path.join(predict_config.model.input_path, 'models', predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)
        return model, predict_config, device

    def __create_batch(self, image: np.ndarray, mask: np.ndarray):
        batch = dict()
        batch['image'] = image.permute(2, 0, 1).unsqueeze(0)
        batch['mask'] = mask[None, None]
        batch['image'] = pad_tensor_to_modulo(batch['image'], 8)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], 8)
        batch = move_to_device(batch, self.device)
        batch['mask'] = (batch['mask'] > 0) * 1
        return batch

    @torch.no_grad()
    def inpaint(self, image: np.ndarray, mask: np.ndarray):
        assert len(mask.shape) == 2
        if np.max(mask) == 1:
            mask = mask * 255
        image = torch.from_numpy(image).float().div(255.)
        mask = torch.from_numpy(mask).float()

        batch = self.__create_batch(image, mask)
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]

        batch = self.model(batch)
        cur_res = batch[self.config.out_key][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        return cur_res
