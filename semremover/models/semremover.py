import os

import cv2
import numpy as np
from PIL import Image

from . import MaskFormer, LaMa


class SemanticObjectRemover:
    def __init__(self, lama_ckpt: str | os.PathLike, lama_config: str | os.PathLike,
                 maskformer_ckpt: str | os.PathLike, label_file: str | os.PathLike):
        self.maskformer = MaskFormer(maskformer_ckpt, label_file)
        self.lama = LaMa(lama_ckpt, lama_config)

    @staticmethod
    def __load_image_to_array(image_path: str | os.PathLike) -> np.ndarray:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        return np.array(img)

    @staticmethod
    def __array_to_image(img_array: np.ndarray) -> Image:
        return Image.fromarray(img_array.astype(np.uint8))

    @staticmethod
    def __dilate_mask(mask: np.ndarray, dilate_factor: int = 15) -> np.ndarray:
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((dilate_factor, dilate_factor), np.uint8), iterations=1)
        return mask

    def remove_objects_from_image(self, input_path: str | os.PathLike, labels: list[str], dilate_kernel_size: int = 15) -> Image:
        """ Remove objects specified by labels from input image. """
        if not os.path.exists(input_path) or os.path.isdir(input_path):
            raise IOError(f"{input_path} is not a file.")

        # Load the image from file into ndarray
        image = self.__load_image_to_array(input_path)

        # Get masks from segmentation
        masks, labels = self.maskformer.segment(image, labels)

        # Save original image if no objects were found to remove
        if masks is None and labels is None:
            return self.__array_to_image(image)

        # Dilate mask to avoid unmasked edge effect
        if dilate_kernel_size > 0:
            masks = [self.__dilate_mask(mask, dilate_kernel_size) for mask in masks]

        # Loop over masks and do in-painting for each selected label
        for mask, label in zip(masks, labels):
            image = self.lama.inpaint(image, mask)

        # return final result
        return self.__array_to_image(image)
