import sys
import argparse
import os

import torch
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL.Image import registered_extensions, OPEN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from maskformer import MaskFormer
from lama import LaMa
from utils import load_img_to_array, save_array_to_img, dilate_mask, show_mask


def setup_args(parser):
    parser.add_argument(
        "input",  type=str, default=None,
        help="Path to a single input image",
    )
    parser.add_argument(
        "-l", "--labels", type=str, required=True, nargs='+',
        help="The labels of objects to remove",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="./results",
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--img_suffix", type=str, default="jpg",
        choices=[ext for ext, format in registered_extensions().items() if format in OPEN],
        help="The output image type suffix."
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=15,
        help="Dilate kernel size.",
    )
    parser.add_argument(
        "--save_masks", type=bool, default=False,
        help="Whether to save the masks to image files."
    )
    parser.add_argument(
        "--lama_ckpt", type=str, default="./pretrained_models/big-lama",
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--lama_config", type=str, default="./lama_config.yaml",
        help="The path to the config file of lama model.",
    )
    parser.add_argument(
        "--maskformer_ckpt", type=str, default="facebook/maskformer-swin-large-ade",
        help="The path to the maskformer checkpoint."
    )
    parser.add_argument(
        "--label_file", type=str, default="./labels.json",
        help="The json file containing the output labels of the maskformer model"
    )


def save_masked_image(image: np.ndarray, mask: np.ndarray, img_mask_path: str | os.PathLike):
    dpi = plt.rcParams['figure.dpi']
    height, width = image.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    plt.imshow(image)
    show_mask(plt.gca(), mask, random_color=False)
    plt.axis('off')
    plt.savefig(img_mask_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def remove_objects_from_image(maskformer: MaskFormer, lama: LaMa, input_path: str | os.PathLike,
                              output_dir: str | os.PathLike, img_suffix: str, dilate_kernel_size: int):
    image = load_img_to_array(input_path)
    masks, labels = maskformer.segment(image, args.labels)

    out_dir = Path(output_dir)
    img_stem = Path(input_path).stem
    if args.save_masks: out_dir = out_dir / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save original image if no objects were found to remove
    if masks is None and labels is None:
        img_final_path = out_dir / f"{img_stem}.{img_suffix}"
        save_array_to_img(image, img_final_path)
        return

    # Dilate mask to avoid unmasked edge effect
    if dilate_kernel_size > 0:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # Loop over masks and do in-painting for each selected label
    for mask, label in zip(masks, labels):
        if args.save_masks:
            # Save the mask
            mask_path = out_dir / f"mask_{label}{img_suffix}"
            save_array_to_img(mask, mask_path)

            # Save the masked image
            img_mask_path = out_dir / f"with_mask_{label}{img_suffix}"
            save_masked_image(image, mask, img_mask_path)

        # Inpaint mask and save image
        img_inpainted = lama.inpaint(image, mask)
        image = img_inpainted

    # Save final result
    img_final_path = out_dir / f"{img_stem}{img_suffix}"
    save_array_to_img(image, img_final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    maskformer = MaskFormer(args.maskformer_ckpt, args.label_file)
    lama = LaMa(args.lama_ckpt, args.lama_config)

    if not os.path.exists(args.input):
        raise IOError(f"{args.input} does not exist.")
    if os.path.isdir(args.input):
        processed = [os.path.splitext(file)[0] for file in os.listdir(args.output_dir)]
        images = [entry.path for entry in os.scandir(args.input) if entry.is_file()
                  and os.path.splitext(entry.name)[0] not in processed]
        if len(images) == 0:
            raise IOError(f"There are no files in {args.input}.")
        for image in tqdm(images):
            remove_objects_from_image(maskformer, lama, image, args.output_dir, args.img_suffix, args.dilate_kernel_size)
    else:
        remove_objects_from_image(maskformer, lama, args.input, args.output_dir, args.img_suffix, args.dilate_kernel_size)
