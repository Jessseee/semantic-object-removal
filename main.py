import sys
import argparse

import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from maskformer_segment import segment_with_maskformer
from lama_inpaint import inpaint_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, show_mask


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--labels", type=str, required=True, nargs='+',
        help="The labels of objects to remove",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=15,
        help="Dilate kernel size.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--save_masks", type=bool, default=False,
        help="Whether to save the masks to image files."
    )
    parser.add_argument(
        "--lama_config", type=str, default="./lama_config.yaml",
        help="The path to the config file of lama model.",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, default="./pretrained_models/big-lama",
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--maskformer_ckpt", type=str, default="facebook/maskformer-swin-large-ade",
        help="The path to the maskformer checkpoint."
    )
    parser.add_argument(
        "--label_file", type=str, default="./labels.json",
        help="The json file containing the output labels of the maskformer model"
    )


def save_masked_image(img, mask, img_mask_p):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    plt.imshow(img)
    show_mask(plt.gca(), mask, random_color=False)
    plt.axis('off')
    plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_img_to_array(args.input_img)

    masks, labels = segment_with_maskformer(img, args.maskformer_ckpt, args.labels, args.label_file)

    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir)
    if args.save_masks: out_dir = out_dir / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loop over masks and do in-painting for each selected label
    for mask, label in zip(masks, labels):
        if args.save_masks:
            # save the mask
            mask_p = out_dir / f"mask_{label}.png"
            save_array_to_img(mask, mask_p)

            # save the masked image
            img_mask_p = out_dir / f"with_mask_{label}.png"
            save_masked_image(img, mask, img_mask_p)

        # Inpaint mask and save image
        img_inpainted = inpaint_with_lama(img, mask, args.lama_config, args.lama_ckpt, device=device)
        img = img_inpainted

    img_final_p = out_dir / f"{img_stem}.png"
    save_array_to_img(img, img_final_p)
