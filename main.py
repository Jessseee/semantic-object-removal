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
        "--sam_model_type", type=str, default="vit_h",
        choices=['vit_h', 'vit_l', 'vit_b'],
        help="The type of sam model to load."
    )
    parser.add_argument(
        "--lama_config", type=str, default="./configs/lama/prediction/default.yaml",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_img_to_array(args.input_img)

    masks = segment_with_maskformer(img, args.maskformer_ckpt, args.labels)
    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the mask
        save_array_to_img(mask, mask_p)

        # save the masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Inpaint mask
        mask_p = out_dir / f"mask_{idx}.png"
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        img_inpainted = inpaint_with_lama(img, mask, args.lama_config, args.lama_ckpt, device=device)
        save_array_to_img(img_inpainted, img_inpainted_p)
        img = img_inpainted
