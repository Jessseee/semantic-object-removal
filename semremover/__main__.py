import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

from .models import SemanticObjectRemover


def package_path(path: str | os.PathLike):
    package_directory = os.path.dirname(os.path.abspath(__file__))
    new_path = os.path.join(package_directory, path)
    return new_path if os.path.exists(new_path) else path


def save_image(image: Image, image_path: str | os.PathLike, output_dir: str | os.PathLike, output_type: str):
    img_stem = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{img_stem}{output_type}")
    image.save(output_path)


def create_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "input", type=str, default=None, nargs='+',
        help="Path to input image(s).",
    )
    parser.add_argument(
        "-l", "--labels", type=str, required=True, nargs='+',
        help="The labels of objects to remove.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="results",
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--output_type", type=str, default=".jpg",
        choices=[ext for ext, format in Image.registered_extensions().items() if format in Image.OPEN],
        help="The output image type suffix."
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=15,
        help="Dilate kernel size.",
    )
    parser.add_argument(
        "--label_file", type=str, default=package_path("models/config/ade20k_labels.json"),
        help="The json file containing the output labels of the maskformer model"
    )
    parser.add_argument(
        "--lama_ckpt", type=str, default=package_path("models/weights/big-lama"),
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--lama_config", type=str, default=package_path("models/config/lama_default.yaml"),
        help="The path to the config file of lama model.",
    )
    parser.add_argument(
        "--maskformer_ckpt", type=str, default="facebook/maskformer-swin-large-ade",
        help="The path to the maskformer checkpoint."
    )
    return parser


parser = create_parser()
args = parser.parse_args(sys.argv[1:])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for object removal.")

output_dir = Path(args.output_dir)
if output_dir.exists() and output_dir.is_file(): raise IOError(f"Output directory {output_dir} is a file.")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Removing the following objects: {args.labels}")

sem_obj_remover = SemanticObjectRemover(args.lama_ckpt, args.lama_config, args.maskformer_ckpt, args.label_file)

for input_path in args.input:
    if not os.path.exists(input_path):
        print(f"{input_path} does not exist.")
        continue
    if os.path.isdir(input_path):
        print(f"{input_path}/")
        processed = [os.path.splitext(entry.name)[0] for entry in os.scandir(output_dir) if entry.is_file()]
        image_paths = [entry.path for entry in os.scandir(input_path) if entry.is_file()
                       and os.path.splitext(entry.name)[0] not in processed]
        if len(image_paths) == 0:
            print(f"There are no (unprocessed) files in {input_path}.")
            continue
        for image_path in tqdm(image_paths):
            inpainted_image = sem_obj_remover.remove_objects_from_image(image_path, args.labels, args.dilate_kernel_size)
            save_image(inpainted_image, image_path, args.output_dir, args.output_type)
    else:
        print(input_path)
        inpainted_image = sem_obj_remover.remove_objects_from_image(input_path, args.labels, args.dilate_kernel_size)
        save_image(inpainted_image, input_path, args.output_dir, args.output_type)
