import os
from pathlib import Path

from PIL import Image


def package_path(path: str | os.PathLike) -> bytes:
    """
    Get path relative to package root.
    :param path: relative path.
    :return: absolute path in package root.
    """
    package_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    return os.path.join(package_directory, path)


def save_image(image: Image, output_dir: str | os.PathLike, output_type: str):
    """
    Save image to path.
    :param image: image object
    :param output_dir: path to output directory
    :param output_type: image file extension
    """
    img_stem = Path(image.filename).stem
    output_path = os.path.join(output_dir, f"{img_stem}{output_type}")
    image.save(output_path)
