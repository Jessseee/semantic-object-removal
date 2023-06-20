from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    package_data={'semremover': ["models/*/*.json, models/*/*.yaml"]},
    install_required=[
        "LaMa @ git+https://github.com/jessseee/lama.git",
        "torch~=2.0",
        "transformers~=4.30",
        "omegaconf~=2.3",
        "opencv-python",
        "numpy",
        "pillow",
        "argparse",
        "tqdm",
    ]
)
