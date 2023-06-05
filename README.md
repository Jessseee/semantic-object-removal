# Dynamic Noise Removal
Segmentation and in-painting inspired by [Inpaint Anything](https://github.com/geekyutao/Inpaint-Anything) by  Tao Yu et al.

## Installation
Install the Python requirements.
```
python -m pip install -r requirements.txt
```
Be sure to download the model weights for [LaMa](https://github.com/advimman/lama) (e.g., [big-lama](https://disk.yandex.ru/d/ouP6l8VJ0HpMZg)). Download the files and put them into `./pretrained_models`.

## Usage
To use the script you can call it with various options of which `input_img` is the only required parameter without default value. To get a window to click on an object to remove use the `coords_type` parameter with the value `click`.
```
    python main.py --input_img <image> --labels car person
```
