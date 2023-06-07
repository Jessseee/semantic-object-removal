# Dynamic Object Removal
Segmentation and in-painting inspired by [Inpaint Anything](https://github.com/geekyutao/Inpaint-Anything) by  Tao Yu et al. Using [MaskFormer](https://github.com/facebookresearch/MaskFormer) for semantic segmentation to select areas to remove using [LaMa](https://github.com/advimman/lama) for in-painting.

## Installation
Install the Python requirements.
```
python -m pip install -r requirements.txt
```
Be sure to download the model weights for [LaMa](https://github.com/advimman/lama) (e.g., [big-lama](https://disk.yandex.ru/d/ouP6l8VJ0HpMZg)). Download the directory and put them into `./pretrained_models`.

## Usage
To use the script you can call it with various options of which `input_img` is the only required parameter without default value. To remove objects from a picture add them to the labels option when running the script. The labels that are available can be found in `labels.json`.

## Example
**input**
```
    python main.py --input_img example/paris.jpg --labels car minibike van
```

![A picture of a street lined with cars in Paris.](example/paris.jpg)

**output**

![A picture of the same street in paris with the cars digitally removed.](example/paris-inpainted.jpg)
