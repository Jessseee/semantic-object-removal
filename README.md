# Dynamic Noise Removal
Segmentation and in-painting heavily based on [Inpaint Anything](https://github.com/geekyutao/Inpaint-Anything) by  Tao Yu et al.

## Installation
Install the Python requirements.
```
python -m pip install -r requirements.txt
```
Pretrained model checkpoints are gracefully provided by Tao Yu for [Segment Anything](https://github.com/facebookresearch/segment-anything) and [LaMa](https://github.com/advimman/lama) (e.g., [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and [big-lama](https://disk.yandex.ru/d/ouP6l8VJ0HpMZg)). Download the files and put them into `./pretrained_models`. 

For simplicity, you can also directly download a zip file of the `pretrained_models` directory from this [Google Drive](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing).

