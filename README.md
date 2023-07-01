# Semantic Object Removal
Using semantic segmentation and in-painting to remove objects based on labels. Inspired by [Inpaint Anything](https://github.com/geekyutao/Inpaint-Anything) by  Tao Yu et al. Using [MaskFormer](https://github.com/facebookresearch/MaskFormer) for semantic segmentation to select areas to remove using [LaMa](https://github.com/advimman/lama) for in-painting.

## Installation
Install the package.
```
python -m pip install semremover
```

Use the `SemanticObjectRemover` in your code.
``` python
from semremover import SemanticObjectRemover

sem_obj_remover = SemanticObjectRemover()
labels = ['car', 'minibike', 'van']
inpainted_image = sem_obj_remover.remove_objects_from_image("example.jpg", labels)
```

## Development

### Installation
Install the Python requirements.
```
python -m pip install -r requirements.txt
```

### Usage
To use the script you can call it with various options. The first positional argument is the input path, which can point to either an image or a directory of images. To remove objects from a picture add them to the labels option when running the script. The default labels can be found in `./semremover/models/config/ade20k_labels.json`.

### Example
**input**
```
python -m semremover example/paris.jpg --labels car minibike van
```

![A picture of a street lined with cars in Paris.](https://github.com/Jessseee/semantic-object-removal/blob/main/example/paris.jpg?raw=true)

**Output**

![A picture of the same street in paris with the cars digitally removed.](https://github.com/Jessseee/semantic-object-removal/blob/main/example/paris-inpainted.jpg?raw=true)
