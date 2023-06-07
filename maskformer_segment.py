import json

from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
from torch import index_select, tensor
import numpy as np


def segment_with_maskformer(image, maskformer_ckpt, to_select, label_file):
    labels = json.load(open(label_file, "r"))
    maskformer = MaskFormerForInstanceSegmentation.from_pretrained(maskformer_ckpt)
    processor = MaskFormerImageProcessor()
    inputs = processor(images=image, return_tensors="pt")
    outputs = maskformer(**inputs)
    batch_size = outputs.class_queries_logits.shape[0]
    _outputs = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[image.shape[:2] for _ in range(batch_size)],
        return_binary_maps=True
    )[0]
    mask_selected = []
    mask_labels = []
    for item in _outputs['segments_info']:
        label = labels[str(item['label_id'])]
        if label in to_select:
            print(f"{label}: {item}")
            mask_selected.append(item['id'])
            mask_labels.append(label)
    masks = index_select(_outputs['segmentation'], 0, tensor(mask_selected)).numpy().astype(np.uint8) * 255
    return masks, mask_labels
