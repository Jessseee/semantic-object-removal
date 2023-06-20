import json
import os

from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
from torch import index_select, tensor
import numpy as np


class MaskFormer:
    def __init__(self, ckpt: str | os.PathLike, label_file: str | os.PathLike):
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(ckpt)
        self.processor = MaskFormerImageProcessor()
        self.labels = json.load(open(label_file, "r"))

    def segment(self, image: np.ndarray, to_select: list[str]):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        batch_size = outputs.class_queries_logits.shape[0]
        outputs = self.processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[image.shape[:2] for _ in range(batch_size)],
            return_binary_maps=True
        )[0]

        mask_selected = []
        mask_labels = []
        for item in outputs['segments_info']:
            label = self.labels[str(item['label_id'])]
            if label in to_select:
                mask_selected.append(item['id'])
                mask_labels.append(label)
        if len(mask_selected) == 0:
            return None, None
        masks = index_select(outputs['segmentation'], 0, tensor(mask_selected)).numpy().astype(np.uint8) * 255
        return masks, mask_labels
