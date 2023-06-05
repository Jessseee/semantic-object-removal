from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
from torch import index_select, tensor

from utils import id_to_label


def segment_with_maskformer(image, maskformer_ckpt, labels):
    maskformer = MaskFormerForInstanceSegmentation.from_pretrained(maskformer_ckpt)
    processor = MaskFormerImageProcessor()
    inputs = processor(images=image, return_tensors="pt")
    outputs = maskformer(**inputs)
    batch_size = outputs.class_queries_logits.shape[0]
    _outputs = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[(360, 640) for _ in range(batch_size)],
        return_binary_maps=True
    )[0]
    index = []
    for item in _outputs['segments_info']:
        label = id_to_label(item['label_id'])
        if label in labels:
            print(item, label)
            index.append(item['id'])
    return index_select(_outputs['segmentation'], 0, tensor(index)).numpy()
