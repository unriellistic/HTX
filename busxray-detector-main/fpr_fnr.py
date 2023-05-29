from typing import Sequence

import orjson
from globox import AnnotationSet, Annotation, BoundingBox, BoxFormat

CLASS_NAMES = ["cig", "guns", "human", "knives", "drugs", "exp"]

'''
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
'''

gt_anno_set = AnnotationSet.from_coco("coco_opstrial.json")
with open("pred.json", "rb") as f:
    pred_dict = orjson.loads(f.read())

pred_annotations = []
for filename, raw_annos in pred_dict.items():
    print(f"Checking annotations for {filename}.")

    anno_bboxes = [BoundingBox.create(
        label=CLASS_NAMES[raw_anno["pred_class"]],
        coords=raw_anno["bbox"],
        confidence=raw_anno["score"],
        box_format=BoxFormat.LTWH
    ) for raw_anno in raw_annos]

    pred_annotations.append(Annotation(
        image_id=filename,
        boxes=anno_bboxes,
    ))

pred_anno_set = AnnotationSet(pred_annotations)

