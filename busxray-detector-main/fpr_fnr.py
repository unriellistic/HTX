import orjson
from globox import AnnotationSet, COCOEvaluator

from tridentnet_to_annotation_set import tridentnet_to_annotation_set

'''
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
'''

gt_anno_set = AnnotationSet.from_coco("coco_opstrial.json")
with open("pred.json", "rb") as f:
    pred_dict = orjson.loads(f.read())

pred_anno_set = tridentnet_to_annotation_set(pred_dict)

evaluator = COCOEvaluator(
    ground_truths=gt_anno_set,
    predictions=pred_anno_set
)

evaluator.show_summary()