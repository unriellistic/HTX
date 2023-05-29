from globox import AnnotationSet, Annotation, BoundingBox, BoxFormat

CLASS_NAMES = ["cig", "guns", "human", "knives", "drugs", "exp"]

def tridentnet_to_annotation_set(pred_dict: dict[str, list]) -> AnnotationSet:
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

    return AnnotationSet(pred_annotations)