from typing import Sequence

from globox import AnnotationSet

def calculate_iou(box1: Sequence, box2: Sequence) -> float:
    """
    Calculate IoU (Intersection over Union) between two bounding boxes in the format of (xmin, ymin, width, height).

    Args:
        box1: Sequence representing the first bounding box (xmin, ymin, width, height).
        box2: Sequence representing the second bounding box (xmin, ymin, width, height).

    Returns:
        IoU value (should be a float between 0 and 1).
    """
    # Extract coordinates of box1
    box_1_x_min, box_1_y_min, box_1_width, box_1_height = box1
    box_1_x_max = box_1_x_min + box_1_width
    box_1_y_max = box_1_y_min + box_1_height

    # Extract coordinates of box2
    box_2_x_min, box_2_y_min, box_2_width, box_2_height = box2
    box_2_x_max = box_2_x_min + box_2_width
    box_2_y_max = box_2_y_min + box_2_height

    # Calculate intersection area
    x1 = max(box_1_x_min, box_2_x_min)
    y1 = max(box_1_y_min, box_2_y_min)
    x2 = min(box_1_x_max, box_2_x_max)
    y2 = min(box_1_y_max, box_2_y_max)
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate box areas
    box1_area = (box_1_x_max - box_1_x_min + 1) * (box_1_y_max - box_1_y_min + 1)
    box2_area = (box_2_x_max - box_2_x_min + 1) * (box_2_y_max - box_2_y_min + 1)

    # Calculate union area
    union = box1_area + box2_area - intersection

    # Calculate IoU
    iou = intersection / union
    return iou


coco = AnnotationSet.from_coco("coco_opstrial.json")
for annotation in coco:
    print(f"{annotation.image_id}: {len(annotation.boxes)} boxes")