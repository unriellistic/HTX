"""
Just a mini testing file for me to try out python logic for debugging.
"""

data = {
  "predictions": [
    {
      "class": "person",
      "confidence": 0.92,
      "bounding_box": {
        "x": 100,
        "y": 200,
        "width": 300,
        "height": 400
      }
    },
    {
      "class": "car",
      "confidence": 0.85,
      "bounding_box": {
        "x": 400,
        "y": 150,
        "width": 200,
        "height": 250
      }
    },
    ...
  ]
}

# Run on every segment

# Initialise empty list of bounding_boxes
bounding_boxes = []
  # Know which segment we're running prediction on
  # run prediction
  # calculate original position
  # Update list of bounding_boxes -> Contains a list of original adjusted [xmin,xmax,ymin,ymax,confidence]

# Sort it according to confidence score
sorted_boxes = sorted(bounding_boxes, key=lambda x: x[4], reverse=True)  # Sort by confidence
# Perform NMS here. Limitations: We only stitch together same class id prediction together



# # If current location has another label, perform non-maximal suppression. (What if there are 2 distinct items in overlap? Maybe a gun overlap with a cig?)

def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.

    Args:
        box1: Tuple or list representing the first bounding box (x_min, y_min, x_max, y_max).
        box2: Tuple or list representing the second bounding box (x_min, y_min, x_max, y_max).

    Returns:
        IoU value.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union = box1_area + box2_area - intersection

    iou = intersection / union

    return iou

def custom_nms(bounding_boxes, iou_threshold):
    """
    Perform a customized non-maximal suppression (NMS)-like algorithm to merge overlapping bounding boxes.

    Args:
        bounding_boxes: List of bounding boxes [(x_min, y_min, x_max, y_max, confidence, class_id)].
        iou_threshold: IoU threshold to determine overlap.

    Returns:
        List of non-overlapping bounding boxes.
    """
    sorted_boxes = sorted(bounding_boxes, key=lambda x: x[4], reverse=True)  # Sort by confidence

    selected_boxes = []
    while sorted_boxes:
        current_box = sorted_boxes.pop(0)
        selected_boxes.append(current_box)

        remaining_boxes = []
        for box in sorted_boxes:
            iou = calculate_iou(current_box, box)
            if iou <= iou_threshold:
                remaining_boxes.append(box)

        sorted_boxes = remaining_boxes

    return selected_boxes

# Assuming you have a list of bounding boxes: [(x_min, y_min, x_max, y_max, confidence, class_id)]
bounding_boxes = 0.5
non_overlapping_boxes = custom_nms(bounding_boxes, iou_threshold)

# Print the resulting non-overlapping bounding boxes
for box in non_overlapping_boxes:
    print(box) = [(10, 20, 50, 60, 0.9, 1), (30, 40, 70, 80, 0.8, 2), (40, 55, 80, 90, 0.9, 3), (15, 20, 80, 90, 0.7, 3)]

# Call the custom NMS function
iou_threshold
