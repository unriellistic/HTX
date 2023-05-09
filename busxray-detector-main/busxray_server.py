import aiofiles, time
from pathlib import Path
from tqdm import tqdm
import cv2, orjson
from sanic import Sanic
from sanic.exceptions import SanicException
from sanic.log import logger
from sanic.response import json
from sanic.worker.loader import AppLoader

from tridentnet_predictor import TridentNetPredictor
import real_time_bus_segmenting_script as rtbss


# change this if you want to use a different model
# predictor should be callable, take openCV-format image as input, and output JSON-compatible predictions
predictor = TridentNetPredictor(config_file="models/tridentnet_fast_R_50_C4_3x.yaml",
    opts=["MODEL.WEIGHTS", "models/model_final_e1027c.pkl"]
)
def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.

    Args:
        box1: Tuple or list representing the first bounding box [{  'bbox': [xmin, ymin, width, height],
                                                                    'score': score,
                                                                    'pred_class': id}].
        box2: Tuple or list representing the second bounding box (x_min, y_min, x_max, y_max).

    Returns:
        IoU value.
    """
    # reformat box to fit existing code that works
    box_1_x_min = box1['bbox'][0]
    box_1_y_min = box1['bbox'][1]
    box_1_x_max = box_1_x_min + box1['bbox'][2]
    box_1_y_max = box_1_y_min + box1['bbox'][3]
    box_2_x_min = box2['bbox'][0]
    box_2_y_min = box2['bbox'][1]
    box_2_x_max = box_2_x_min + box2['bbox'][2]
    box_2_y_max = box_2_y_min + box2['bbox'][3]

    # Existing code that works
    x1 = max(box_1_x_min, box_2_x_min)
    y1 = max(box_1_y_min, box_2_y_min)
    x2 = min(box_1_x_max, box_2_x_max)
    y2 = min(box_1_y_max, box_2_y_max)

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box_1_x_max - box_1_x_min + 1) * (box_1_y_max - box_1_y_min + 1)
    box2_area = (box_2_x_max - box_2_x_min + 1) * (box_2_y_max - box_2_y_min + 1)

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
    sorted_boxes = sorted(bounding_boxes, key=lambda x: x["score"], reverse=True) # Sort by confidence

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

# Get image
img_path = Path(r"C:\Users\User1\Desktop\alp\busxray-detector-main\PA8506K Higer 49 seats-Threat-3-final_color.jpg")
logger.info("Processing image " + str(img_path))
start_time = time.perf_counter()
img_cv2 = cv2.imread(str(img_path))

# Crop and segment image
_, segment_images_info = rtbss.calling_relevant_class_function(cv2_image=img_cv2, segment_size=640, overlap_percent=0.5)

# Initialise empty list of bounding_boxes
bounding_boxes = []

# Run on every segment
for segment, segment_info in tqdm(segment_images_info.items()):

    # Get original coordinates from segment name
    x_offset = int(segment.split("_")[2])
    y_offset = int(segment.split("_")[1])

    # run AI prediction
    predictions = predictor(segment_info) # should be COCO format (json compatible)

    # If prediction was made
    if predictions:
    
        # if predictions has multiple annotations return, iterate through each one
        for prediction in predictions:
            # calculate original coordinates
            prediction["bbox"][0] = prediction["bbox"][0] + x_offset
            prediction["bbox"][1] = prediction["bbox"][1] + y_offset

            # Update list of bounding_boxes -> Contains a list of original adjusted [xmin,xmax,ymin,ymax,confidence]
            bounding_boxes.append(prediction)

# Perform NMS here. Limitations: We only stitch together same class id prediction together
# Assuming you have a list of bounding boxes: [(bbox, score, pred_class)]
IOU_THRESHOLD = 0.5
non_overlapping_boxes = custom_nms(bounding_boxes, IOU_THRESHOLD)


# save the prediction to json file (not needed as this is done on client side)
# output_path = Path(app.config.OUTPUT_FOLDER) / Path(img.name).with_suffix(".json")
# with open(output_path, "wb") as f:
#     f.write(orjson.dumps(predictions, option=orjson.OPT_INDENT_2))
logger.info(f"...done. ({(time.perf_counter() - start_time):.3f} s)")

print(json(non_overlapping_boxes))


