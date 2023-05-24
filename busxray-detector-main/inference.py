"""
Takes in a cv2 image, calls segmentation function on it, and returns back the cleaned NMS annotations.

Note: Currently the dict values to look for are tailored for tridentnet output. YOLO may be a different set of dict values to access.
"""
from pathlib import Path
from matplotlib.pyplot import box
from tqdm import tqdm
import cv2
import real_time_bus_segmenting_script as rtbss
import copy
from typing import Union, Tuple, List, Dict, Any

def calculate_iou(box1: Union[Tuple, List], box2: Union[Tuple, List]) -> float:
    """
    Calculate IoU (Intersection over Union) between two bounding boxes in the format of (xmin, ymin, width, height).

    Args:
        box1: Tuple or list representing the first bounding box (xmin, ymin, width, height).
        box2: Tuple or list representing the second bounding box (xmin, ymin, width, height).

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

def custom_nms(bounding_boxes: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    """
    Perform a customized non-maximal suppression (NMS)-like algorithm to merge overlapping bounding boxes.

    Args:
        bounding_boxes: A list of dictionaries containing "score" as a key.
        iou_threshold: IoU threshold to determine overlap.

    Returns:
        selected_boxes: List of non-overlapping bounding boxes.
    """
    # Sort bounding boxes by score in descending order
    sorted_boxes = sorted(bounding_boxes, key=lambda x: x["score"], reverse=True)

    selected_boxes = []
    while sorted_boxes:
        # Pop the box with the highest score from the sorted list
        current_box = sorted_boxes.pop(0)
        selected_boxes.append(current_box)

        remaining_boxes = []
        for box in sorted_boxes:
            # Calculate IoU between the current box and other boxes
            iou = calculate_iou(current_box["bbox"], box["bbox"])

            # If IoU is below the threshold, keep the box for further consideration
            if iou <= iou_threshold:
                remaining_boxes.append(box)

        sorted_boxes = remaining_boxes

    return selected_boxes

def inference(cv2_image: Any, predictor: Any, segment_size: int, crop_image: bool, IOU_THRESHOLD: float = 0.3,
              display: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Perform inference on segmented images, and apply non-maximal suppression.

    Args:
        cv2_image: The input image in a cv2.imread format.
        predictor: The AI predictor function.
        segment_size: The size of each segment for image processing.
        crop_image: Flag indicating whether to crop the image.
        IOU_THRESHOLD: IoU threshold for non-maximal suppression (default: 0.3).
        display: Flag indicating whether to display intermediate results (default: False).

    Returns:
        bounding_boxes: Lists containing bounding boxes before non-maximal suppression.
        non_overlapping_boxes: Lists containing bounding boxes after non-maximal suppression.
    """
    # Crop and segment the image
    rtbss_obj = rtbss.ImageProcessor(cv2_image)
    if crop_image:
        rtbss_obj.crop_image()
    rtbss_obj.segment_image(segment_size=segment_size)
    segment_images_info = rtbss_obj.get_segment_cv2_info()

    # Initialize empty list of bounding boxes
    bounding_boxes = []

    # Process each segment
    for segment, segment_cv2_info in tqdm(segment_images_info.items()):
        # Get original coordinates from segment name
        x_offset = int(segment.split("_")[2])
        y_offset = int(segment.split("_")[1])

        # Run AI prediction
        predictions = predictor(segment_cv2_info)  # Should be COCO format (json compatible)
        temp_bbox = []

        # Convert prediction into a standardised format
        cleaned_predictions = []
        
        for prediction in predictions:
            cleaned_prediction = {"bbox": find_list_value(prediction),
                               "score": find_float_value(prediction),
                               "pred_class": find_int_value(prediction)}
            cleaned_predictions.append(cleaned_prediction)

        # If predictions were made
        if cleaned_predictions:
            # Iterate through each prediction
            for prediction in cleaned_predictions:
                if display:
                    orig_prediction = copy.deepcopy(prediction)
                    temp_bbox.append(orig_prediction)

                # Calculate original coordinates
                prediction["bbox"][0] += x_offset
                prediction["bbox"][1] += y_offset

                # Update list of bounding boxes in this format: 
                """
                prediction = {
                    "bbox": [xmin, ymin, width, height], 
                    "score": float, 
                    "pred_class": int
                }
                """
                bounding_boxes.append(prediction)

        # Display intermediate results for debugging
        if display:
            draw_annotations(segment_cv2_info, temp_bbox)

    # Perform non-maximal suppression
    non_overlapping_boxes = custom_nms(bounding_boxes, IOU_THRESHOLD)

    return bounding_boxes, non_overlapping_boxes

def find_float_value(dictionary: Dict[str, Union[float, int, List]]) -> Union[float, None]:
    """
    Find a float value in the dictionary and return it.

    Args:
        dictionary: The input dictionary.

    Returns:
        The float value found in the dictionary, or None if no float value is found.
    """
    for value in dictionary.values():
        if isinstance(value, float):
            return value
    return None

def find_int_value(dictionary: Dict[str, Union[float, int, List]]) -> Union[int, None]:
    """
    Find a float value in the dictionary and return it.

    Args:
        dictionary: The input dictionary.

    Returns:
        The float value found in the dictionary, or None if no float value is found.
    """
    for value in dictionary.values():
        if isinstance(value, int):
            return value
    return None

def find_list_value(dictionary: Dict[str, Union[float, int, List]]) -> Union[List, None]:
    """
    Find a float value in the dictionary and return it.

    Args:
        dictionary: The input dictionary.

    Returns:
        The float value found in the dictionary, or None if no float value is found.
    """
    for value in dictionary.values():
        if isinstance(value, List):
            return value
    return None

def draw_annotations(cv2_image, list_of_predictions) -> None:
    annotated_image = cv2_image.copy()
    for prediction in list_of_predictions:
        bbox = prediction['bbox']
        score = prediction['score']
        pred_class = prediction['pred_class']

        xmin, ymin, width, height = bbox
        xmax = xmin + width
        ymax = ymin + height
        cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 4)
        cv2.putText(annotated_image, f"Class: {pred_class}", (int(xmin)-4, int(ymin)-14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(annotated_image, f"Score: {score:.2f}", (int(xmin)-4, int(ymin)-44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the annotated image
    cv2.namedWindow('Annotated Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Annotated Image', cv2_image.shape[1], cv2_image.shape[0])
    cv2.imshow("Annotated Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_predictions_vs_results(cv2_image, list_of_predictions: list, list_of_NMSed_bbox: list) -> None:
    # Make a copy of the image to draw on
    annotated_image = cv2_image.copy()
    nms_annotated_image = cv2_image.copy()
    # Draw predictions (red)
    for prediction in list_of_predictions:
        bbox = prediction['bbox']
        score = prediction['score']
        pred_class = prediction['pred_class']

        xmin, ymin, width, height = bbox
        xmax = xmin + width
        ymax = ymin + height
        cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 4)
        cv2.putText(annotated_image, f"Class: {pred_class}", (int(xmin)-4, int(ymin)-14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(annotated_image, f"Score: {score:.2f}", (int(xmin)-4, int(ymin)-44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(r"busxray-detector-main\prediction.jpg", annotated_image)

    # Draw suppressed predictions (green)
    for prediction in list_of_NMSed_bbox:
        bbox = prediction['bbox']
        score = prediction['score']
        pred_class = prediction['pred_class']

        xmin, ymin, width, height = bbox
        xmax = xmin + width
        ymax = ymin + height
        # Draw for original predictions image
        cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
        cv2.putText(annotated_image, f"Class: {pred_class}", (int(xmin)-4, int(ymin)-14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(annotated_image, f"Score: {score:.2f}", (int(xmin)-4, int(ymin)-44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # Draw for NMS suppression image
        cv2.rectangle(nms_annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
        cv2.putText(nms_annotated_image, f"Class: {pred_class}", (int(xmin)-4, int(ymin)-14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(nms_annotated_image, f"Score: {score:.2f}", (int(xmin)-4, int(ymin)-44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite(r"busxray-detector-main\nmsed.jpg", nms_annotated_image)

    cv2.imwrite(r"busxray-detector-main\example2.jpg", annotated_image)
    open_image(r"busxray-detector-main\example2.jpg")
    print("Done")

def open_image(image_path):
    # To find out roughly what's the pixel intensity and at which X,Y coordinates.
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    # Load image from file
    image = mpimg.imread(image_path)
    # Create Matplotlib figure and axes
    _, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    # Show Matplotlib window
    plt.show()

if __name__ == "__main__":
    print("This script is not meant to be run directly. Use busxray_server.py instead.")
    print("Please import it as a module and call the inference() function, unless you're debugging.")
    import os
    # Get image
    img_path = Path(r"test images\manydogs.jpg")
    assert os.path.exists(img_path), f"Path '{img_path}' does not exist."
    cv2_image = cv2.imread(str(img_path))
    from tridentnet_predictor import TridentNetPredictor
    
    path_to_model = r"busxray-detector-main\models\tridentnet_fast_R_50_C4_3x.yaml"
    path_to_model_weight = r"busxray-detector-main\models\model_final_e1027c.pkl"
    assert os.path.exists(path_to_model), f"Path '{path_to_model}' does not exist."
    assert os.path.exists(path_to_model_weight), f"Path '{path_to_model_weight}' does not exist."
    # change this if you want to use a different model
    # predictor should be callable, take openCV-format image as input, and output JSON-compatible predictions
    predictor = TridentNetPredictor(config_file=path_to_model,
        opts=["MODEL.WEIGHTS", path_to_model_weight]
    )
    segment_size = 4000
    crop_image = False
    predictions, nmsed_bbox = inference(cv2_image, predictor, segment_size, crop_image, IOU_THRESHOLD=0.3, display=False)
    draw_predictions_vs_results(cv2_image, predictions, nmsed_bbox)


    