'''
Reasons why the code was breaking originally
1. Detectron2 takes in COCO format but outputs a different format, thus (xmin, ymin, width, height) had to be changed to (xmin, ymin, xmax, ymax)
(NOTE: The TridentNet predictor code has been updated to output (xmin, ymin, width, height) in accordance with COCO format.)
2. Appending to predictions does not work as it simply adds to otiginal predictions
3. .copy is a shallow copy, while .deepcopy makes copying process recursive to fix 2. so that the offset will work back into original image

'''

import copy
from pathlib import Path

from tqdm import tqdm
import cv2

import real_time_bus_segmenting_script as rtbss
from tridentnet_predictor import TridentNetPredictor

def calculate_iou(box1, box2):
    '''
    Calculate the IOU (intersection over union) of two bounding boxes.
    Inputs:
    - box1 and box2: dict containing {"bbox": [xmin, ymin, width, height]}
    Outputs:
    - IOU of the two boxes (should be a float between 0 and 1)
    '''
    box_1_x_min = box1['bbox'][0]
    box_1_y_min = box1['bbox'][1]
    box_1_x_max = box_1_x_min + box1['bbox'][2]
    box_1_y_max = box_1_y_min + box1['bbox'][3]

    box_2_x_min = box2['bbox'][0]
    box_2_y_min = box2['bbox'][1]
    box_2_x_max = box_2_x_min + box2['bbox'][2]
    box_2_y_max = box_2_y_min + box2['bbox'][3]    

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

def custom_nms(bounding_boxes, iou_threshold = 0.5):
    '''
    Non-Maximal Suppression (NMS) of bounding boxes.
    Input:
    - bounding_boxes: a list of dicts containing "score" as a key.
    - iou_threshold: the threshold above which similar boxes will be suppressed
    Output:
    - selected_boxes: a list of dicts containing only the "best" bounding boxes.
    '''
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

def annotating(cv2_image, predictor, segment_size = 640, iou_threshold = 0.5, display = True):    
    rtbss_obj = rtbss.ImageProcessor(cv2_image)
    rtbss_obj.segment_image(segment_size=segment_size)
    segment_images_info = rtbss_obj.get_segment_cv2_info()
    bounding_boxes = []

    for segment, segment_cv2_info in tqdm(segment_images_info.items()):

        x_offset = int(segment.split("_")[2])
        y_offset = int(segment.split("_")[1])

        predictions = predictor(segment_cv2_info)

        temp_bbox = []

        if predictions:
            for prediction in predictions:
                # add the non-offset predictions for display
                if display:
                    temp_bbox.append(prediction)

                # make a copy of the prediction so adding offsets won't affect the temp bbox (which is used for display)
                orig_prediction = copy.deepcopy(prediction)
                orig_prediction["bbox"][0] += x_offset
                orig_prediction["bbox"][1] += y_offset

                bounding_boxes.append(orig_prediction)

        # show the annotations on this segment
        if display:
            draw_annotations(segment_cv2_info, temp_bbox)

    non_overlapping_boxes = custom_nms(bounding_boxes, iou_threshold)

    return bounding_boxes, non_overlapping_boxes

def draw_annotations(cv2_image, list_of_predictions):
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

    cv2.namedWindow('Annotated Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Annotated Image', 900, 900)
    cv2.imshow("Annotated Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_predictions_vs_results(cv2_image, list_of_predictions: list, list_of_NMSed_bbox: list):
    annotated_image = cv2_image.copy()
    nms_annotated_image = cv2_image.copy()

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
        cv2.imwrite("prediction.jpg", annotated_image)

    for prediction in list_of_NMSed_bbox:
        bbox = prediction['bbox']
        score = prediction['score']
        pred_class = prediction['pred_class']

        xmin, ymin, width, height = bbox
        xmax = xmin + width
        ymax = ymin + height

        cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 4)
        cv2.putText(annotated_image, f"Class: {pred_class}", (int(xmin)-4, int(ymin)-14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(annotated_image, f"Score: {score:.2f}", (int(xmin)-4, int(ymin)-44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(nms_annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 4)
        cv2.putText(nms_annotated_image, f"Class: {pred_class}", (int(xmin)-4, int(ymin)-14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(nms_annotated_image, f"Score: {score:.2f}", (int(xmin)-4, int(ymin)-44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite("nmsed.jpg", nms_annotated_image)

    cv2.imwrite("example.jpg", annotated_image)
    open_image(r"C:\alp\HTX\busxray-detector-main\example.jpg")
    print("done")

def open_image(image_path):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    image = mpimg.imread(image_path)
    _, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    plt.show()

if __name__ == "__main__":
    img_path = Path(r"C:\alp\HTX\test images\3dogs.jpg")
    cv2_image = cv2.imread(str(img_path))

    predictor = TridentNetPredictor(config_file="models/tridentnet_fast_R_50_C4_3x.yaml", opts=["MODEL.WEIGHTS", "models/model_final_e1027c.pkl"])
    predictions, nmsed_predictions = annotating(cv2_image, predictor, segment_size=1800, display=True)
    draw_predictions_vs_results(cv2_image, predictions, nmsed_predictions)
