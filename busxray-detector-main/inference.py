"""
Takes in a cv2 image, calls segmentation function on it, and returns back the cleaned NMS annotations.

Note: Currently the dict values to look for are tailored for tridentnet output. YOLO may be a different set of dict values to access.
"""

from tqdm import tqdm
import cv2

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

def annotating(cv2_image):
    import real_time_bus_segmenting_script as rtbss
    from tridentnet_predictor import TridentNetPredictor
    # change this if you want to use a different model
    # predictor should be callable, take openCV-format image as input, and output JSON-compatible predictions
    predictor = TridentNetPredictor(config_file="models/tridentnet_fast_R_50_C4_3x.yaml",
        opts=["MODEL.WEIGHTS", "models/model_final_e1027c.pkl"]
    )
    # Crop and segment image
    # temp code
    rtbss_obj = rtbss.ImageProcessor(cv2_image)
    rtbss_obj.segment_image(segment_size=3000)
    segment_images_info = rtbss_obj.get_segment_cv2_info()
    # _, segment_images_info = rtbss.calling_relevant_class_function(cv2_image=cv2_image, segment_size=640, overlap_percent=0.5)

    # Initialise empty list of bounding_boxes
    bounding_boxes = []

    # Run on every segment
    for segment, segment_cv2_info in tqdm(segment_images_info.items()):

        # Get original coordinates from segment name
        x_offset = int(segment.split("_")[2])
        y_offset = int(segment.split("_")[1])

        # run AI prediction
        predictions = predictor(segment_cv2_info) # should be COCO format (json compatible)

        """
        Perform inverse scaling
        """
        original_width = segment_cv2_info.shape[1]
        original_height = segment_cv2_info.shape[0]
        resized_width = 1333
        resized_height = 2000

        # Calculate scaling factors
        width_scale = original_width / resized_width
        height_scale = original_height / resized_height

        temp_bbox = []
        DISPLAY = True
        # If prediction was made
        if predictions:
            
            # if predictions has multiple annotations return, iterate through each one
            for prediction in predictions:
                
                # Update the bounding box coordinates in the prediction dictionary
                # Inverse scaling of bounding box coordinates
                # x, y, width, height = prediction['bbox']
                # x *= width_scale
                # y *= height_scale
                # width *= width_scale
                # height *= height_scale
                # prediction['bbox'] = [x, y, width, height]

                # temp code
                if DISPLAY:
                    temp_bbox.append(prediction)

                # calculate original coordinates
                prediction["bbox"][0] = prediction["bbox"][0] + x_offset
                prediction["bbox"][1] = prediction["bbox"][1] + y_offset

                # Update list of bounding_boxes -> Contains a list of original adjusted [xmin,xmax,ymin,ymax,confidence]
                bounding_boxes.append(prediction)
        # temp code
        if DISPLAY:
            draw_annotations(segment_cv2_info, temp_bbox)
    # Perform NMS here. Limitations: We only stitch together same class id prediction together
    # Assuming you have a list of bounding boxes: [(bbox, score, pred_class)]
    IOU_THRESHOLD = 0.5
    non_overlapping_boxes = custom_nms(bounding_boxes, IOU_THRESHOLD)

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

    # Display the annotated image
    cv2.namedWindow('Annotated Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Annotated Image', cv2_image.shape[1], cv2_image.shape[0])
    cv2.imshow("Annotated Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_predictions_vs_results(cv2_image, list_of_predictions: list, list_of_NMSed_bbox: list):
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
        cv2.imwrite("prediction.jpg", annotated_image)

    # Draw suppressed predictions (green)
    for prediction in list_of_NMSed_bbox:
        bbox = prediction['bbox']
        score = prediction['score']
        pred_class = prediction['pred_class']

        xmin, ymin, width, height = bbox
        xmax = xmin + width
        ymax = ymin + height
        cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(annotated_image, f"Class: {pred_class}", (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(annotated_image, f"Score: {score:.2f}", (int(xmin), int(ymin)-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(nms_annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(nms_annotated_image, f"Class: {pred_class}", (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(nms_annotated_image, f"Score: {score:.2f}", (int(xmin), int(ymin)-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite("nmsed.jpg", nms_annotated_image)

    # Display the annotated image
    # cv2.namedWindow('Annotated Image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Annotated Image', cv2_image.shape[1], cv2_image.shape[0])
    # cv2.imshow("Annotated Image", annotated_image)
    cv2.imwrite("example.jpg", annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    open_image(r"C:\Users\User1\Desktop\alp\busxray-detector-main\example.jpg")
    print("done")

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
    from pathlib import Path

    # Get image
    img_path = Path(r"C:\Users\User1\Desktop\alp\test images\small-dogs-front-white-background-11290714.jpg")
    cv2_image = cv2.imread(str(img_path))
    predictions, nmsed_bbox = annotating(cv2_image)
    draw_predictions_vs_results(cv2_image, predictions, nmsed_bbox)

    # non_overlapping_boxes = annotating(cv2_image)
    test0 = {'bbox': [1292.3928709030151, 293.3261470794678, 610.1982421875, 245.49490356445312], 'score': 0.6807042956352234, 'pred_class': 62}
    test1 = {'bbox': [2571.0000972747803, 262.0, 640.0, 640.0], 'score': 0.7205166816711426, 'pred_class': 72}
    test2 = {'bbox': [2252.8322143554688, 637.2929344177246, 436.9841003417969, 328.2892150878906], 'score': 0.5942885875701904, 'pred_class': 9}
    test3 = {'bbox': [2482.857177734375, 620.1806755065918, 358.6601257324219, 330.1961364746094], 'score': 0.6686525344848633, 'pred_class': 9}
    test4 = {'bbox': [1592.4032287597656, 1705.140869140625, 328.7428283691406, 532.9769897460938], 'score': 0.9177506566047668, 'pred_class': 0}
    test5 = {'bbox': [1629.2117919921875, 1699.0096740722656, 365.1980895996094, 532.4293823242188], 'score': 0.5183762311935425, 'pred_class': 0}
    test6 = {'bbox': [1221.0665893554688, 1441.1936492919922, 598.6458129882812, 200.3468475341797], 'score': 0.5160679817199707, 'pred_class': 9}
    test7 = {'bbox': [1592.6559448242188, 1704.6822204589844, 328.85931396484375, 479.3410339355469], 'score': 0.8814602494239807, 'pred_class': 0}
    test8 = {'bbox': [1629.5948486328125, 1699.3317565917969, 364.8812255859375, 478.6779479980469], 'score': 0.7484795451164246, 'pred_class': 0}

    result0 = {'bbox': [1592.4032287597656, 1705.140869140625, 328.7428283691406, 532.9769897460938], 'score': 0.9177506566047668, 'pred_class': 0}
    result1 = {'bbox': [2571.0000972747803, 262.0, 640.0, 640.0], 'score': 0.7205166816711426, 'pred_class': 72}
    result2 = {'bbox': [1292.3928709030151, 293.3261470794678, 610.1982421875, 245.49490356445312], 'score': 0.6807042956352234, 'pred_class': 62}
    result3 = {'bbox': [2482.857177734375, 620.1806755065918, 358.6601257324219, 330.1961364746094], 'score': 0.6686525344848633, 'pred_class': 9}
    result4 = {'bbox': [2252.8322143554688, 637.2929344177246, 436.9841003417969, 328.2892150878906], 'score': 0.5942885875701904, 'pred_class': 9}
    result5 = {'bbox': [1221.0665893554688, 1441.1936492919922, 598.6458129882812, 200.3468475341797], 'score': 0.5160679817199707, 'pred_class': 9}
    
    # Compile the test dictionaries into a list
    test_list = [test0, test1, test2, test3, test4, test5, test6, test7, test8]
    # Compile the result dictionaries into a list
    result_list = [result0, result1, result2, result3, result4, result5]
    # draw_predictions_vs_results(cv2_image, test_list, result_list)


    
