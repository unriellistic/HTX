"""
This script provides two functions to segment an image into multiple smaller parts and adjust the annotation file 
accordingly for each segment. The output is saved in the same folder as specified in the --root-dir.

Update notes for V3: 
- Updated segmented files to include cutoff threshold and information loss
- 

Explanation:

The first function segment_image segments an input image into smaller parts of 640 by 640 pixels with a specified overlap percentage between adjacent segments. 
The function takes the path of the image to be segmented, overlap percentage, and segment size as input arguments. 
The function reads the input image using OpenCV, calculates the number of rows and columns required to segment the 
image based on its size and overlap percentage, creates an output directory to store the segmented images, 
segments the image into multiple parts using nested loops, and saves each segment as a PNG image file in the output directory.

The second function adjust_annotations_for_segment adjusts the annotation file for a segmented image. 
The function takes the paths of the segmented image and its original annotation file as input arguments. 
The function first loads the segmented image to get its dimensions and then parses the original 
annotation XML file using the ElementTree module. Next, the function creates a new XML file for the segmented image 
and adjusts the bounding box coordinates of each object annotation to match the coordinates of the corresponding object 
in the segmented image. Finally, the function saves the adjusted annotation to a new XML file for the segmented image.

Input arguments:
--root-dir: specifies the folder containing the adjusted image and annotation files. The path specified must contain a folder of both the XML and image file with the same name, only difference being the .jpg or .xml.
    e.g. adjusted_355_annotated.jpg and adjusted_355_annotated.xml
--overlap-portion: Float value, indicates fraction of each segment that should overlap adjacent segments. from 0 to 1. default=0.5 (50%)
--segment-size: Integer value, indicate size of each segment. default=640 (each image will be 640x640)
--cutoff-threshold: Float value, indicates threshold by which to not label an annotation that has been segmented
Full example:
To run the segmenting function:
python segment_bus_images.py --root-dir "D:\leann\busxray_woodlands\annotations_adjusted" --overlap-portion 640 --overlap-portion 0.5 --cutoff-threshold 0.3
-> This will cause the function to look at root directory at <annotations_adjusted>, splits the segment in 640x640 pieces. 
-> The overlap will be half of the image size, in this case half of 640 is 320. So the next segment after the first x_start = 0, x_end = 640, will be x_start = 320, x_end = 920.
-> Meaning the sliding window will be in increments of 320 pixels, in both width and height.

@author: Alp
@last modified: 13/4/2023 2:20pm

Things to work on:
- Think of how to "mask" the < 30% threshold portion of the annotation. 
    - Possible routes are using gaussian blur (might introduce artefacts which causes model degradation),
    - or simply snip away those parts of the image (have to run some sampling to see how much this method cuts away other portions of the image.)
        - function implemented, can consider decoupling it for user to turn on or off.
- Implement code to update for truncated images
"""
import cv2
import os
import general_scripts as gs
import numpy as np
# For adjusting XML segmentation
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom # For pretty formatting
import re # Find the dimension of the segmented image to find where the annotated boxes are
import argparse
from tqdm import tqdm
import json # For tracking of stats

def segment_image(image_path, segment_size, overlap_percent):
    """
    Segments an image of any dimension into pieces of specified by <segment_size>,
    with a specified overlap percentage specified by <overlap_percent>.

    Args:
    image_path (str): The file path of the image to be segmented.
    segment_size (int): Integer number for segment size.
    overlap_percent (float): The percentage of overlap between adjacent segments (0 to 1).

    Returns:
    None: The function saves the segmented images to the same directory as the original image.
    """

    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Get the height and width of the image
    height, width = img.shape[:2]

    # Calculate the number of rows and columns required to segment the image
    overlap_pixels = int(segment_size * overlap_percent)
    segment_stride = segment_size - overlap_pixels
    num_rows = int(np.ceil((height - segment_size) / segment_stride)) + 1
    num_cols = int(np.ceil((width - segment_size) / segment_stride)) + 1

    # Create the output directory if it does not exist
    _, filename = gs.path_leaf(image_path)
    output_dir = os.path.join(os.path.abspath(os.getcwd()), gs.change_file_extension(filename, "") + '_segmented')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Segment the image into pieces of 640 by 640 with the specified overlap percentage
    for row in range(num_rows):
        for col in range(num_cols):
            y_start = row * segment_stride
            y_end = y_start + segment_size
            x_start = col * segment_stride
            x_end = x_start + segment_size

            # Check if the remaining section of the image is less than 640 pixels
            if y_end > height:
                y_end = height
                y_start = height - segment_size
            if x_end > width:
                x_end = width
                x_start = width - segment_size

            segment = img[y_start:y_end, x_start:x_end]
            segment_path = output_dir + '\segment_{}_{}.png'.format(y_start, x_start)
            cv2.imwrite(segment_path, segment)


def adjust_annotations_for_segment(segment_path, original_annotation_path, output_annotation_path, cutoff_threshold, special_items):
    """
    Adjusts the Pascal VOC annotation standard for an image segment.

    Args:
    segment_path (str): Path of the image segment.
    original_annotation_path (str): Path of the original XML annotation file.
    output_annotation_path (str): Path to directory to store annotation.
    cutoff_threshold (float): Value to specific threshold at which to remove an object's annotation even though it's in the image
    special_items (list): A list of string items that contains the names of object to avoid threshlding (e.g. cig)

    Returns:
    log_dict: The function saves the adjusted annotation to a new XML file for the segmented image and outputs a log file to track statistics.
    """

    # Load the segment image to get its dimensions
    segment_img = cv2.imread(segment_path)
    segment_height, segment_width, segment_depth = segment_img.shape

    # Parse the original annotation XML file
    tree = ET.parse(original_annotation_path)
    root = tree.getroot()

    # Create a new XML file for the segmented image
    _, filename = os.path.split(segment_path)
    output_annotation_path = os.path.join(output_annotation_path, gs.change_file_extension(filename, "") + f'_{int(float(cutoff_threshold)*100)}percent.xml')
    segmented_annotation = ET.Element('annotation')
    ET.SubElement(segmented_annotation, 'folder').text = os.path.dirname(segment_path)
    ET.SubElement(segmented_annotation, 'filename').text = filename
    ET.SubElement(segmented_annotation, 'path').text = segment_path
    source = ET.SubElement(segmented_annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(segmented_annotation, 'size')
    ET.SubElement(size, 'width').text = str(segment_width)
    ET.SubElement(size, 'height').text = str(segment_height)
    ET.SubElement(size, 'depth').text = str(segment_depth)
    cutoff_thres_info = ET.SubElement(segmented_annotation, 'cutoff_threshold_info')
    ET.SubElement(cutoff_thres_info, 'cutoff_threshold').text = str(cutoff_threshold)
    ET.SubElement(cutoff_thres_info, 'annotation_info_loss').text = str(0)

    # Get coordinate values from segment image name
    numbers = re.findall(r'\d+', filename)
    xmin_segment = int(numbers[1])
    ymin_segment = int(numbers[0])
    xmax_segment = xmin_segment + segment_width
    ymax_segment = ymin_segment + segment_height
    segment_x_coordinates = range(xmin_segment, xmax_segment, 1)
    segment_y_coordinates = range(ymin_segment, ymax_segment, 1)

    def create_new_object_annotation(xmin, ymin, xmax, ymax):
        # Create a new object annotation with the adjusted bounding box coordinates
        segmented_obj = ET.SubElement(segmented_annotation, 'object')
        ET.SubElement(segmented_obj, 'name').text = obj.find('name').text
        ET.SubElement(segmented_obj, 'pose').text = obj.find('pose').text
        ET.SubElement(segmented_obj, 'truncated').text = obj.find('truncated').text
        ET.SubElement(segmented_obj, 'difficult').text = obj.find('difficult').text
        segmented_bbox = ET.SubElement(segmented_obj, 'bndbox')
        ET.SubElement(segmented_bbox, 'xmin').text = str(xmin)
        ET.SubElement(segmented_bbox, 'ymin').text = str(ymin)
        ET.SubElement(segmented_bbox, 'xmax').text = str(xmax)
        ET.SubElement(segmented_bbox, 'ymax').text = str(ymax)

    def range_overlap(range1, range2):
        """Whether range1 and range2 overlap."""
        x1, x2 = range1.start, range1.stop
        y1, y2 = range2.start, range2.stop
        return x1 <= y2 and y1 <= x2
    
    def calculate_size_of_area(xmin, xmax, ymin, ymax):
        return (xmax - xmin)*(ymax - ymin)

    # Log file
    log_dict = {'num_of_reject': 0,
                'num_of_total': 0,}
    
    # Dictionary of values that we want to mask. Stores a list of x and y values that we will be cutting off
    mask_dict = {"plane_coordinate_to_mask": [],
                 "object_coordinates": {}}

    # Loop over the object annotations in the original annotation file
    for index, obj in enumerate(root.findall('object')):
        # Get object type and bounding box coordinates for the current object
        object_name = obj.find('name')
        bbox = obj.find('bndbox')
        xmin_original = int(bbox.find('xmin').text)
        ymin_original = int(bbox.find('ymin').text)
        xmax_original = int(bbox.find('xmax').text)
        ymax_original = int(bbox.find('ymax').text)
        original_x_coordinates = range(xmin_original, xmax_original, 1)
        original_y_coordinates = range(ymin_original, ymax_original, 1)

        # Check whether any point of the annotation is in the current segment
        if range_overlap(original_x_coordinates, segment_x_coordinates) and range_overlap(original_y_coordinates, segment_y_coordinates):

            # Adjust the bounding box coordinates to be relative to the top left corner of the segment
            xmin_adjusted = max(0, xmin_original - xmin_segment)
            ymin_adjusted = max(0, ymin_original - ymin_segment)

            # If bounding box exists past segment xmax, label it at the segment's boundary
            if xmax_original > xmax_segment:
                xmax_adjusted = segment_width
            # Else re-adjust xmax by subtracting the location of the segment's x-min coordinate
            else:
                xmax_adjusted = xmax_original - xmin_segment
            
            # If bounding box exists past segment ymax, label it at the segment's boundary
            if ymax_original > ymax_segment:
                ymax_adjusted = segment_width
            # Else re-adjust xmax by subtracting the location of the segment's x-min coordinate
            else:
                ymax_adjusted = ymax_original - ymin_segment
            
            # Update object count
            log_dict['num_of_total'] += 1

            # Check if percentage overlap is greater than threshold. It finds the original area divided by adjusted area and checks if it's more than cutoff_threshold.
            adjusted_area = calculate_size_of_area(xmin=xmin_adjusted, xmax=xmax_adjusted, ymin=ymin_adjusted, ymax=ymax_adjusted)
            original_area = calculate_size_of_area(xmin=xmin_original, xmax=xmax_original, ymin=ymin_original, ymax=ymax_original)

            # Check if annotation is above threshold, or if annotation is in special_items. If either, create annotation for it.
            if ((adjusted_area/original_area) >= float(cutoff_threshold)) or any(object_name.text == item for item in special_items):
                # Store annotation in XML file
                create_new_object_annotation(xmin_adjusted, ymin_adjusted, xmax_adjusted, ymax_adjusted)

                # Store object's coordinate for statistic tracking
                mask_dict["object_coordinates"][f"obj_index_{index}"] = {
                                                                            "object_name": object_name,
                                                                            "xmin": xmin_adjusted,
                                                                            "ymin": ymin_adjusted,
                                                                            "xmax": xmax_adjusted,
                                                                            "ymax": ymax_adjusted
                                                                         }
            # Else, reject it and log the data for it.
            else:
                # Increment rejected box
                log_dict['num_of_reject'] += 1
                
                """
                Find out percentage of image it'll cut off if not included.

                Find which border direction to cut off from, from left, right, top, bot. 
                Afterwards, find the plane to cut off from
                """
                x_value_to_mask = 0
                y_value_to_mask = 0
                # plane_to_cut refers to indicating whether the cut is from the top, bot, left or right.
                x_plane_to_cut = "left"
                y_plane_to_cut = "top"

                # Checks distance from left of segment to object vs right of segment to object
                if 0 + xmax_adjusted < segment_width - xmin_adjusted:
                    # means object is nearer to the left side of image
                    x_value_to_mask = xmax_adjusted
                    x_plane_to_cut = "left"
                else:
                    # means object is nearer to the right side of image
                    x_value_to_mask = xmin_adjusted
                    x_plane_to_cut = "right"

                # Checks distance from top of segment to object vs bot of segment to object
                if 0 + ymax_adjusted < segment_height - ymin_adjusted:
                    # means object is nearer to the top side of image
                    y_value_to_mask = ymax_adjusted
                    y_plane_to_cut = "top"
                else:
                    # means object is nearer to the bot side of image
                    y_value_to_mask = ymin_adjusted
                    y_plane_to_cut = "bot"

                """
                Check which plane cuts away lesser parts of the image.
                At this point, we calculate distance from border, and use min as a logic to compare either left or right 'x' with either top or bot 'y'.
                afterwards, save the initial 'x' or 'y' mask value.
                """
                if min(segment_width - x_value_to_mask, 0 + x_value_to_mask) < min(segment_height - y_value_to_mask, 0 + y_value_to_mask):
                    # if the 'if statement' is true, it means distance from left or right to x_value_to_mask is smaller than distance from top or bot to y_value_to_mask
                    mask_dict['plane_coordinate_to_mask'].append((x_plane_to_cut, x_value_to_mask))
                else:
                    # else, y_value_to_mask is smaller, append this instead.
                    mask_dict['plane_coordinate_to_mask'].append((y_plane_to_cut, y_value_to_mask))
            
    """
    A function that calculates how much usable annotations will be cut off if we trim the photo
    """
    def search_info_loss():

        """
        A range converter function to convert coordinate into a form usable by the range_overlap function
        """
        def convert_coordinates_to_range(xmin, xmax, ymin, ymax):
            range_form_x_coordinates = range(xmin, xmax, 1)
            range_form_y_coordinates = range(ymin, ymax, 1)
            return range_form_x_coordinates, range_form_y_coordinates
        
        """
        mask_dict is a dictionary in this format:
        {
            "plane_coordinate_to_mask": [(x_left_or_right, x_value_to_mask), ..., (x_left_or_right, x_value_to_mask)],
            "object_coordinates": {
                                    "obj_index_1": {
                                        "object_name": object_name,
                                        "xmin": xmin_adjusted,
                                        "ymin": ymin_adjusted,
                                        "xmax": xmax_adjusted,
                                        "ymax": ymax_adjusted
                                    }, 
                                    ...,
                                    "obj_index_6": {
                                        "object_name": object_name,
                                        "xmin": xmin_adjusted,
                                        "ymin": ymin_adjusted,
                                        "xmax": xmax_adjusted,
                                        "ymax": ymax_adjusted
                                    },
                                }
        }
        """
        
        # Find how much percentage is masked off
        total_percentage_of_usable_annotation_cut = 0.0

        # Find percentage of annotation lost if plane gets masked off
        # masking_coordinates is a tuple with items, (x_left_or_right, x_value_to_mask) or (y_top_or_bot, y_value_to_mask).
        for masking_coordinates in mask_dict["plane_coordinate_to_mask"]:
            
            """
            Save coordinate in this format to run the range_overlap function
            """
            if masking_coordinates[0] == "left":
                # coordinate to mask is from left to x-coordinate
                masking_x_coordinates, masking_y_coordinates = convert_coordinates_to_range(xmin=0, xmax=masking_coordinates[1], ymin=0, ymax=segment_height)
            elif masking_coordinates[0] == "right":
                # coordinate to mask is from right to x-coordinate
                masking_x_coordinates, masking_y_coordinates = convert_coordinates_to_range(xmin=masking_coordinates[1], xmax=segment_width, ymin=0, ymax=segment_height)
            elif masking_coordinates[0] == "top":
                # coordinate to mask is from top to y-coordinate
                masking_x_coordinates, masking_y_coordinates = convert_coordinates_to_range(xmin=0, xmax=segment_width, ymin=0, ymax=masking_coordinates[1])
            # else statement basically means this: elif masking_coordinates[0] == "bot":
            else:
                # coordinate to mask is from bot to y-coordinate
                masking_x_coordinates, masking_y_coordinates = convert_coordinates_to_range(xmin=0, xmax=segment_width, ymin=masking_coordinates[1], ymax=segment_height)

            # Get each object coordinates
            # object_coordinates is a dictionary with "obj_index_1", ..., "obj_index_6" items
            for object_coordinates in mask_dict["object_coordinates"].values():
                object_x_coordinates, object_y_coordinates = convert_coordinates_to_range(xmin=object_coordinates["xmin"], 
                                                                                         xmax=object_coordinates["xmax"],
                                                                                         ymin=object_coordinates["ymin"],
                                                                                         ymax=object_coordinates["ymax"])

                # Check if annotations are intersecting with masking coordinates
                if range_overlap(masking_x_coordinates, object_x_coordinates) and range_overlap(masking_y_coordinates, object_y_coordinates):

                    # Calculate original area of annotation
                    adjusted_area_of_annotation = calculate_size_of_area(xmin=object_coordinates["xmin"], xmax=object_coordinates["xmax"], ymin=object_coordinates["ymin"], ymax=object_coordinates["ymax"])
                    
                    # Check which direction to cut from
                    if masking_coordinates[0] == "left":
                        # xmax will be the masking_coordinate since if we start cutting from the left, the plane at which we cut will be xmax
                        cutoff_area_of_annotation = calculate_size_of_area(xmax=masking_coordinates[1],
                                                                           xmin=object_coordinates["xmin"], ymin=object_coordinates["ymin"], ymax=object_coordinates["ymax"])
                    elif masking_coordinates[0] == "right":
                        # xmin will be the masking_coordinate since if we start cutting from the right, the plane at which we cut will be xmin
                        cutoff_area_of_annotation = calculate_size_of_area(xmin=masking_coordinates[1],
                                                                           xmax=object_coordinates["xmax"], ymin=object_coordinates["ymin"], ymax=object_coordinates["ymax"])
                    elif masking_coordinates[0] == "top":
                        # ymax will be the masking_coordinate since if we start cutting from the top, the plane at which we cut will be ymax
                        cutoff_area_of_annotation = calculate_size_of_area(ymax=masking_coordinates[1],
                                                                           xmax=object_coordinates["xmax"], ymin=object_coordinates["ymin"], xmin=object_coordinates["xmin"])

                    # else statement basically means this: elif masking_coordinates[0] == "bot":
                    else:
                        # ymin will be the masking_coordinate since if we start cutting from the bot, the plane at which we cut will be ymin
                        cutoff_area_of_annotation = calculate_size_of_area(ymin=masking_coordinates[1],
                                                                           xmax=object_coordinates["xmax"], ymax=object_coordinates["ymax"], xmin=object_coordinates["xmin"])
                    
                    # Update total percentage cut off relative to annotation's area size. So if 1/2 of a annotation is cut off, it'll be + 0.5.
                    total_percentage_of_usable_annotation_cut += (cutoff_area_of_annotation/adjusted_area_of_annotation)

        # Divide total percentage gathered with number of annotations in image
        # A pre-emptive check incase somehow object_coordinates doesn't have any values. 
        # Note: mask_dict["object_coordinates"] contains a dictionary of annotated objects in the segment image.
        if len(mask_dict["object_coordinates"].values()):
            total_percentage_of_usable_annotation_cut = total_percentage_of_usable_annotation_cut/len(mask_dict["object_coordinates"].values())
        # End of function
        return total_percentage_of_usable_annotation_cut
                    
    # Tabulate statistics
    # search_info_loss() returns "total_percentage_of_usable_annotation_cut"
    info_loss = round(100 * search_info_loss(), 2)
    
    # Update XML file
    annotation_info_loss = cutoff_thres_info.find('annotation_info_loss')
    annotation_info_loss.text = str(info_loss)
    # Update log_dict
    log_dict["info_loss"] = info_loss

    # Create an XML string with pretty formatting
    xml_string = minidom.parseString(ET.tostring(segmented_annotation)).toprettyxml(indent='    ')

    # Write the XML string to a file
    with open(output_annotation_path, 'w') as f:
        f.write(xml_string)
    
    return log_dict

def mask_out_object_features_below_threshold(image_path):

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", help="directory to the image and annotation files", default=r"annotations_adjusted")
    parser.add_argument("--overlap-portion", help="fraction of each segment that should overlap adjacent segments. from 0 to 1", default=0.5)
    parser.add_argument("--segment-size", help="size of each segment", default=640)
    parser.add_argument("--cutoff-threshold", help="cutoff threshold to determine whether to exclude annotation from the new segment", default=0.3)
    parser.add_argument("--special-items", help="a list of string items to supercede the threshold set", default=['cig'])

    # uncomment below if want to debug in IDE
    # import sys
    # Manually set the command-line arguments for debugging
    # sys.argv = ['segment_bus_images_v3.py', '--root-dir', r"D:\leann\busxray_woodlands\annotations_adjusted", '--overlap-portion', "0.5", '--cutoff-threshold', "0.3"]

    args = parser.parse_args()

    # Get path to directory
    # Check if default parameter is applied, if so get full path.
    if args.root_dir == "annotations_adjusted":
        path_to_dir = os.path.join(os.getcwd(), args.root_dir)
    # Else, use path specified by user
    else:
        path_to_dir = args.root_dir

    # Load the images
    list_of_images = gs.load_images(path_to_dir)
    
    # Segment up the images
    # print("Processing images...")
    # os.chdir(path_to_dir)
    # for image in tqdm(list_of_images):
    #     segment_image(image_path=image,
    #                 segment_size=int(args.segment_size), 
    #                 overlap_percent=float(args.overlap_portion))

    # Segment up the annotation
    print("Processing XML files...")

    """
    Log file in the form of:
    {
      ["Overall total num of annotation"] = total_annotation_for_all_images
      ["Overall total num of reject"] = total_rejects_for_all_images
      ["image info"] = image_stats_dict
    }
    """
    log_dict = {}
    lousy_root_checker = False
    for root, dirs, _ in os.walk(path_to_dir):
        
        # If dirs is empty, means no subdir found. This step is mainly for tqdm to work, because if don't check for empty dirs, after it successfully iterates
        # through the subdirectories, i think it somehow checks it again, but it knows that it has already parsed it, which results in tqdm printing additional lines
        # lousy_root_checker is a short-term solution for tdqm to work so that it doesn't skip pass the first direction
        if not dirs and lousy_root_checker:
            continue
        else:
            # Turn on lousy_root_check
            lousy_root_checker = True # a short-term solution for tdqm to work
            
            """
            image log file in the form of:
            {
              "image's total annotation": total_annotation_for_one_image,
              "image's total reject": total_annotation_for_one_image,
              "image's total info loss": total_info_loss,
              "image's segment info": segment_stats_dict,
            }
            """
            image_stats_dict = {}
            # Go through the list of subdirectories
            for subdir in tqdm(dirs, position=0, leave=True):
                
                """
                segment log file in the form of:
                {
                'num_of_reject': 0,
                'num_of_total': 0,
                'info_loss': 0.0
                }
                """
                segment_stats_dict = {}

                # Go through each file in the list
                for file in os.listdir(os.path.join(root, subdir)):
                    
                    # In case the script is re-run after it had ran before, check that we only call the
                    # adjust_annotations_for_segment function on png images.
                    if file.endswith(".png"):
                        # Matches with the file name. ALERT HARD CODED NAME HERE!!!
                        name_of_original_xml_file = subdir[0:-10]+".xml"

                        # XML file created within function. Return function returns statistics. New key generated for each segment, values will be the stats for each segment.
                        # Only PNGs should be here
                        segment_stats_dict[f"{file}"] = adjust_annotations_for_segment(segment_path=os.path.join(root, subdir, file), 
                                                    original_annotation_path=os.path.join(root, name_of_original_xml_file),
                                                    output_annotation_path=os.path.join(root, subdir),
                                                    cutoff_threshold=args.cutoff_threshold,
                                                    special_items=args.special_items)
                        
                # Tabulate total statistics for single image
                total_rejects_for_one_image = 0
                total_annotation_for_one_image = 0
                total_info_loss_for_one_image = 0.0
                # Go through each segment and sum up the stats
                for stats in segment_stats_dict.values():
                    total_rejects_for_one_image += stats["num_of_reject"]
                    total_annotation_for_one_image += stats['num_of_total']
                    total_info_loss_for_one_image += stats['info_loss']

                # Re-adjusts info loss based on the number of segments (this part may need re-adjusting)
                total_info_loss_for_one_image = total_info_loss_for_one_image/len(segment_stats_dict.keys())

                # Tabulate total stats for one image in a subdir
                image_stats_dict[f"{subdir}"] = {
                                                "image's total annotation": total_annotation_for_one_image,
                                                "image's total reject": total_rejects_for_one_image,
                                                "image's total info loss": total_info_loss_for_one_image,
                                                "image's segment info": segment_stats_dict,
                                                }
            
            # Tabulate total statistics across all images
            total_rejects_for_all_images = 0
            total_annotation_for_all_images = 0
            total_info_loss_for_all_images = 0
            # Go through each segment and sum up the stats
            for stats in image_stats_dict.values():
                total_rejects_for_all_images += stats["image's total reject"]
                total_annotation_for_all_images += stats["image's total annotation"]
                total_info_loss_for_all_images += stats["image's total info loss"]
            # Re-adjusts info loss based on the number of images
            total_info_loss_for_all_images = total_info_loss_for_all_images/len(image_stats_dict.keys())

            # Tabulate total stats for one image in a subdir
            log_dict["Percentage threshold value set"] = args.cutoff_threshold
            log_dict["Overall total num of annotation"] = total_annotation_for_all_images
            log_dict["Overall total num of reject"] = total_rejects_for_all_images
            log_dict[r"Overall % of reject"] = round(total_rejects_for_all_images/total_annotation_for_all_images * 100, 2)
            log_dict["Overall total num of passed"] = total_annotation_for_all_images - total_rejects_for_all_images
            log_dict[r"Overall % of passed"] = round((total_annotation_for_all_images - total_rejects_for_all_images)/total_annotation_for_all_images * 100, 2)
            log_dict[r"Overall % of info loss"] = round(total_info_loss_for_all_images, 2)
            log_dict["image info"] = image_stats_dict

            head, _ = gs.path_leaf(args.root_dir)
            with open(os.path.join(head, f"busxray_stats_with_{args.cutoff_threshold}.json"), 'w') as outfile:
                json.dump(log_dict, outfile, indent=4)



    # For individual folder testing, uncomment if applicable.
    """
    SEGMENT_DIR = r"D:\leann\busxray_woodlands\annotations_adjusted\adjusted_1610_annotated_segmented"
    ANNOTATION_PATH = r"D:\leann\busxray_woodlands\annotations_adjusted\adjusted_1610_annotated.xml"
    os.chdir(SEGMENT_DIR)
    segment_list = gs.load_images(SEGMENT_DIR)
    for image in segment_list:
        adjust_annotations_for_segment(image, ANNOTATION_PATH)
    """