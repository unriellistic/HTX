"""
This script provides two functions to segment an image into multiple smaller parts and adjust the annotation file 
accordingly for each segment. The output is saved in the same folder as specified in the --root-dir.

Update notes for V3: 
- Updated segmented files to include cutoff threshold and information loss
- Fixed all the bug, works as planned

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
@last modified: 18/4/2023 3:49pm

Things to work on:
- Think of how to "mask" the < 30% threshold portion of the annotation. 
    - Possible routes are using gaussian blur (might introduce artefacts which causes model degradation),
    - or simply snip away those parts of the image (have to run some sampling to see how much this method cuts away other portions of the image.)
        - function implemented, can consider decoupling it for user to turn on or off.
        - stopped at trying to figure out how to not double count for 2 masked object cancelling the same object area
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
    output_annotation_path = os.path.join(output_annotation_path, gs.change_file_extension(filename, "") + f'_{int(float(cutoff_threshold)*100)}_percent.xml')
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
        """
        a function that checks whether range1 and range2 overlap.
        """
        x1, x2 = range1.start, range1.stop
        y1, y2 = range2.start, range2.stop
        return x1 <= y2 and y1 <= x2
    
    def convert_coordinates_to_range(xmin, xmax, ymin, ymax):
        """
        a function to convert coordinate into a form usable by the range_overlap function
        """
        range_form_x_coordinates = range(xmin, xmax, 1)
        range_form_y_coordinates = range(ymin, ymax, 1)
        return range_form_x_coordinates, range_form_y_coordinates
    
    def calculate_size_of_area(xmin, xmax, ymin, ymax):
        return (xmax - xmin)*(ymax - ymin)
    
    def check_if_info_loss(info_loss):
        """
        Checks if there is any info loss, if there isn't returns False
        """
        if info_loss == 0.0:
            return False
        else:
            return True



    # Log file
    log_dict = {'num_of_reject': 0,
                'num_of_total': 0,}
    
    # Dictionary of values that we want to mask. Stores a list of x and y values that we will be cutting off
    mask_dict = {"plane_coordinate_to_mask": [],
                 "object_coordinates": {},
                 "mask_object_coordinates": {}}

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
                Store mask object's coordinate to see what's the best way to crop it out.
                Meaning, we will check if the x or y coordinate intersects with annotated boxes, if it doesn't we will cut via the plane
                that prioritises not cutting off any boxes first, and then if both plane will cut off boxes, we prioritise the one with the
                least amount of information loss
                """
                # Store mask object's coordinate
                mask_dict["mask_object_coordinates"][f"mask_obj_index_{index}"] = {
                                                                            "object_name": object_name,
                                                                            "xmin": xmin_adjusted,
                                                                            "ymin": ymin_adjusted,
                                                                            "xmax": xmax_adjusted,
                                                                            "ymax": ymax_adjusted
                                                                         }
    
    # At this point, the code finishes checking each segment's threshold limit and continues to check which is the best way to crop the image
    
    def find_best_plane_to_crop():
        """
        Find which plane cuts off the least amount of information and updates mask_dict["plane_coordinate_to_mask"].
        Additionally, it keeps track of the segment's info loss due to the cropping by the masked objects

        Function checks for different scenario's intersection with annotation:
            1) If both x and y plane has, we cut the one that will affect the least amount
            2) If only x-plane has, and y doesn't, no matter how big the value of y is, we choose to cut y since it doesn't result in any loss of annotation
            3) Likewise if only y-plane has, we do the same
            4) If both plane don't have information loss, we cut the one with the lesser value

        Returns:
        total_percentage_of_annotation_cut_for_segment: A float value that contains the information loss for this segment due to the cropping
        """
        def find_smallest_distance_to_border(left, right, top, bot):
            """
            Find out border direction to cut off from it's relative direction.
            left: compare with left border. argument value should be xmax
            right: compares with right border. argument value should be xmin
            top: compares with top border. argument value should be ymax
            bot: compares with bot border. argument value should be ymin

            Returns:
            plane_to_cut, value_to_mask
            plane_to_cut: string, indicates direction of cut ("top", "bot", "left", "right")
            value_to_mask: integer, indicates which pixel-plane to cut
            """
            x_plane_to_cut = "left"
            x_value_to_mask = 0
            y_plane_to_cut = "top"
            y_value_to_mask = 0

            # Calculate distance from border, from left-border to object vs right-border to object
            if left < segment_width - right:
                # means object is closer to the left border
                x_value_to_mask = left
                x_plane_to_cut = "left"
            else:
                # means object is closer to the right border
                x_value_to_mask = segment_width - right
                x_plane_to_cut = "right"

            # Calculate distance from border, from top-border to object vs bot-border to object
            if top < segment_height - bot:
                # means object is closer to the top border
                y_value_to_mask = top
                y_plane_to_cut = "top"
            else:
                # means object is closer to the bot border
                y_value_to_mask = segment_height - bot
                y_plane_to_cut = "bot"

            """
            Check which plane cuts away lesser parts of the image.
            """
            if x_value_to_mask < y_value_to_mask:
                # if the 'if statement' is true, it means distance from left or right to x_value_to_mask is smaller than distance from top or bot to y_value_to_mask
                return x_plane_to_cut, x_value_to_mask
            else:
                # else, y_value_to_mask is smaller, append this instead.
                return y_plane_to_cut, y_value_to_mask
        
        # Keep track of total info loss for the segment 
        total_percentage_of_annotation_cut_for_segment = 0
        
        # Iterate through each mask object
        for mask_object_coordinates in mask_dict["mask_object_coordinates"].values():
            
            """
            Generate 4 different mask planes to compare with object to see which results in least info loss
            plane 1: from left of image to mask_object's xmax
            plane 2: from right of image to mask_object's xmin
            plane 3: from top of image to mask_object's ymax
            plane 4: from bot of image to mask_object's ymin
            """
            # Get a range of mask's coordinates to check if it intersects with annotated objects
            dict_of_planes_coordinates = {  
                                            "plane 1": {
                                                    "xmin": 0,
                                                    "xmax": mask_object_coordinates["xmax"],
                                                    "ymin": 0,
                                                    "ymax": segment_height},
                                            "plane 2": {
                                                    "xmin": mask_object_coordinates["xmin"],
                                                    "xmax": segment_width,
                                                    "ymin": 0,
                                                    "ymax": segment_height},
                                            "plane 3": {
                                                    "xmin": 0,
                                                    "xmax": segment_width,
                                                    "ymin": 0,
                                                    "ymax": mask_object_coordinates["ymax"]},
                                            "plane 4": {
                                                    "xmin": 0,
                                                    "xmax": segment_width,
                                                    "ymin": mask_object_coordinates["ymin"],
                                                    "ymax": segment_height},
                                            }
            # To keep track of which plane has the lowest info loss, and it's direction
            # total_area_of_annotation_cut will have the value of 
            total_area_of_annotation_cut = 1000000000.0 # just an arbitary large value for algo to converge downwards
            total_object_annotation_area = 0.0
            plane_direction = "left"
            value_to_mask = 0
            # Boolean variables to see which plane has info loss, and compare those planes without info loss to find the plane to cut that results in
            # the least amount of image loss. Default is True (which means there is info loss)
            plane_has_info_loss_questionmark = {"plane 1": [True, 0], "plane 2": [True, 0], "plane 3": [True, 0], "plane 4": [True, 0]}

            # Check for 4 planes, which results in the least loss of info
            for plane, coordinates in dict_of_planes_coordinates.items():
                
                # Get a range of the mask object's coordinates to check if it intersects with objects
                mask_object_x_coordinates, mask_object_y_coordinates = convert_coordinates_to_range(xmin=coordinates["xmin"], xmax=coordinates["xmax"], ymin=coordinates["ymin"], ymax=coordinates["ymax"])
                
                # Keep track of total object area to eventually divide with cutoff_area_of_annotation,
                # temp_area_of_annotation_cut to keep track of the current plane's info loss
                total_object_annotation_area = 0.0
                temp_area_of_annotation_cut = 0.0
                

                # Iterate through each annotated object
                for object_coordinates in mask_dict["object_coordinates"].values():
                    
                    # Get a range of object's coordinates to check if it intersects with mask object
                    object_x_coordinates, object_y_coordinates = convert_coordinates_to_range(xmin=object_coordinates["xmin"], xmax=object_coordinates["xmax"], ymin=object_coordinates["ymin"], ymax=object_coordinates["ymax"])

                    # Calculate object's area of annotation
                    object_area_of_annotation = calculate_size_of_area(xmin=object_coordinates["xmin"], xmax=object_coordinates["xmax"], ymin=object_coordinates["ymin"], ymax=object_coordinates["ymax"])

                    # Sum the total so we can divide the cutoff_area_of_annotation/total_object_annotation_area to get the percentage of info loss
                    total_object_annotation_area += object_area_of_annotation

                    # Check if object overlaps with mask object
                    if range_overlap(mask_object_x_coordinates, object_x_coordinates) and range_overlap(mask_object_y_coordinates, object_y_coordinates):
                        
                        """
                            plane 1: from left of image to mask_object's xmax
                            plane 2: from right of image to mask_object's xmin
                            plane 3: from top of image to mask_object's ymax
                            plane 4: from bot of image to mask_object's ymin
                        """
                        # Find out each plane's cutoff area
                        if plane == "plane 1":
                            # Calculate object's that will be cut off by plane 1 (from left to mask object). Use object's y value, and the MIN of mask object's xmax value and object's xmax.
                            # Rationale is if the mask completely crops out the object, we simply use the object's max since the whole object will be cropped out
                            # else, if object is partially cropped out, we calculate area starting from the cropped out point, which is mask object's max.
                            cutoff_area_of_annotation = calculate_size_of_area( xmax=min(coordinates["xmax"], object_coordinates["xmax"]), 
                                                                                xmin=object_coordinates["xmin"],
                                                                                ymin=object_coordinates["ymin"], 
                                                                                ymax=object_coordinates["ymax"])
                        elif plane == "plane 2":
                            # Calculate object's that will be cut off by plane 2 (from right to mask object). Use object's y value, and the MAX of mask object's xmin value and object's xmin.
                            # Rationale is if the mask completely crops out the object, we simply use the object's min since the whole object will be cropped out
                            # else, if object is partially cropped out, we calculate area starting from the cropped out point, which is mask object's min.
                            cutoff_area_of_annotation = calculate_size_of_area( xmax=object_coordinates["xmax"], 
                                                                                xmin=max(coordinates["xmin"], object_coordinates["xmin"]), 
                                                                                ymin=object_coordinates["ymin"], 
                                                                                ymax=object_coordinates["ymax"])
                        elif plane == "plane 3":
                            # Calculate object's that will be cut off by plane 3 (from top to mask object). Use object's x value, and the MIN of mask object's ymax value and object's ymax.
                            # Rationale is if the mask completely crops out the object, we simply use the object's max since the whole object will be cropped out
                            # else, if object is partially cropped out, we calculate area starting from the cropped out point, which is mask object's max.
                            cutoff_area_of_annotation = calculate_size_of_area( xmax=object_coordinates["xmax"], 
                                                                                xmin=object_coordinates["xmin"], 
                                                                                ymin=object_coordinates["ymin"], 
                                                                                ymax=min(coordinates["ymax"], object_coordinates["ymax"]))
                        else:
                            # Calculate object's that will be cut off by plane 4 (from bot to mask object). Use object's x value, and the MAX of mask object's ymin value and object's ymin.
                            # Rationale is if the mask completely crops out the object, we simply use the object's min since the whole object will be cropped out
                            # else, if object is partially cropped out, we calculate area starting from the cropped out point, which is mask object's min.
                            cutoff_area_of_annotation = calculate_size_of_area( xmax=object_coordinates["xmax"], 
                                                                                xmin=object_coordinates["xmin"], 
                                                                                ymin=max(coordinates["ymin"], object_coordinates["ymin"]), 
                                                                                ymax=object_coordinates["ymax"])
                        
                        # Calculate percentage cut off relative to annotation's area size. (e.g. if 1/2 of a annotation is cut off, it'll be + 0.5)
                        # and add to total_area_of_annotation_cut for this plane
                        temp_area_of_annotation_cut += cutoff_area_of_annotation
                    
                """
                At this point, we have summed up the total info loss for one plane by iterating through every object. 
                """
                # Check if total info loss for this plane is less than previous plane. Have to put LTOE sign so that if multiple planes have 0.0, 
                # it's able to continue checking for the next condition.
                # Additionally, check if temp area cut must be less than total object area, if not less, means it cut away everything, and if so, don't include.
                if (temp_area_of_annotation_cut <= total_area_of_annotation_cut) and (temp_area_of_annotation_cut < total_object_annotation_area):

                    # Check if current value_to_mask is                    
                    total_area_of_annotation_cut = temp_area_of_annotation_cut
                    
                    # Find out which plane and update the direction and value
                    # The check_if_info_loss() function will return False if info_loss == 0.0,
                    # which means this plane doesn't have info loss and can be considered for choosing which plane to cut from
                    if plane == "plane 1":
                        plane_direction = "left"
                        value_to_mask = coordinates["xmax"]
                        plane_has_info_loss_questionmark["plane 1"] = [check_if_info_loss(temp_area_of_annotation_cut), value_to_mask]

                    elif plane == "plane 2":
                        plane_direction = "right"
                        value_to_mask = coordinates["xmin"]
                        plane_has_info_loss_questionmark["plane 2"] = [check_if_info_loss(temp_area_of_annotation_cut), value_to_mask]
                    elif plane == "plane 3":
                        plane_direction = "top"
                        value_to_mask = coordinates["ymax"]
                        plane_has_info_loss_questionmark["plane 3"] = [check_if_info_loss(temp_area_of_annotation_cut), value_to_mask]
                    else:
                        plane_direction = "bot"
                        value_to_mask = coordinates["ymin"]
                        plane_has_info_loss_questionmark["plane 4"] = [check_if_info_loss(temp_area_of_annotation_cut), value_to_mask]
                
                
                
            """
            At this point, we have found which plane has the lowest information loss after cutting an object for this particular mask object.
            """
            # Find which planes did not suffer from info loss
            list_of_planes_for_consideration = []
            for plane, boolean_values in plane_has_info_loss_questionmark.items():
                # Meaning, if not False, we consider it to find which plane has the least amount of image cut
                if not boolean_values[0]:
                    list_of_planes_for_consideration.append(plane)
            
            # If there are at least 2 planes that don't have info loss, we find which plane has the smallest distance cut
            if len(list_of_planes_for_consideration) > 1:
                
                # Returns the plane with the smallest distance cut and the value to cut
                plane_direction, value_to_mask = find_smallest_distance_to_border(left=(plane_has_info_loss_questionmark["plane 1"][1] if plane_has_info_loss_questionmark["plane 1"][0]==False else segment_width),
                                                                                  right=(plane_has_info_loss_questionmark["plane 2"][1] if plane_has_info_loss_questionmark["plane 2"][0]==False else 0),
                                                                                  top=(plane_has_info_loss_questionmark["plane 3"][1] if plane_has_info_loss_questionmark["plane 3"][0]==False else segment_height),
                                                                                  bot=(plane_has_info_loss_questionmark["plane 4"][1] if plane_has_info_loss_questionmark["plane 4"][0]==False else 0)
                                                                                  )
            
            # Save plane and value to cut. Save info loss as a percentage by dividing total area of annotation that's cut against total object area
            mask_dict['plane_coordinate_to_mask'].append((plane_direction, value_to_mask, 0.0 if total_object_annotation_area == 0.0 else total_area_of_annotation_cut/total_object_annotation_area))

        """
        At this point, we've finished calculating the best planes for each mask object. We now proceed to sum up the total area cut
        """
        # Divide total percentage gathered with number of annotations in image
        # A pre-emptive check incase mask_dict["plane_coordinate_to_mask"] doesn't have any values. 
        total_percentage_of_annotation_cut_for_segment = 0.0
        if len(mask_dict["plane_coordinate_to_mask"]):
            for mask_tuple in mask_dict["plane_coordinate_to_mask"]:
                """
                mask_tuple is a tuple that contains information in this format:
                mask_dict['plane_coordinate_to_mask'].append((plane_direction, 
                                                                value_to_mask, 
                                                                0.0 if total_area_of_annotation_cut == 0.0 else total_area_of_annotation_cut/total_object_annotation_area))
                """
                total_percentage_of_annotation_cut_for_segment += mask_tuple[2]
                # total_percentage_of_annotation_cut_for_segment/len(mask_dict["object_coordinates"].values())
        # End of function
        return total_percentage_of_annotation_cut_for_segment
       

    # Tabulate info loss via find_best_place_to_crop() and multiply it by 100 to transform 0.2 -> 20%
    segment_info_loss = round(100 * find_best_plane_to_crop(), 2)
    
    # Update XML file
    annotation_info_loss = cutoff_thres_info.find('annotation_info_loss')
    annotation_info_loss.text = str(segment_info_loss)
    # Update log_dict
    log_dict["segment_info_loss"] = segment_info_loss

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
    parser.add_argument("--root-dir", help="directory to the image and annotation files", default=r"C:\alp\HTX\cleanup_busxray_script\annotations_adjusted")
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
                    total_info_loss_for_one_image += stats['segment_info_loss']

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
            total_info_loss_for_all_images = total_info_loss_for_all_images/(len(image_stats_dict.keys()) if len(image_stats_dict.keys())!=0 else 1)

            # Tabulate total stats for one image in a subdir
            log_dict["Percentage threshold value set"] = args.cutoff_threshold
            log_dict["Overall total num of annotation"] = total_annotation_for_all_images
            log_dict["Overall total num of reject"] = total_rejects_for_all_images
            log_dict[r"Overall % of reject"] = str(round((total_rejects_for_all_images/(total_annotation_for_all_images if total_annotation_for_all_images!=0 else 1)) * 100, 2)) + r"%"
            log_dict["Overall total num of passed"] = total_annotation_for_all_images - total_rejects_for_all_images
            log_dict[r"Overall % of passed"] = str(round((total_annotation_for_all_images - total_rejects_for_all_images)/(total_annotation_for_all_images if total_annotation_for_all_images!=0 else 1) * 100, 2)) + r"%"
            log_dict[r"Overall % of info loss"] = str(round(total_info_loss_for_all_images, 2)) + r"%"
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