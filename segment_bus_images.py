"""
This script provides two functions to segment an image into multiple smaller parts and adjust the annotation file 
accordingly for each segment.

The first function segment_image segments an input image into smaller parts of 640 by 640 pixels with a 
specified overlap percentage between adjacent segments. 
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

def segment_image(image_path, segment_size=640, overlap_percent=0.5):
    """
    Segments an image of any dimension into pieces of 640 by 640 with a specified overlap percentage.

    Args:
    image_path (str): The file path of the image to be segmented.
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


def adjust_annotations_for_segment(segment_path, original_annotation_path, output_annotation_path):
    """
    Adjusts the Pascal VOC annotation standard for an image segment.

    Args:
    segment_path (str): The file path of the image segment.
    annotation_path (str): The file path of the XML annotation file for the original image.

    Returns:
    None: The function saves the adjusted annotation to a new XML file for the segmented image.
    """

    # Load the segment image to get its dimensions
    segment_img = cv2.imread(segment_path)
    segment_height, segment_width, segment_depth = segment_img.shape

    # Parse the original annotation XML file
    tree = ET.parse(original_annotation_path)
    root = tree.getroot()

    # Create a new XML file for the segmented image
    _, filename = os.path.split(segment_path)
    output_annotation_path = os.path.join(output_annotation_path, gs.change_file_extension(filename, "") + '.xml')
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
    
    # Loop over the object annotations in the original annotation file
    for obj in root.findall('object'):
        # Get the bounding box coordinates for the current object
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
            
            create_new_object_annotation(xmin_adjusted, ymin_adjusted, xmax_adjusted, ymax_adjusted)    

    # Create an XML string with pretty formatting
    xml_string = minidom.parseString(ET.tostring(segmented_annotation)).toprettyxml(indent='    ')

    # Write the XML string to a file
    with open(output_annotation_path, 'w') as f:
        f.write(xml_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", help="folder containing the image and annotation files", default=r"D:\leann\busxray_woodlands\annotations_adjusted")
    parser.add_argument("--overlap-portion", help="fraction of each segment that should overlap adjacent segments. from 0 to 1", default=0.5)
    parser.add_argument("--segment-size", help="size of each segment", default=640)

    args = parser.parse_args()

    # Segment up the images
    os.chdir(args.root_dir)
    list_of_images = gs.load_images_from_folder(args.root_dir)
    
    print("Processing images.")
    for image in tqdm(list_of_images):
        segment_image(image_path=image,
                    segment_size=args.segment_size, 
                    overlap_percent=args.overlap_portion)

    # Segment up the annotation
    # SEGMENT_DIR = r"D:\leann\busxray_woodlands\annotations_adjusted\adjusted_1610_annotated_segmented"
    # ANNOTATION_PATH = r"D:\leann\busxray_woodlands\annotations_adjusted\adjusted_1610_annotated.xml"

    print("Processing XML files.")
    for root, dirs, _ in os.walk(args.root_dir):

        # Go through the list of subdirectories
        for subdir in dirs:
         
            # Go through each file in the list
            for file in os.listdir(os.path.join(root, subdir)):
                
                # Matches with the file name. ALERT HARD CODED NAME HERE!!!
                name_of_original_xml_file = subdir[0:-10]+".xml"
                # Only PNGs should be here
                adjust_annotations_for_segment(segment_path=os.path.join(root, subdir, file), 
                                               original_annotation_path=os.path.join(root, name_of_original_xml_file),
                                               output_annotation_path=os.path.join(root, subdir))

    # os.chdir(SEGMENT_DIR)
    # segment_list = gs.load_images_from_folder(SEGMENT_DIR)
    # for image in segment_list:
    #     adjust_annotations_for_segment(image, ANNOTATION_PATH)