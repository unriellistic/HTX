"""
Images with XML files must have the same name. Only difference being the file extension.
e.g.
    image file: PC1866G Scania KIB4X2 49 seats-threat-100.tiff
    annotation file: PC1866G Scania KIB4X2 49 seats-threat-100.xml

Image file can be any supported image format, while annotation file must be in .xml format. 

This script does 2 functions:
1. It cleans up the black boxes and white boundary in the bus image and relabels the file with adjusted_<filename>.jpg
    e.g. 355_annotated.jpg -> adjusted_355_annotated.jpg
2. It re-adjusts the pascal VOC annotation according to how much the image was cropped and relabels the file with adjusted_<filename>.xml
    e.g. 355_annotated.xml -> adjusted_355_annotated.xml
    Note: If image does not have it's corresponding xml file (clean images), script ignores it and does not adjust the xml file.

Input arguments:
--root-dir-images: specifies the folder containing the image files. The path specified must contain a folder of the image file with the same name as the annotated file in the annotation folder.
    e.g. 355_annotated.jpg and 355_annotated.xml
    Note: recursive function can be applied which search
--root-dir-annotations: specifies the folder containing the annotation files. The path specified must contain a folder of the XML file with the same name as the image file in the image folder.
    e.g. ..\images\355_annotated.jpg and ..\labels\355_annotated.xml
--target-dir: specifies the folder to place the cropped bus images.
--display: an optional argument that can be specified to just display the annotated image without running the cropping function.
--display-path: specifies the path to a singular image file to display.
--store: will cause the files generated to be stored at directory where image/annotation was found
--recursive-search: if true, will search both image and root dir recursively. Only works if both image and annotation file are in the same folder (can be subdirs, but same subdirs)

Full example:
To run the cropping function:
python crop_bus_images.py --root-dir-images "D:\leann\busxray_woodlands\annotations" --root-dir-annotations "D:\leann\busxray_woodlands\annotations" --target-dir "D:\leann\busxray_woodlands\annotations_adjusted"

To store files in the directory it was found:
python crop_bus_images.py --store --root-dir-images "D:\leann\busxray_woodlands\annotations" --root-dir-annotations "D:\leann\busxray_woodlands\annotations"

To display the all cropped images:
python crop_bus_images.py --target-dir "D:\leann\busxray_woodlands\annotations_adjusted" --display-only

To display one image:
python crop_bus_images.py --display --display-path "D:\leann\busxray_woodlands\annotations_adjusted\adjusted_355_annotated.jpg"


This will cause the function to look at root directory at <annotations> and saves the file at <annotations_adjusted>.

@current_author: Alp
@created at: 4/4/2023 10:23am
@last updated: 4/5/2023 4:48pm
    Patch notes:
        4/5/2023: 
            + Re-factored resize_image_and_xml_annotation, took out image cropping functions and created a new crop_image function.
            + Made adjustments to iteration values for find_black_to_white_transition function to improve efficiency by a factor of 10.

"""

import cv2
import general_scripts as gs
import os, pathlib, argparse
from tqdm import tqdm
import xml.dom.minidom as minidom # For pretty formatting
import numpy as np
import re # to extract unique number from filename

def find_black_to_white_transition(image):

    # Convert to grayscale if it's not already in grayscale
    if image.shape[2] != 1:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Constants
    BUFFER_SPACE_FROM_LEFT_BLACK_BOX = 200 # For top_to_bot and bot_to_top. Ensures that residual black lines don't affect top and bot crop.
    BUFFER_SPACE_TO_REFIND_SMALLEST_X_VALUE = 100 # Larger value to be more careful of the left black box
    BUFFER_SPACE_TO_REFIND_SMALLEST_Y_VALUE = 40 # Top down is usually more clear cut, can set to a lower value
    PIXEL_VALUE_TO_JUMP_FOR_X_VALUE = 20 # No need to iterate through every x-plane, one diagonal line in an image is ~30 pixels

    # Check if image is 16-bit or 8-bit and adjust pixel value accordingly
    if gray_image.dtype == np.uint8:
        PIXEL_INTENSITY_VALUE = 128
    else:
        PIXEL_INTENSITY_VALUE = 32768

    # Functions to find optimal xy co-ordinates to crop

    def left_to_right():
        # Iterate over pixels starting from left side of the image and moving towards the right
        # to find first black-to-white transition
        # Start at half of height to avoid white background (0-60) + light specks at 60~200
        most_left_x = image.shape[1]
        x_value_to_start_from = 50 # Sometimes there is white part at the start of the image, then it doesn't crop out the black portion.
        # Start from middle part of image, then iterate to the bottom
        for y in range(int(image.shape[0]/2), gray_image.shape[0]-1, 20):
            for x in range(x_value_to_start_from, gray_image.shape[1] - 1):
                if gray_image[y, x] < PIXEL_INTENSITY_VALUE and gray_image[y, x + 1] >= PIXEL_INTENSITY_VALUE:
                    # Found black-to-white transition
                    # Check if most_left_x has a x-value smaller than current x, if smaller it means it's positioned more left in the image.
                    # And since we don't want to cut off any image, we find the x that has the smallest value, which indicates that it's at the
                    # leftest-most part of the image
                    if most_left_x > x:
                        most_left_x = x
                        # Check if this will lead to out-of-bound index error
                        if most_left_x - BUFFER_SPACE_TO_REFIND_SMALLEST_X_VALUE < 0:
                            x_value_to_start_from = 0
                        else:
                            x_value_to_start_from = most_left_x - BUFFER_SPACE_TO_REFIND_SMALLEST_X_VALUE
                    # Found the transition, stop finding for this y-value
                    break
        return most_left_x

    def right_to_left():
        # Iterate over pixels starting from right side of the image and moving towards the left
        # to find first black-to-white transition
        # Start at half of height of image because that's the fattest part of the bus
        for y in range(int(image.shape[0]/2), gray_image.shape[0]-1, 20):
            for x in range(gray_image.shape[1]-1, 0, -1):
                if gray_image[y, x] >= PIXEL_INTENSITY_VALUE and gray_image[y, x - 1] < PIXEL_INTENSITY_VALUE:
                    # Found the y-coordinate in the center of the image's black-to-white transition
                    return x
        # If no transition detected, don't crop anything
        return gray_image.shape[1]

    def top_to_bot():
        # Iterate over pixels starting from top side of the image and moving towards the bottom
        # to find first black-to-white transition
        most_top_y = image.shape[0]
        y_value_to_start_from = 0
        # Start at halfway point because the highest point is always at the end of the bus
        # Jump by PIXEL_VALUE_TO_JUMP_FOR_X_VALUE for efficiency. Potential to improve algorithm here.
        for x in range(x_start + BUFFER_SPACE_FROM_LEFT_BLACK_BOX, x_end, PIXEL_VALUE_TO_JUMP_FOR_X_VALUE):
            for y in range(y_value_to_start_from, gray_image.shape[0]-1):
                if gray_image[y, x] >= PIXEL_INTENSITY_VALUE and gray_image[y+1, x] < PIXEL_INTENSITY_VALUE:
                    # Found black-to-white transition
                    # Check if most_top_y has a y-value larger than current y, if larger it means it's positioned lower in the image.
                    # And since we don't want to cut off any image, we find the y that has the smallest value, which indicates that it's at the
                    # top-most part of the image
                    if most_top_y > y:
                        most_top_y = y
                        # Check if this will lead to out-of-bound index error
                        if most_top_y - BUFFER_SPACE_TO_REFIND_SMALLEST_Y_VALUE < 0:
                            y_value_to_start_from = 0
                        else:
                            y_value_to_start_from = most_top_y - BUFFER_SPACE_TO_REFIND_SMALLEST_Y_VALUE
                    # Found the transition, stop finding for this x-value
                    break
        return most_top_y

    def bot_to_top():
        # Iterate over pixels starting from bottom side of the image and moving towards the top
        # to find first black-to-white transition
        most_bot_y = 0
        y_value_to_start_from = gray_image.shape[0] - 1
        # Start at halfway point because the highest point is always at the end of the bus
        # Jump by PIXEL_VALUE_TO_JUMP_FOR_X_VALUE for efficiency. Potential to improve algorithm here.
        for x in range(x_start + BUFFER_SPACE_FROM_LEFT_BLACK_BOX, x_end, PIXEL_VALUE_TO_JUMP_FOR_X_VALUE):
            for y in range(y_value_to_start_from, 0, -1):
                if gray_image[y, x] >= PIXEL_INTENSITY_VALUE and gray_image[y-1, x] < PIXEL_INTENSITY_VALUE:
                    # Found black-to-white transition
                    # Check if most_top_y has a y-value larger than current y, if larger it means it's positioned lower in the image.
                    # And since we don't want to cut off any image, we find the y that has the smallest value, which indicates that it's at the
                    # top part of the image
                    if most_bot_y < y:
                        most_bot_y = y
                        # Check if this will lead to out-of-bound index error
                        if most_bot_y + BUFFER_SPACE_TO_REFIND_SMALLEST_Y_VALUE > gray_image.shape[0] - 1:
                            y_value_to_start_from = gray_image.shape[0] - 1
                        else:
                            y_value_to_start_from = most_bot_y + BUFFER_SPACE_TO_REFIND_SMALLEST_Y_VALUE
                    # Found the transition, stop finding for this x-value
                    break
        return most_bot_y

    # Needs to run in this order as top_to_bot() utilises the x_start and x_end value.
    # Trim left black box
    x_start = left_to_right()
    # Trim right white empty space
    x_end = right_to_left()
    # Trim top white empty space
    y_start = top_to_bot()
    # Trim bot white empty space
    y_end = bot_to_top()

    return x_start, x_end, y_start, y_end

def adjust_xml_annotation(xml_file_path, new_coordinates, output_dir_path):
    import xml.etree.ElementTree as ET
    """
    Adjusts the coordinates in a Pascal VOC annotated XML file based on new coordinates provided and writes the modified XML
    to a new file.

    Args:
    xml_file_path (str): The file path of the Pascal VOC annotated XML file.
    new_coordinates (tuple): A tuple containing the new coordinates in the format (xmin, ymin, xmax, ymax).
    output_dir_path (str): The file path where the modified XML should be written to.
    
    Returns:
    None: The function writes the modified XML to the output file path.
    """

    # Parse the XML file and get the root element
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Get the original image size and adjust it
    size_elem = root.find('size')

    # Calculate the new width and height
    new_width = new_coordinates[1] - new_coordinates[0]
    new_height = new_coordinates[3] - new_coordinates[2]

    # Calculate the offset for the new coordinates
    x_offset = new_coordinates[0]
    y_offset = new_coordinates[2]

    # Create a new XML file for the segmented image
    adjusted_annotation = ET.Element('annotation')
    _, filename = os.path.split(xml_file_path)
    ET.SubElement(adjusted_annotation, 'folder').text = os.path.dirname(xml_file_path)
    ET.SubElement(adjusted_annotation, 'filename').text = filename
    ET.SubElement(adjusted_annotation, 'path').text = output_dir_path
    source = ET.SubElement(adjusted_annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(adjusted_annotation, 'size')
    ET.SubElement(size, 'width').text = str(new_width)
    ET.SubElement(size, 'height').text = str(new_height)
    ET.SubElement(size, 'depth').text = size_elem.find('depth')
    # Write the x and y offset into JSON file for future inference usage
    offset = ET.SubElement(adjusted_annotation, 'original_image_offset')
    ET.SubElement(offset, 'x_offset').text = str(x_offset)
    ET.SubElement(offset, 'y_offset').text = str(y_offset)

    def create_new_object_annotation(obj_elem, xmin, ymin, xmax, ymax):
        # Create a new object annotation with the adjusted bounding box coordinates
        segmented_obj = ET.SubElement(adjusted_annotation, 'object')
        ET.SubElement(segmented_obj, 'name').text = obj_elem.find('name').text
        ET.SubElement(segmented_obj, 'pose').text = obj_elem.find('pose').text
        ET.SubElement(segmented_obj, 'truncated').text = obj_elem.find('truncated').text
        ET.SubElement(segmented_obj, 'difficult').text = obj_elem.find('difficult').text
        segmented_bbox = ET.SubElement(segmented_obj, 'bndbox')
        ET.SubElement(segmented_bbox, 'xmin').text = xmin
        ET.SubElement(segmented_bbox, 'ymin').text = ymin
        ET.SubElement(segmented_bbox, 'xmax').text = xmax
        ET.SubElement(segmented_bbox, 'ymax').text = ymax

    # Adjust the coordinates of each object in the XML file
    for obj_elem in root.findall('object'):
        bbox_elem = obj_elem.find('bndbox')
        create_new_object_annotation(obj_elem,
                                     xmin=str(int(bbox_elem.find('xmin').text) - x_offset),
                                     xmax=str(int(bbox_elem.find('xmax').text) - x_offset),
                                     ymin=str(int(bbox_elem.find('ymin').text) - y_offset),
                                     ymax=str(int(bbox_elem.find('ymax').text) - y_offset))

    # Create an XML string with pretty formatting
    xml_string = minidom.parseString(ET.tostring(adjusted_annotation)).toprettyxml(indent='     ')

    # temporary solution for 16-bit depth images with different XML name
    # output_dir_path = output_dir_path[:-15] + "temp_image_low.xml"

    # Write the XML string to a file
    with open(output_dir_path, 'w') as f:
        f.write(xml_string)
    
    return

def crop_image(input_file_image, output_dir_path):
    """
    The crop_image function takes in an image, crops and savees it, and returns the cropped coordinates.
    
    Args:
    input_file_image (str): The path to the image file.
    """
    # Read image from file
    image = cv2.imread(input_file_image, cv2.IMREAD_UNCHANGED)

    # Check if image is a .tif format. If it is, it won't have shape[2] attribute to specify the channel size, give it one for easy future processing
    if image.dtype == np.uint16:
        image.shape = (image.shape[0], image.shape[1], 1)
    # Find boundaries to crop
    x_start, x_end, y_start, y_end = find_black_to_white_transition(image)

    # Calculate new dimensions
    new_height = y_end - y_start
    new_width = x_end - x_start

    # Crop the image from the left-hand side
    cropped_image = image[y_start:y_end, x_start:x_end, ]

    # Resize image
    resized_image = cv2.resize(cropped_image, (new_width, new_height))

    # Get image_head and image_filename
    image_head, image_filename = gs.path_leaf(input_file_image)

    # Check if user wants to store at current directory
    if output_dir_path == "store":
        # Write resized image to current directory
        image_file_path = os.path.join(image_head, f"adjusted_{image_filename}")
        # Save image
        cv2.imwrite(image_file_path, resized_image)
    # Else, store at target directory
    else:
        # Write resized image to output directory
        image_file_path = os.path.join(output_dir_path, f"adjusted_{image_filename}")
        # Save image
        cv2.imwrite(image_file_path, resized_image)

    return x_start, x_end, y_start, y_end


def resize_image_and_xml_annotation(input_file_image, input_file_label, output_dir_path):
    """
    The resize_image_and_xml_annotation function takes an input image file path and an output directory path as input parameters. 
    This function calls the crop_image function as well as the adjust_xml_annotation

    Args:
    input_file_image (str): The path to the image file.
    input_file_annotation (str): The path to the annotation file.
    output_dir_path (str): The file path where the modified XML should be written to.
    
    Returns:
    None: function writes the modified XML and resized image to the output file path.
    """
    # Crop image and get coordinates cropped
    x_start, x_end, y_start, y_end = crop_image(input_file_image, output_dir_path)

    # Get label_head and label_filename
    label_head, label_filename = gs.path_leaf(input_file_label)

    # Check if user wants to store at current directory
    if output_dir_path == "store":
        # Save XML file path
        adjusted_xml_file_path = os.path.join(label_head, f"adjusted_{label_filename}")
        
        # Check if XML file exists, if it does, perform adjustment, else ignore
        if os.path.exists(input_file_label):
            # Adjust annotation (aka labels)
            adjust_xml_annotation(  xml_file_path=input_file_label, 
                                    new_coordinates=(x_start, x_end, y_start, y_end), 
                                    output_dir_path=label_head)
    # Else, store at target directory
    else:
        # Save XML file path
        adjusted_xml_file_path = os.path.join(output_dir_path, f"adjusted_{label_filename}")

        # Check if XML file exists, if it does, perform adjustment, else ignore
        if os.path.exists(input_file_label):
            # Adjust annotation (aka labels)
            adjust_xml_annotation(  xml_file_path=input_file_label, 
                                    new_coordinates=(x_start, x_end, y_start, y_end), 
                                    output_dir_path=adjusted_xml_file_path)

    return None

# To find out roughly what's the pixel intensity and at which X,Y coordinates.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def open_image(image_path):
    """
    The open_image function takes:
    1) an image file path as input and loads the image using the Matplotlib library's mpimg.imread() function. 
    2) It then creates a new Matplotlib figure and axes, and displays the image using the ax.imshow() function with a gray color map.
    3a) The function also defines a new function called on_mouse_move() which handles mouse motion events on the image. 
    3b) The function determines the intensity of the pixel under the mouse cursor by rounding the x and y coordinates of the event and indexing into the image array. 
    3c) It then updates the title of the image axes with the current intensity value.
    3d) The open_image function connects the on_mouse_move() function to the mouse motion event using the fig.canvas.mpl_connect() method. 
    4) Finally, it displays the Matplotlib window with the image using the plt.show() function.
    3 doesn't really work I think, but doesn't matter.
"""
    # Load image from file
    image = mpimg.imread(image_path)

    # Create Matplotlib figure and axes
    _, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    # Show Matplotlib window
    plt.show()

# Function finds unique identifier in the form of -<number>-. If filename contains -<number>threat- or -threat-, it won't pick that up.
def extract_unique_number(filename):
    pattern = r'-(\d+)-'
    matches = re.findall(pattern, filename)
    if len(matches) > 1:
        print('Warning: multiple matches found for pattern, selecting the first one found')
    if matches:
        return matches[0]
    else:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--root-dir-images", help="folder containing the image files", default=r"annotations")
    # parser.add_argument("--root-dir-annotations", help="folder containing annotation files", default=r"annotations")
    # parser.add_argument("--recursive-search", help="if true, will search both image and root dir recursively", action="store_true", default=False)
    # parser.add_argument("--target-dir", help="folder to place the cropped bus images", default=r"annotations_adjusted")
    # parser.add_argument("--store", help="if true, will save both image and root dir in the directory found at", action="store_true", default=False)
    # parser.add_argument("--display", help="display the annotated images", action="store_true")
    # parser.add_argument("--display-path", help="path to display a single image file", required=False)

    # uncomment below if want to debug in IDE
    parser.add_argument("--root-dir-images", help="folder containing the image files", default=r"D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images_dualenergy")
    parser.add_argument("--root-dir-annotations", help="folder containing annotation files", default=r"D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images_dualenergy")
    parser.add_argument("--recursive-search", help="if true, will search both image and root dir recursively", action="store_true", default=False)
    parser.add_argument("--target-dir", help="folder to place the cropped bus images", default=r"D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_dualenergy")
    parser.add_argument("--store", help="if true, will save both image and root dir in the directory found at", action="store_true", default=False)
    parser.add_argument("--display", help="display the annotated images", action="store_true", default=True)
    parser.add_argument("--display-path", help="path to display a single image file", required=False, default=r"D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_monochrome\adjusted_PC1866G Scania KIB4X2 49 seats-Threat-139-temp_image_low_segmented\segment_640_1280.tif")



    args = parser.parse_args()
    """
    Just to clear some bug that occurs if root-dir default options settings is left as such
    """
    # Get path to root directory image
    # Check if default parameter is applied, if so get full path.
    if args.root_dir_images == "annotations":
        path_to_root_dir_images = os.path.join(os.getcwd(), args.root_dir_images)
    # Else, use path specified by user
    else:
        path_to_root_dir_images = args.root_dir_images

    # Get path to root directory annotation
    # Check if default parameter is applied, if so get full path.
    if args.root_dir_annotations == "annotations":
        path_to_root_dir_annotations = os.path.join(os.getcwd(), args.root_dir_annotations)
    # Else, use path specified by user
    else:
        path_to_root_dir_annotations = args.root_dir_annotations

    # Get path to target directory
    # Check if default parameter is applied, if so get full path.
    if args.target_dir == "annotations_adjusted":
        path_to_target_dir = os.path.join(os.getcwd(), args.target_dir)
    # Else, use path specified by user
    else:
        path_to_target_dir = args.target_dir

    # If user didn't specify display, just perform cropping without displaying
    if not args.display_path or not args.display:
        # Load images from folder
        images = gs.load_images(path_to_root_dir_images, recursive=args.recursive_search, file_type="all")
        # Create the output directory if it does not exist
        if args.store==False and not os.path.exists(path_to_target_dir):
            os.makedirs(path_to_target_dir)

        list_of_non_existent_labels = []
        for image in tqdm(images):
            # function to return the file extension
            file_extension = pathlib.Path(image).suffix

            # Get path to image and label
            _, image_name = gs.path_leaf(image)
            input_file_image = os.path.join(path_to_root_dir_images, image)
            input_file_label = os.path.join(path_to_root_dir_annotations, gs.change_file_extension(image_name, new_file_extension=".xml"))

            # Temporary solution for xml with different names
            # image_path, image_filename = gs.path_leaf(input_file_image)
            # label_filename = image_filename[:-18]+"final_color.xml"
            # input_file_label = os.path.join(path_to_root_dir_annotations, label_filename)

            # If label does not exist
            if os.path.exists(input_file_label) is False:
                _, temp_filename = gs.path_leaf(gs.change_file_extension(image, new_file_extension=".xml"))
                list_of_non_existent_labels.append(temp_filename)
                # Check if we want to store in current directory
                if args.store:
                    crop_image(input_file_image, "store")
                else:
                    crop_image(input_file_image, path_to_target_dir)
            
            # else, if it does exist
            else:
                # Resize + adjust XML function and save it there
                # Check if we want to store in current directory
                if args.store:
                    resize_image_and_xml_annotation(input_file_image=input_file_image,
                                                    input_file_label=input_file_label,
                                                    output_dir_path="store")
                else:
                    resize_image_and_xml_annotation(input_file_image=input_file_image,
                                                    input_file_label=input_file_label,
                                                    output_dir_path=path_to_target_dir)
        
    # If display option selected, then display the whole list in the --target-dir, or if a --display-path is specified, display just a singular image
    if args.display or args.display_path:
        if args.display_path:
            display_path = args.display_path
            print("Displaying image", display_path)
            open_image(display_path)
        else:
            for file in os.listdir(args.target_dir):
                if os.path.splitext(file)[1] == ".jpg":
                    display_path = os.path.join(args.target_dir, file)
                    print("Displaying image", display_path)
                    open_image(display_path)

    # print out non-existent labels
    print("List of non-existent labels:", list_of_non_existent_labels)
    # To open an image to check
    # open_image(r"D:\leann\busxray_woodlands\annotations_adjusted\adjusted_1610_annotated.jpg")
    # open_image(r"D:\leann\busxray_woodlands\annotations_adjusted\adjusted_1610_annotated_segmented\segment_156_397.png")
    