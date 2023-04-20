"""
This script does 2 functions:
1. It cleans up the black boxes and white boundary in the bus image and relabels the file with adjusted_<filename>.jpg
    e.g. 355_annotated.jpg -> adjusted_355_annotated.jpg
2. It re-adjusts the pascal VOC annotation according to how much the image was cropped and relabels the file with adjusted_<filename>.xml
    e.g. 355_annotated.xml -> adjusted_355_annotated.xml

Variables to change:
    -ROOT_DIR: Folder where images and XML annotations are stored. They need to be the same name except for the extension. Script will crash if either one of the file is not present.
    -TARGET_DIR: Folder to store the cropped images and adjusted XML files

Input arguments:
--root-dir: specifies the folder containing the image and annotation files. The path specified must contain a folder of both the XML and image file with the same name, only difference being the .jpg or .xml.
    e.g. 355_annotated.jpg and 355_annotated.xml
--target-dir: specifies the folder to place the cropped bus images.
--display: an optional argument that can be specified to just display the annotated image without running the cropping function.
--display-path: specifies the path to a singular image file to display.

Full example:
To run the cropping function:
python crop_bus_images.py --root-dir "D:\leann\busxray_woodlands\annotations" --target-dir "D:\leann\busxray_woodlands\annotations_adjusted"

To display the all cropped images:
python crop_bus_images.py --target-dir "D:\leann\busxray_woodlands\annotations_adjusted" --display-only

To display one image:
python crop_bus_images.py --display --display-path "D:\leann\busxray_woodlands\annotations_adjusted\adjusted_355_annotated.jpg"


This will cause the function to look at root directory at <annotations> and saves the file at <annotations_adjusted>.

@current_author: Alp
@last modified: 12/4/2023 2:48pm
"""

import cv2
import general_scripts as gs
import os, pathlib, argparse
from tqdm import tqdm
import xml.dom.minidom as minidom # For pretty formatting

def find_black_to_white_transition(image):

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Constants
    BUFFER_SPACE_FROM_LEFT_BLACK_BOX = 100 # For top_to_bot and bot_to_top. Ensures that residual black lines don't affect top and bot crop.
    BUFFER_SPACE_TO_REFIND_SMALLEST_XY_VALUE = 100

    # Functions to find optimal xy co-ordinates to crop

    def left_to_right():
        # Iterate over pixels starting from left side of the image and moving towards the right
        # to find first black-to-white transition
        # Start at half of height to avoid white background (0-60) + light specks at 60~200
        most_left_x = image.shape[1]
        x_value_to_start_from = 0
        # Start from middle part of image, then iterate to the bottom
        for y in range(int(image.shape[0]/2), gray_image.shape[0]-1, 20):
            for x in range(x_value_to_start_from, gray_image.shape[1] - 1):
                if gray_image[y, x] < 128 and gray_image[y, x + 1] >= 128:
                    # Found black-to-white transition
                    # Check if most_left_x has a x-value smaller than current x, if smaller it means it's positioned more left in the image.
                    # And since we don't want to cut off any image, we find the x that has the smallest value, which indicates that it's at the
                    # leftest-most part of the image
                    if most_left_x > x:
                        most_left_x = x
                        # Check if this will lead to out-of-bound index error
                        if most_left_x - BUFFER_SPACE_TO_REFIND_SMALLEST_XY_VALUE < 0:
                            x_value_to_start_from = 0
                        else:
                            x_value_to_start_from = most_left_x - BUFFER_SPACE_TO_REFIND_SMALLEST_XY_VALUE
                    # Found the transition, stop finding for this y-value
                    break
        return most_left_x

    def right_to_left():
        # Iterate over pixels starting from right side of the image and moving towards the left
        # to find first black-to-white transition
        # Start at half of height of image because that's the fattest part of the bus
        for y in range(int(image.shape[0]/2), gray_image.shape[0]-1, 20):
            for x in range(gray_image.shape[1]-1, 0, -1):
                if gray_image[y, x] >= 128 and gray_image[y, x - 1] < 128:
                    # Found the y-coordinate in the center of the image's black-to-white transition
                    return x
        # If no transition detected, don't crop anything
        return gray_image.shape[1]

    def top_to_bot():
        # Iterate over pixels starting from top side of the image and moving towards the bottom
        # to find first black-to-white transition
        most_top_y = image.shape[0]
        y_value_to_start_from = 0
        for x in range(x_start + BUFFER_SPACE_FROM_LEFT_BLACK_BOX, x_end):
            for y in range(y_value_to_start_from, gray_image.shape[0]-1):
                if gray_image[y, x] >= 128 and gray_image[y+1, x] < 128:
                    # Found black-to-white transition
                    # Check if most_top_y has a y-value larger than current y, if larger it means it's positioned lower in the image.
                    # And since we don't want to cut off any image, we find the y that has the smallest value, which indicates that it's at the
                    # top-most part of the image
                    if most_top_y > y:
                        most_top_y = y
                        # Check if this will lead to out-of-bound index error
                        if most_top_y - BUFFER_SPACE_TO_REFIND_SMALLEST_XY_VALUE < 0:
                            y_value_to_start_from = 0
                        else:
                            y_value_to_start_from = most_top_y - BUFFER_SPACE_TO_REFIND_SMALLEST_XY_VALUE
                    # Found the transition, stop finding for this x-value
                    break
        return most_top_y

    def bot_to_top():
        # Iterate over pixels starting from bottom side of the image and moving towards the top
        # to find first black-to-white transition
        most_bot_y = 0
        y_value_to_start_from = gray_image.shape[0] - 1
        for x in range(x_start + BUFFER_SPACE_FROM_LEFT_BLACK_BOX, x_end):
            for y in range(y_value_to_start_from, 0, -1):
                if gray_image[y, x] >= 128 and gray_image[y-1, x] < 128:
                    # Found black-to-white transition
                    # Check if most_top_y has a y-value larger than current y, if larger it means it's positioned lower in the image.
                    # And since we don't want to cut off any image, we find the y that has the smallest value, which indicates that it's at the
                    # top part of the image
                    if most_bot_y < y:
                        most_bot_y = y
                        # Check if this will lead to out-of-bound index error
                        if most_bot_y + BUFFER_SPACE_TO_REFIND_SMALLEST_XY_VALUE > gray_image.shape[0] - 1:
                            y_value_to_start_from = gray_image.shape[0] - 1
                        else:
                            y_value_to_start_from = most_bot_y + BUFFER_SPACE_TO_REFIND_SMALLEST_XY_VALUE
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

    # Write the XML string to a file
    with open(output_dir_path, 'w') as f:
        f.write(xml_string)
    
    return

def resize_image_and_xml_annotation(input_file_name, output_dir_path):
    """
    The resize_image_and_xml_annotation function takes an input image file path and an output directory path as input parameters. 
    The function first reads the image from the input file path using the OpenCV library's cv2.imread() function. 
    It then uses another function called find_black_to_white_transition() to identify the boundaries of the image that need to be cropped to remove any black borders or edges.

    After finding the boundaries, the function crops the image from the left-hand side and resizes it to a new height and width using the cv2.resize() function. 
    It then writes the resized image to a new file path in the output directory with the prefix "adjusted_" using the cv2.imwrite() function.

    The function then extracts the filename from the input file path and replaces the file extension with ".xml" to get the corresponding annotation file name. 
    It saves the new XML file path with the same "adjusted_" prefix as the image file. It then calls another function called adjust_xml_annotation() 
    which adjusts the coordinates of the annotations in the XML file to match the resized image's dimensions and new crop boundaries.

    Finally, the function returns the resized image.

    Args:
    input_file_name (str): The file path of the image file.
    output_dir_path (str): The file path where the modified XML should be written to.
    
    Returns:
    None: function writes the modified XML and resized image to the output file path.
    """
    # Read image from file
    image = cv2.imread(input_file_name)
    # Find boundaries to crop
    x_start, x_end, y_start, y_end = find_black_to_white_transition(image)

    # Calculate new dimensions
    new_height = y_end - y_start
    new_width = x_end - x_start

    # Crop the image from the left-hand side
    cropped_image = image[y_start:y_end, x_start:x_end, ]

    # Resize image
    resized_image = cv2.resize(cropped_image, (new_width, new_height))

    # Get head and filename, because annotated XML file has same name as image
    head, filename = gs.path_leaf(input_file_name)

    # Write resized image to file
    image_file_path = os.path.join(output_dir_path, f"adjusted_{filename}")
    cv2.imwrite(image_file_path, resized_image)
    
    # Change file extension to XML.
    xml_file_name = gs.change_file_extension(filename, ".xml")
    # Get XML file path
    xml_file_path = os.path.join(head, xml_file_name)
    # Save XML file path
    adjusted_xml_file_path = os.path.join(output_dir_path, f"adjusted_{xml_file_name}")
    # Run XML adjustment function
    adjust_xml_annotation(xml_file_path=xml_file_path, 
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", help="folder containing the image and annotation files", default=r"C:\Users\User1\Desktop\alp\cleanup_busxray_script\annotations")
    parser.add_argument("--target-dir", help="folder to place the cropped bus images", default=r"C:\Users\User1\Desktop\alp\cleanup_busxray_script\annotations_adjusted")
    parser.add_argument("--display", help="display the annotated images", action="store_true")
    parser.add_argument("--display-path", help="path to display a single image file", required=False)

    args = parser.parse_args()

    # uncomment below if want to debug in IDE
    # import sys
    # sys.argv = ['crop_bus_images.py', '--root-dir', r"C:\Users\User1\Desktop\alp\for soo kng\annotations"]

    # Get path to root directory
    # Check if default parameter is applied, if so get full path.
    if args.root_dir == "annotations":
        path_to_root_dir = os.path.join(os.getcwd(), args.root_dir)
    # Else, use path specified by user
    else:
        path_to_root_dir = args.root_dir

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
        images = gs.load_images(path_to_root_dir)

        # Create the output directory if it does not exist
        if not os.path.exists(path_to_target_dir):
            os.makedirs(path_to_target_dir)

        for image in tqdm(images):
            # function to return the file extension
            file_extension = pathlib.Path(image).suffix
            # Resize + adjust XML function and save it there
            input_file_name = os.path.join(path_to_root_dir, image)
            resize_image_and_xml_annotation(input_file_name=input_file_name,
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

    # To open an image to check
    # open_image(r"D:\leann\busxray_woodlands\annotations_adjusted\adjusted_1610_annotated.jpg")
    # open_image(r"D:\leann\busxray_woodlands\annotations_adjusted\adjusted_1610_annotated_segmented\segment_156_397.png")
    