"""
This script does 2 functions:
1. It cleans up the black boxes and white boundary in the bus image.
2. It re-adjusts the pascal VOC annotation according to how much the image was cropped.

Variables to change:
    -ROOT_DIR: Folder where images and XML annotations are stored. They need to be the same name except for the extension. Script will crash if either one of the file is not present.
    -TARGET_DIR: Folder to store the cropped images and adjusted XML files

@created: 4/3/2023
@author: Alp
"""

import cv2
import general_scripts as gs
import os, pathlib, argparse

from tqdm import tqdm

#ROOT_DIR = r"D:\leann\busxray_woodlands\annotations"
#TARGET_DIR = r"D:\leann\busxray_woodlands\annotations_adjusted"
#IMAGE_DIR = r"../busxray_woodlands sample/PA8506K Higer 49 seats-clean-1-1 Monochrome.tiff"

def find_black_to_white_transition(image_path):
    # Read image from file
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Constants
    BUFFER_SPACE_FROM_LEFT_BLACK_BOX = 10 # For top_to_bot and bot_to_top
    BUFFER_SPACE_TO_REFIND_SMALLEST_XY_VALUE = 30

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
    # Adjust the width and height of the image element
    size_elem.find('width').text = str(new_width)
    size_elem.find('height').text = str(new_height)

    # Calculate the offset for the new coordinates
    # new_coordinates[0] == x_start
    # new_coordinates[1] == x_end
    # new_coordinates[2] == y_start
    # new_coordinates[3] == y_end
    x_offset = new_coordinates[0]
    y_offset = new_coordinates[2]

    # Adjust the coordinates of each object in the XML file
    for obj_elem in root.findall('object'):
        bbox_elem = obj_elem.find('bndbox')

        # Get the original coordinates of the bounding box
        xmin = int(bbox_elem.find('xmin').text)
        ymin = int(bbox_elem.find('ymin').text)
        xmax = int(bbox_elem.find('xmax').text)
        ymax = int(bbox_elem.find('ymax').text)

        # Adjust the coordinates based on the new coordinates and offset values
        bbox_elem.find('xmin').text = str(int(xmin - x_offset))
        bbox_elem.find('ymin').text = str(int(ymin - y_offset))
        bbox_elem.find('xmax').text = str(int(xmax - x_offset))
        bbox_elem.find('ymax').text = str(int(ymax - y_offset))

    # Write the modified XML to the output file path
    tree.write(output_dir_path)

def resize_image_and_xml_annotation(input_file_name, output_dir_path):
    # Read image from file
    image = cv2.imread(input_file_name)

    # Find boundaries to crop
    x_start, x_end, y_start, y_end = find_black_to_white_transition(input_file_name)

    # Calculate new dimensions
    new_height = y_end - y_start
    new_width = x_end - x_start

    # Crop the image from the left-hand side
    cropped_image = image[y_start:y_end, x_start:x_end, ]

    # Resize image
    resized_image = cv2.resize(cropped_image, (new_width, new_height))

    # Write resized image to file
    image_file_path = os.path.join(output_dir_path, f"adjusted_{input_file_name}")
    cv2.imwrite(image_file_path, resized_image)

    # Get head and filename, because annotated XML file has same name as image
    _, filename = gs.path_leaf(input_file_name)
    # Change file extension to XML.
    xml_file_name = gs.change_file_extension(filename, ".xml")
    # Save XML file path
    xml_file_path = os.path.join(output_dir_path, f"adjusted_{xml_file_name}")
    # Run XML adjustment function
    adjust_xml_annotation(xml_file_path=xml_file_name, 
                          new_coordinates=(x_start, x_end, y_start, y_end), 
                          output_dir_path=xml_file_path)

    return resized_image

# To find out roughly what's the pixel intensity and at which X,Y coordinates.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def open_image(image_path):
    # Load image from file
    image = mpimg.imread(image_path)

    # Create Matplotlib figure and axes
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    # Define function to handle mouse motion
    def on_mouse_move(event):
        if event.xdata is not None and event.ydata is not None:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            intensity = image[y, x]
            ax.set_title(f"Intensity: {intensity}")

    # Connect mouse motion event to handler function
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    # Show Matplotlib window
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", help="folder containing the image and annotation files", default=r"D:\leann\busxray_woodlands\annotations")
    parser.add_argument("--target-dir", help="folder to place the cropped bus images", default=r"D:\leann\busxray_woodlands\annotations_adjusted")
    parser.add_argument("--display-only", help="don't crop images, just display an annotated image", action="store_true")
    parser.add_argument("--display-path", help="image file to display after adjustments", required=False)

    args = parser.parse_args()

    if not args.display_only:
        # Load images from folder
        cwd = os.chdir(args.root_dir)
        images = gs.load_images_from_folder(cwd)
        for image in tqdm(images):
            # function to return the file extension
            file_extension = pathlib.Path(image).suffix
            # Resize + adjust XML function and save it there
            resize_image_and_xml_annotation(image, args.target_dir)

    if args.display_path:
        display_path = args.display_path
    else:
        for file in os.listdir(args.target_dir):
            if os.path.splitext(file)[1] == ".jpg":
                display_path = os.path.join(args.target_dir, file)
                break

    # To open an image to check
    #open_image(r"D:\leann\busxray_woodlands\annotations_adjusted\adjusted_1610_annotated.jpg")
    print("Displaying image", display_path)
    open_image(display_path)