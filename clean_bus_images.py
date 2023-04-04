"""
This script cleans up the black boxes and white boundary in the bus image.
Variables to change:
    -ROOT_DIR: Folder where images are stored
    -TARGET_DIR: Folder to store the images


@created: 4/3/2023
"""

import cv2
import general_scripts as gs

ROOT_DIR = r"C:\alp\busxray_woodlands sample"
TARGET_DIR = r"C:\alp\busxray_woodlands sample"
IMAGE_DIR = r"../busxray_woodlands sample/PA8506K Higer 49 seats-clean-1-1 Monochrome.tiff"

def find_black_to_white_transition(image_path):
    # Read image from file
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    buffer_space_from_left_black_box = 10 # For top_to_bot and bot_to_top

    def left_to_right():
        # Iterate over pixels starting from left side of the image and moving towards the right
        # to find first black-to-white transition
        # Start at 200 to avoid white background + light specks at 60~200
        for y in range(200, gray_image.shape[0]):
            for x in range(gray_image.shape[1] - 1):
                if gray_image[y, x] < 128 and gray_image[y, x + 1] >= 128:
                    # Found black-to-white transition
                    return x

    def right_to_left():
        # Iterate over pixels starting from right side of the image and moving towards the left
        # to find first black-to-white transition
        # Start at half of height of image because that's the fattest part of the bus
        for y in range(int(image.shape[0]/2), gray_image.shape[0]):
            for x in range(gray_image.shape[1]-1, 0, -1):
                if gray_image[y, x] >= 128 and gray_image[y, x - 1] < 128:
                    # Found the y-coordinate in the center of the image's black-to-white transition
                    return x

    def top_to_bot():
        # Iterate over pixels starting from top side of the image and moving towards the bottom
        # to find first black-to-white transition
        most_top_y = 10000
        y_value_to_start_from = 0
        for x in range(x_start + buffer_space_from_left_black_box, x_end):
            for y in range(y_value_to_start_from, gray_image.shape[0]):
                if gray_image[y, x] >= 128 and gray_image[y+1, x] < 128:
                    # Found black-to-white transition
                    # Check if most_top_y has a y-value larger than current y, if larger it means it's positioned lower in the image.
                    # And since we don't want to cut off any image, we find the y that has the smallest value, which indicates that it's at the
                    # top part of the image
                    if most_top_y > y:
                        most_top_y = y
                        # Check if this will lead to out-of-bound index error
                        if most_top_y - 30 < 0:
                            y_value_to_start_from = 0
                        else:
                            y_value_to_start_from = most_top_y - 30
                    # Found the transition, stop finding for this x-value
                    break
        return most_top_y

    def bot_to_top():
        # Iterate over pixels starting from bottom side of the image and moving towards the top
        # to find first black-to-white transition
        most_bot_y = 0
        y_value_to_start_from = gray_image.shape[0] - 1
        for x in range(x_start + buffer_space_from_left_black_box, x_end):
            for y in range(y_value_to_start_from, 0, -1):
                if gray_image[y, x] >= 128 and gray_image[y-1, x] < 128:
                    # Found black-to-white transition
                    # Check if most_top_y has a y-value larger than current y, if larger it means it's positioned lower in the image.
                    # And since we don't want to cut off any image, we find the y that has the smallest value, which indicates that it's at the
                    # top part of the image
                    if most_bot_y < y:
                        most_bot_y = y
                        # Check if this will lead to out-of-bound index error
                        if most_bot_y + 30 > gray_image.shape[0] - 1:
                            y_value_to_start_from = gray_image.shape[0] - 1
                        else:
                            y_value_to_start_from = most_bot_y + 30
                    # Found the transition, stop finding for this x-value
                    break
        return most_bot_y

    x_start = left_to_right()
    x_end = right_to_left()
    y_start = top_to_bot()
    y_end = bot_to_top()

    return x_start, x_end, y_start, y_end

def resize_image(input_image_path, output_image_path):
    # Read image from file
    image = cv2.imread(input_image_path)

    # Find boundaries to crop
    x_start, x_end, y_start, y_end = find_black_to_white_transition(input_image_path)

    # Calculate new dimensions
    new_height = y_end
    new_width = x_end

    # Crop the image from the left-hand side
    cropped_image = image[y_start:y_end, x_start:x_end, ]

    # Resize image
    resized_image = cv2.resize(cropped_image, (new_width, new_height))

    # Write resized image to file
    cv2.imwrite(output_image_path, resized_image)

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
    import os
    import pathlib

    # Load images from folder
    cwd = os.chdir(ROOT_DIR)
    images = gs.load_images_from_folder(cwd)
    for index, image in enumerate(images):
        # function to return the file extension
        file_extension = pathlib.Path(image).suffix
        # Get new path for image
        new_image_location_and_name = os.path.join(TARGET_DIR, f"{index}{file_extension}")
        # Resizing function and save it there
        resize_image(image, new_image_location_and_name)

    # To open an image to check
    # open_image('../busxray_woodlands sample/test.jpg')
    # open_image('../busxray_woodlands sample/PA8506K Higer 49 seats-clean-1-1 DualEnergy.jpg')