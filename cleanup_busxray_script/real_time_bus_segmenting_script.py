"""
NOT COMPLETE
This script receives a cv2-formatted image and performs the following:
    1) It segments up the image and stores them in a dict format in the format of:
        image_dict = {"0_0": 
                        {
                            "cv2_image": <cv2-format image>,
                            "json_file_info": <json-format image>},
                        {
                            "cv2_image": <cv2-format image>,
                            "json_file_info": <json-format image>},
                        }...
                    }

Some parts of the code contains this #canbeimproved , these are parts where we can consider changing the algorithm to improve the processing speed
"""

import os
import cv2
import numpy as np
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ImageProcessor:
    """
    Define ImageProcessor class with 
    """
    def __init__(self, input_cv2_image):
        """
        input_cv2_image: needs to be a cv2 image
        """
        # Check if input is a cv2 image
        if isinstance(self.input_cv2_image, np.ndarray):
            self.input_cv2_image = input_cv2_image
        else:
            raise TypeError("Image must be a numpy ndarray.")

        # Nothing is cropped at the start
        self.image_dict["cropped_coordinates"] = {"xmin": 0,
                                                  "xmax": 0,
                                                  "ymin": 0,
                                                  "ymax": 0}

    # Define function to store the cropped coordinates
    def crop_image(self):
        
        print("Cropping image...")
        x_start, x_end, y_start, y_end = find_black_to_white_transition(self.input_cv2_image)
        # Update cropped coordinates
        self.image_dict["cropped_coordinates"] = {"xmin": x_start,
                                                "xmax": x_end,
                                                "ymin": y_start,
                                                "ymax": y_end}

        # Algorithm to detect the edge of the bus in the image
        def find_black_to_white_transition(image):

            # Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Constants
            BUFFER_SPACE_FROM_LEFT_BLACK_BOX = 100 # For top_to_bot and bot_to_top. Ensures that residual black lines don't affect top and bot crop.
            BUFFER_SPACE_TO_REFIND_SMALLEST_XY_VALUE = 100 # This needs to be big for some files that thinks the edge is too much towards the center. But results in slower computation, #canbeimproved

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
            return  x_start, x_end, y_start, y_end

        
        


class ImageHandler(FileSystemEventHandler):
    # Define ImageHandler class to handle file system events
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    # Define function to handle file creation events
    def on_created(self, event):
        # Ignore events that aren't new files
        if event.is_directory:
            return
        # Process new image
        self.processor.process_image(os.path.basename(event.src_path))


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "input/"
    output_dir = "output/"
