"""
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
"""

import cv2, numpy as np
import time

def time_func(func):
    """
    A function to keep track and print out time taken for each function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"function {func.__name__} took {elapsed_time:.3f} seconds to run.")
        return result
    return wrapper


class ImageProcessor:
    """
    Define ImageProcessor class with functions that crops white and black borders, and segments itself
    """
    def __init__(self, input_cv2_image):
        """
        input_cv2_image: needs to be a cv2 image
        """
        # Check if input is a cv2 image
        if isinstance(input_cv2_image, np.ndarray):
            self.original_image = input_cv2_image
        else:
            raise TypeError("Image must be a numpy ndarray.")

        # Nothing is cropped at the start
        self.image_dict = {"cropped_coordinates": {"xmin": 0,
                                                  "xmax": 0,
                                                  "ymin": 0,
                                                  "ymax": 0}}
        self.segment_image_info = {}

    # Define function to store the cropped coordinates
    @time_func
    def crop_image(self):
        
        # Algorithm to detect the edge of the bus in the image
        def find_black_to_white_transition(image):

            # Convert to grayscale if it's not already in grayscale
            if image.shape[2] != 1:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Constants
            BUFFER_SPACE_FROM_LEFT_BLACK_BOX = 200 # For top_to_bot and bot_to_top. Ensures that residual black lines don't affect top and bot crop.
            BUFFER_SPACE_FROM_BOT_OF_IMAGE = 100 # For images with random white boxes at the bottom
            BUFFER_SPACE_TO_REFIND_SMALLEST_X_VALUE = 100 # Larger value to be more careful of the left black box
            BUFFER_SPACE_TO_REFIND_SMALLEST_Y_VALUE = 40 # Top down is usually more clear cut, can set to a lower value
            PIXEL_VALUE_TO_JUMP_FOR_X_VALUE = 20 # No need to iterate through every x-plane, one diagonal line in an image is ~30 pixels
            PIXEL_VALUE_TO_JUMP_FOR_Y_VALUE = 10 # No need to iterate through every y-plane, each feature in an image is ~20 pixels
            # 20 was chosen due to domain exploration, artefacts are at most ~10 pixels long
            NUM_OF_PIXEL_TO_AVERAGE = 20 # To take the average of this amount of pixels in a line. (Horizontal for x, vertical for y)

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

                # Start from middle part of image, then iterate to the bottom - some buffer space
                for y in range(int(image.shape[0]/2), gray_image.shape[0]-BUFFER_SPACE_FROM_BOT_OF_IMAGE, PIXEL_VALUE_TO_JUMP_FOR_Y_VALUE):
                    for x in range(x_value_to_start_from, gray_image.shape[1] - 1, int(NUM_OF_PIXEL_TO_AVERAGE/4)):
                        # Check for line brightness
                        line_pixels = gray_image[y, x : x + NUM_OF_PIXEL_TO_AVERAGE]
                        avg_pixel_value = line_pixels.mean()
                        # This is to check for completely white pixels as well
                        # Because sometimes the black-border does not fully cover from top to bot, then algo would mistakenly stop at an all white plane
                        if avg_pixel_value > PIXEL_INTENSITY_VALUE:
                            
                            # Found black-to-white transition
                            # Check if most_left_x has a x-value smaller than current x, if smaller it means it's positioned more left in the image.
                            # And since we don't want to cut off any image, we find the x that has the smallest value, which indicates that it's at the
                            # leftest-most part of the image
                            if most_left_x > x:
                                most_left_x = x
                            # Found the transition, stop finding for this y-value
                            break
                return most_left_x

            def right_to_left():
                # Iterate over pixels starting from right side of the image and moving towards the left
                # to find first black-to-white transition
                # Start at half of height of image because that's the fattest part of the bus
                for y in range(int(image.shape[0]/2), gray_image.shape[0]-1, PIXEL_VALUE_TO_JUMP_FOR_Y_VALUE):
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
                    for y in range(y_value_to_start_from, gray_image.shape[0]):
                        
                        line_pixels = gray_image[y : y + NUM_OF_PIXEL_TO_AVERAGE, x]
                        avg_pixel_value = line_pixels.mean()
                        # Find a shift in pixel intensity
                        if avg_pixel_value < int(PIXEL_INTENSITY_VALUE * 1.9):
                            for index, pixel in enumerate(line_pixels):
                                if pixel < int(PIXEL_INTENSITY_VALUE * 1.9):
                                    # Found black-to-white transition
                                    # Check if most_top_y has a y-value larger than current y, if larger it means it's positioned lower in the image.
                                    # And since we don't want to cut off any image, we find the y that has the smallest value, which indicates that it's at the
                                    # top-most part of the image
                                    if most_top_y > y + index:
                                        most_top_y = y + index
                                        # Check if this will lead to out-of-bound index error
                                        if most_top_y - BUFFER_SPACE_TO_REFIND_SMALLEST_Y_VALUE < 0:
                                            y_value_to_start_from = 0
                                        else:
                                            y_value_to_start_from = most_top_y - BUFFER_SPACE_TO_REFIND_SMALLEST_Y_VALUE
                                        # Found the transition, stop finding for this x-value
                                        break
                            # Found the transition, stop finding for this x-value
                            break
                return most_top_y

            def bot_to_top():
                # Iterate over pixels starting from bottom side of the image and moving towards the top
                # to find first black-to-white transition
                most_bot_y = 0
                y_value_to_start_from = gray_image.shape[0]
                # Start at halfway point because the highest point is always at the end of the bus
                # Jump by PIXEL_VALUE_TO_JUMP_FOR_X_VALUE for efficiency. Potential to improve algorithm here.
                for x in range(x_start + BUFFER_SPACE_FROM_LEFT_BLACK_BOX, x_end, PIXEL_VALUE_TO_JUMP_FOR_X_VALUE):
                    for y in range(y_value_to_start_from, 0, -1):

                        line_pixels = gray_image[y - NUM_OF_PIXEL_TO_AVERAGE : y, x]
                        avg_pixel_value = line_pixels.mean()
                        # Find a shift in pixel intensity
                        if avg_pixel_value < int(PIXEL_INTENSITY_VALUE * 1.9):
                            for index, pixel in enumerate(reversed(line_pixels)):
                                if pixel < int(PIXEL_INTENSITY_VALUE * 1.9):
                                    # Found black-to-white transition
                                    # Check if most_top_y has a y-value larger than current y, if larger it means it's positioned lower in the image.
                                    # And since we don't want to cut off any image, we find the y that has the smallest value, which indicates that it's at the
                                    # top part of the image
                                    if most_bot_y < y - index:
                                        most_bot_y = y - index
                                        # Check if this will lead to out-of-bound index error
                                        if most_bot_y + BUFFER_SPACE_TO_REFIND_SMALLEST_Y_VALUE > gray_image.shape[0] - 1:
                                            y_value_to_start_from = gray_image.shape[0] - 1
                                        else:
                                            y_value_to_start_from = most_bot_y + BUFFER_SPACE_TO_REFIND_SMALLEST_Y_VALUE
                                        # Found the transition, stop finding for this x-value
                                        break
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
        
        # Performing function
        print("Cropping image...")
        x_start, x_end, y_start, y_end = find_black_to_white_transition(self.original_image)
        # Update cropped coordinates
        self.image_dict["cropped_coordinates"] = {"xmin": x_start,
                                                "xmax": x_end,
                                                "ymin": y_start,
                                                "ymax": y_end}

    @time_func
    def segment_image(self, segment_size=640, overlap_percent=0.5):
        """
        Segments an image of any dimension into pieces of specified by <segment_size>,
        with a specified overlap percentage specified by <overlap_percent>.

        Args:
        segment_size (int): Integer number to specify the pixel length and width of the segment size. Default=640
        overlap_percent (float): The percentage of overlap between adjacent segments (0 to 1). Default=0.5, 50%

        Returns:
        None: The function saves the segmented images in it's np-array form to the segment_image_info dictionary.
        """
        print("Segmenting image...")
        # Read the image using OpenCV
        img = self.original_image.copy()

        # Crop the img copy based off the crop_image() function, which uses self.image_dict["cropped_coordinates"].
        img = img[self.image_dict["cropped_coordinates"]["ymin"]:self.image_dict["cropped_coordinates"]["ymax"], self.image_dict["cropped_coordinates"]["xmin"]:self.image_dict["cropped_coordinates"]["xmax"]]
    
        # Get the height and width of the image
        height, width = img.shape[:2]

        # Calculate the number of rows and columns required to segment the image
        overlap_pixels = int(segment_size * overlap_percent)
        segment_stride = segment_size - overlap_pixels
        num_rows = int(np.ceil((height - segment_size) / segment_stride)) + 1
        num_cols = int(np.ceil((width - segment_size) / segment_stride)) + 1

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
                self.segment_image_info['segment_{}_{}'.format(y_start + self.image_dict["cropped_coordinates"]["ymin"], x_start + self.image_dict["cropped_coordinates"]["xmin"])] = segment

    def get_segment_info(self):
        """
        Helper function that returns the dict segment_image_info dictionary. 
        Key contains the segment name, segment_0_0 means the image was taken from the xmin=0, ymin=0 portion of the cropped image.
        Value contains the np-array of the image.
        Returns: 
        - self.segment_image_info: (dict),
            key: segment name 
            value: np.array of segment image
        """
        return self.segment_image_info

def calling_relevant_class_function(cv2_image, segment_size=640, overlap_percent=0.5):
    """
    Takes an image as an input, and calls the relevant class function to do the necessary pre-processing for the image.

    Args:
    cv2_image: A cv2-formatted image, an np-array.
    segment_size (int): Integer number to specify the pixel length and width of the segment size. Default=640
    overlap_percent (float): The percentage of overlap between adjacent segments (0 to 1). Default=0.5, 50%

    Return:
    image_class: A class object that performs functions on the cv2 image inputted.
    image_class.get_segment_info(): a dict structure which contains all the image info and it's segment info

    """
    # Create an object
    image_class = ImageProcessor(input_cv2_image=cv2_image)
    # Calls crop function to crop away black and white line
    image_class.crop_image()
    # Calls the segment function to segment up the image
    image_class.segment_image(segment_size=segment_size, overlap_percent=overlap_percent)
    # get_segment_info returns a dict of segment info in the format of
    # key: segment_0_0
    # value: np.array
    return image_class, image_class.get_segment_info()

if __name__ == "__main__":
    test_image = cv2.imread(r"D:\leann\busxray_woodlands\annotations_adjusted\adjusted_1610_annotated.jpg")
    # Create an object
    test = ImageProcessor(input_cv2_image=test_image)
    # Calls crop function to crop away black and white line
    test.crop_image()
    # Calls the segment function to segment up the image
    test.segment_image(segment_size=640, overlap_percent=0.5)
    # get_segment_info returns a dict of segment info in the format of
    # key: segment_0_0
    # value: np.array
    print(test.get_segment_info())
