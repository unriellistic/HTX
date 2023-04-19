"""
NOT COMPLETE
This Python script uses the watchdog library to monitor a folder for new images. When a new image is added to the folder, 
the script processes the image using an AI object detection algorithm (replace INPUT_FUNCTION with your own function) and 
saves the output image to another folder.

To use this script, you will need to install the opencv-python, numpy, and watchdog libraries. Then, modify the input_dir 
and output_dir variables to match the input and output directories on your system.

Next, create an object of the ImageProcessor class and pass in the input and output directories as arguments. 
This object will handle the image processing.

Finally, set up a FileSystemEventHandler object to handle file creation events, and use an Observer object to monitor 
the input directory for changes. When a new file is created, the ImageHandler class will call the process_image method 
of the ImageProcessor object to process the new image.

To run the script, simply execute the Python file in a terminal or command prompt window. The script will run continuously 
in the background, monitoring the input directory for changes. To stop the script, press Ctrl+C in the terminal or command prompt window.

"""

import os
import cv2
import numpy as np
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ImageProcessor:
    # Define ImageProcessor class with input and output directories
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    # Define function to process image
    def process_image(self, filename):
        # Check that the file is a JPEG image
        if not filename.endswith(".jpg"):
            return
        # Define input and output paths for image
        input_path = os.path.join(self.input_dir, filename)
        output_path = os.path.join(self.output_dir, filename)

        # Load image
        image = cv2.imread(input_path)

        # Run image through AI detection algorithm
        # Replace INPUT_FUNCTION with your own function
        image = INPUT_FUNCTION(image)

        # Save output image
        cv2.imwrite(output_path, image)


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

    # Create ImageProcessor object
    processor = ImageProcessor(input_dir, output_dir)

    # Set up watchdog observer to monitor input directory for changes
    event_handler = ImageHandler(processor)
    observer = Observer()
    observer.schedule(event_handler, input_dir, recursive=False)
    observer.start()

    try:
        # Keep script running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
