import numpy as np

class ImageSlice:
    """
    Class for an image slice from slice_image.
    Contains the image itself, and the x and y coordinates of the slice's origin.
    """
    def __init__(self, x_offset: int, y_offset: int, img: np.ndarray):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.img = img

def slice_image(img: np.ndarray, segment_size: int = 640, overlap_portion: float = 0.5) -> list[ImageSlice]:
    """
    Slices an image of any dimension into pieces of 640 by 640 with a specified overlap percentage.

    Args:
    img (np.ndarray): The image to be sliced.
    overlap_portion (float): The amount of overlap between adjacent slices (0 to 1).

    Returns a list of ImageSlice.
    """
    height, width = img.shape[:2]

    # Calculate the number of rows and columns required to segment the image
    overlap_pixels = int(segment_size * overlap_portion)
    segment_stride = segment_size - overlap_pixels
    num_rows = int(np.ceil((height - segment_size) / segment_stride)) + 1
    num_cols = int(np.ceil((width - segment_size) / segment_stride)) + 1

    slices = []

    for row in range(num_rows):
        for col in range(num_cols):
            y_start = row * segment_stride
            y_end = y_start + segment_size
            x_start = col * segment_stride
            x_end = x_start + segment_size

            # Check if the remaining section of the image is less than segment_size. If so, move the slice back
            if y_end > height:
                y_end = height
                y_start = height - segment_size
            if x_end > width:
                x_end = width
                x_start = width - segment_size

            imgslice = ImageSlice(x_start, y_start, img[y_start:y_end, x_start:x_end])
            slices.append(imgslice)

    return slices