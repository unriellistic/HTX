"""
Just a mini testing file for me to try out python logic for debugging.
"""
import numpy as np
height = 1000
segment_size = 2000
# Check if segment size specified is within height/width of image
if segment_size > height:
    print(f"Segment size larger than image's height. Changing segment_size to be {height}")
    segment_size = height
elif segment_size > width:
    print(f"Segment size larger than image's width. Changing segment_size to be {width}")
    segment_size = width
overlap_pixels = int(segment_size * 0.5)
segment_stride = segment_size - overlap_pixels
print("(height - segment_size)/ segment_stride:", (height - segment_size)/ segment_stride)
print("np.ceil((height - segment_size) / segment_stride:", np.ceil((height - segment_size) / segment_stride))
print(int(np.ceil((height - segment_size) / segment_stride)))
num_rows = int(np.ceil((height - segment_size) / segment_stride)) + 1