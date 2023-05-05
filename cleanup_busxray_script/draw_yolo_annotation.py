"""
A script to check if the xml2yolo works as intended.
"""
from PIL import Image, ImageDraw

image_filename = r"D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_monochrome\adjusted_PC3031K Scania K410 49 seats-Threat-227-temp_image_low_segmented\segment_320_0_cleaned.tif"
label_filename = r"D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_monochrome\adjusted_PC3031K Scania K410 49 seats-Threat-227-temp_image_low_segmented\segment_320_0_cleaned.txt"


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]

def draw_image(img, bboxes):
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=2)
    img.save("example.jpg")
    img.show()

# bboxes = []
# img = Image.open(image_filename)
# with open(label_filename, 'r', encoding='utf8') as f:
#     for line in f:
#         data = line.strip().split(' ')
#         bbox = [float(x) for x in data[1:]]
#         bboxes.append(yolo_to_xml_bbox(bbox, img.width, img.height))

# draw_image(img, bboxes)

quick_test = r"D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_monochrome\adjusted_PC3031K Scania K410 49 seats-Threat-227-temp_image_low.tif"
quick_test_xml = r"D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_monochrome\adjusted_PC3031K Scania K410 49 seats-Threat-227-temp_image_low.xml"

import xml.etree.ElementTree as ET

# Load the XML file
tree = ET.parse(quick_test_xml)

# Get the root element
root = tree.getroot()

# Define a list to hold the extracted bounding box data
bbox_list = []

# Loop through each object element in the XML and extract the bounding box information
for obj in root.findall('object'):
    xmin = int(obj.find('bndbox/xmin').text)
    ymin = int(obj.find('bndbox/ymin').text)
    xmax = int(obj.find('bndbox/xmax').text)
    ymax = int(obj.find('bndbox/ymax').text)
    bbox_list.append([xmin, ymin, xmax, ymax])

import cv2
import general_scripts as gs
img = cv2.imread(quick_test)
cv2.imwrite(gs.change_file_extension(quick_test, ".jpg"), img)
img = Image.open(gs.change_file_extension(quick_test, ".jpg"))
img.show()
# img = img.convert("RGB")
draw_image(img, bbox_list)