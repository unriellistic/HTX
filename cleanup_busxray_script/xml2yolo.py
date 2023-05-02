""""
Script converts XML to YOLO format. Change the ROOT_DIR variable which contains both the image and xml file, and both with the same name.
It then saves the YOLO formatted .txt file into the same directory as well.
"""
import xml.etree.ElementTree as ET
import os
import json
import general_scripts as gs
from tqdm import tqdm

def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


classes = ["cig", "guns", "human", "knives", "drugs", "exp"]
# input_dir = r"D:\BusXray\scanbus_training\Compiled_Threat_Images\Original files_adjusted\adjusted_PA8506K Higer 49 seats-Threat-4-final_color_segmented"
# output_dir = r"D:\BusXray\scanbus_training\Compiled_Threat_Images\Original files_adjusted\adjusted_PA8506K Higer 49 seats-Threat-4-final_color_segmented"
# image_dir = r"D:\BusXray\scanbus_training\Compiled_Threat_Images\Original files_adjusted\adjusted_PA8506K Higer 49 seats-Threat-4-final_color_segmented"

ROOT_DIR = r"D:\BusXray\scanbus_training\Segmented files"
SCAN_BUS_DIR = r"D:\BusXray\scanbus_training\Segmented files"
# identify all the xml files in the annotations folder (input directory)
files = gs.load_images(path_to_images=ROOT_DIR, file_type=".xml", recursive=True)

# loop through each 
for fil in tqdm(files):
    filepath, basename = gs.path_leaf(fil)
    filename = os.path.splitext(basename)[0]
    # check if the label contains the corresponding image file
    if not os.path.exists(os.path.join(filepath, f"{filename}.jpg")):
        print(f"{filename} image does not exist!")
        continue

    result = []

    # parse the content of the xml file
    tree = ET.parse(fil)
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    for obj in root.findall('object'):
        label = obj.find("name").text
        # check for new classes and append to list
        if label not in classes:
            classes.append(label)
        index = classes.index(label)
        pil_bbox = [int(x.text) for x in obj.find("bndbox")]
        yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
        # convert data to string
        bbox_string = " ".join([str(x) for x in yolo_bbox])
        result.append(f"{index} {bbox_string}")

    if result:
        # generate a YOLO format text file for each xml file
        with open(os.path.join(filepath, f"{filename}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(result))

# generate the classes file as reference
with open(os.path.join(filepath, "classes.txt"), 'w', encoding='utf8') as f:
    f.write(json.dumps(classes))