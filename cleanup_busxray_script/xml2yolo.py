""""
Script converts XML to YOLO format. 
Change the ROOT_DIR variable which contains both the image and xml file, and both with the same name.
It then saves the YOLO formatted .txt file into the same directory as well.
If no corresponding XML file is found, generates a blank txt file instead.
"""
import xml.etree.ElementTree as ET
import os
import general_scripts as gs
from tqdm import tqdm

def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

CLASS_LABELS = ["cig", "guns", "human", "knives", "drugs", "exp"]

ROOT_DIR = r"D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_dualenergy"
# identify all the image files in the folder (input directory)
files = gs.load_images(path_to_images=ROOT_DIR, file_type="all", recursive=True)

# loop through each image
for file in tqdm(files):
    filepath, basename = gs.path_leaf(file)

    # Only run the algorithm on segments with cleaned in their name
    # To run on the adjusted image, can change this check here
    if "cleaned" in basename:
            
        filename = os.path.splitext(basename)[0]
        label_file = os.path.join(filepath, f"{filename}.xml")
        result = []
        # check if the image contains the corresponding label file
        if not os.path.exists(label_file):
            print(f"{filename} label does not exist!")
        else:
            # parse the content of the xml file
            tree = ET.parse(label_file)
            root = tree.getroot()
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)

            for obj in root.findall('object'):
                label = obj.find("name").text
                # check for new CLASS_LABELS and append to list
                if label not in CLASS_LABELS:
                    CLASS_LABELS.append(label)
                index = CLASS_LABELS.index(label)
                pil_bbox = [int(x.text) for x in obj.find("bndbox")]
                yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
                # convert data to string
                bbox_string = " ".join([str(x) for x in yolo_bbox])
                result.append(f"{index} {bbox_string}")

        # generate a YOLO format text file for each xml file
        with open(os.path.join(filepath, f"{filename}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(result))

# generate the class file as reference
# with open(os.path.join(filepath, "class.txt"), 'w', encoding='utf8') as f:
#     f.write(json.dumps(CLASS_LABELS))