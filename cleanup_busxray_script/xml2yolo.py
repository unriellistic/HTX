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

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_xml_to_yolo(root_dir, classes):

    # identify all the image files in the folder (input directory)
    files = gs.load_files(path_to_files=root_dir, file_type="images", recursive=True)
    
    print("Converting xml file to txt...")
    # loop through each image
    for file in tqdm(files):
        filepath, basename = gs.path_leaf(file)

        # Only run the algorithm on segments with cleaned in their name
        # To run on the adjusted image, can change this check here
        if "cleaned" in basename:
            filename = os.path.splitext(basename)[0]
            label_file = os.path.join(filepath, f"{filename}.xml")
            out_file = open(os.path.join(filepath, f"{filename}.txt"), 'w')
            # check if the image contains the corresponding label file
            if not os.path.exists(label_file):
                print(f"{label_file} label does not exist! Creating empty txt file...")
                out_file.close()
            else:
                # parse the content of the xml file
                tree = ET.parse(label_file)
                root = tree.getroot()
                size = root.find('size')
                w = int(size.find('width').text)
                h = int(size.find('height').text)

                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    cls = obj.find('name').text
                    if cls not in classes or int(difficult)==1:
                        continue
                    cls_id = classes.index(cls)
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = convert((w,h), b)
                    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                # close file
                out_file.close()

if __name__ == "__main__":
    print("This script is not meant to be run directly.")
    print("Please import it as a module and call the convert_xml_to_yolo() function, unless you're debuging.")
    ROOT_DIR = r"D:\BusXray\scanbus_training\temp"
    CLASSES = ["cig", "guns", "human", "knives", "drugs", "exp"]
    convert_xml_to_yolo(root_dir=ROOT_DIR, classes=CLASSES)
