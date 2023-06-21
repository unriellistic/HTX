"""
A script to check if the xml2yolo works as intended. Plots the annotation on the image. Code to change at the bottom
"""
import argparse
from PIL import Image, ImageDraw
import cv2

CLASS_NAMES = ["cig", "guns", "human", "knives", "drugs", "exp"]

"""
To check YOLO formatted annotations (.txt files)
"""
def draw_yolo_annotations(image_path: str, label_path: str, save_path: str = None, display: bool = True) -> None:
    annotated_image = cv2.imread(image_path)
    with open(label_path, "r") as f:
        for line in f:
            line = line.split()
            class_name = CLASS_NAMES[int(line[0])]

            # YOLO format: xmid, ymid, w, h
            x, y, w, h = map(float, line[1:])

            # Convert to x1, y1, x2, y2
            xmin = int((x - w / 2) * annotated_image.shape[1])
            ymin = int((y - h / 2) * annotated_image.shape[0])
            xmax = int((x + w / 2) * annotated_image.shape[1])
            ymax = int((y + h / 2) * annotated_image.shape[0])

            # Draw the bounding box
            cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 4)
            cv2.putText(annotated_image, class_name, (int(xmin)-4, int(ymin)-14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # save the image
    if save_path:
        cv2.imwrite(save_path, annotated_image)

    # Display the annotated image
    if display:
        cv2.namedWindow('Annotated Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Annotated Image', annotated_image.shape[1], annotated_image.shape[0])
        cv2.imshow("Annotated Image", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

"""
To check Pascal VOC images (.xml files)
"""
import xml.etree.ElementTree as ET
import cv2
from general_scripts import general_scripts as gs
def draw_image(img: Image, bboxes: list[list], save_path: str, display: bool = False):
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=2)
    img.save(save_path)

    if display:
        img.show()

def draw_xml_annotations(image_path: str, xml_path: str):
    # Load the XML file
    tree = ET.parse(xml_path)

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
    img = cv2.imread(image_path)
    cv2.imwrite(gs.change_file_extension(image_path, ".jpg"), img)
    img = Image.open(gs.change_file_extension(image_path, ".jpg"))
    draw_image(img, bbox_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str)
    parser.add_argument("--label-path", type=str)
    parser.add_argument("--output", type=str, default="output.jpg")
    parser.add_argument("--format", type=str, default="yolo")
    args = parser.parse_args()

    if args.format == "yolo":
        draw_yolo_annotations(args.image_path, args.label_path, args.output)
    elif args.format == "voc":
        draw_xml_annotations(args.image_path, args.label_path, args.output)
    else:
        raise ValueError("Format must be either 'yolo' or 'voc'")