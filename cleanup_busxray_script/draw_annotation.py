"""
A script to check if the xml2yolo works as intended. Plots the annotation on the image. Code to change at the bottom
"""
import argparse
from PIL import Image, ImageDraw
import cv2, os, random

"""
To check YOLO formatted annotations (.txt files)
"""
def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_YOLO_annotations(image_path, txt_path, save_path):
    """
    This function will add rectangle boxes on the images.
    """
    # flag_people_or_car_data = 0  #变量 代表类别
    source_file = open(txt_path) if os.path.exists(txt_path) else []
    image = cv2.imread(image_path)
    try:
        height, width, channels = image.shape
    except:
        print('no shape info.')
        return 0

    box_number = 0
    for line in source_file:  # 例遍 txt文件得每一行
        staff = line.split()  # 对每行内容 通过以空格为分隔符对字符串进行切片

        x_center, y_center, w, h = float(
            staff[1])*width, float(staff[2])*height, float(staff[3])*width, float(staff[4])*height
        x1 = round(x_center-w/2)
        y1 = round(y_center-h/2)
        x2 = round(x_center+w/2)
        y2 = round(y_center+h/2)

        plot_one_box([x1, y1, x2, y2], image, line_thickness=None)

        cv2.imwrite(save_path, image)
        box_number += 1
    return box_number

"""
To check Pascal VOC images (.xml files)
"""
import xml.etree.ElementTree as ET
import cv2
import general_scripts as gs
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

    if args.format not in ["yolo", "voc"]:
        raise ValueError("Format must be either 'yolo' or 'voc'")
    
    if args.format == "yolo":
        draw_YOLO_annotations(args.image_path, args.label_path, args.output)
    elif args.format == "voc":
        draw_xml_annotations(args.image_path, args.label_path, args.output)