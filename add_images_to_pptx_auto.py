"""
The script automates the creation of multiple slides with a specific content:
- One picture
- Two subtexts (Filename, Remarks)
"""
import collections 
import collections.abc
from pptx import Presentation
import os

prs = Presentation()

# If want to see slide layout, uncomment below
# for slide in prs.slide_layouts:
#     for shape in slide.placeholders:
#         print('%d %d %s' % (prs.slide_layouts.index(slide), shape.placeholder_format.idx, shape.name))
class MySlide:
    def __init__(self, data):
        self.layout = prs.slide_layouts[data[3]]
        self.slide = prs.slides.add_slide(self.layout)
        self.title = self.slide.shapes.title
        self.title.text = data[0]
        self.subtitle = self.slide.placeholders[1]
        self.subtitle.text = data[1]

        for shape in self.slide.placeholders:
            print('%d %s %s' % (
                shape.placeholder_format.idx,
                shape.placeholder_format.type,
                shape.name))
            print()
        if data[2] != "":
            self.img = self.slide.placeholders[1].insert_picture(data[2])
            available_width = self.img.width
            available_height = self.img.height
            image_width, image_height = self.img.image.size
            placeholder_aspect_ratio = float(available_width) / float(available_height)
            image_aspect_ratio = float(image_width) / float(image_height)

            # Get initial image placeholder left and top positions
            pos_left, pos_top = self.img.left, self.img.top

            self.img.crop_top = 0
            self.img.crop_left = 0
            self.img.crop_bottom = 0
            self.img.crop_right = 0

            # ---if the placeholder is "wider" in aspect, shrink the self.img width while
            # ---maintaining the image aspect ratio
            if placeholder_aspect_ratio > image_aspect_ratio:
                self.img.width = int(image_aspect_ratio * available_height)
                self.img.height = available_height

            # ---otherwise shrink the height
            else:
                self.img.height = int(available_width / image_aspect_ratio)
                self.img.width = available_width

            # Set the self.img left and top position to the initial placeholder one
            self.img.left, self.img.top = pos_left, pos_top

            # Or if we want to center it vertically:
            # self.img.top = self.img.top + int(self.img.height/2)
           
slides = []
import pandas as pd
df = pd.read_excel(r"D:\\Eyefox Data\\CAG\\Staged Scene\\EYEFOX_STATS_16JAN.xls", sheet_name='Data for 160123')=

print(df.columns.tolist())
info_about_images = []

for index, row in df.iterrows():
    # Overall = Whether Eyefox got detect
    # Filename = Top Filename (View 1)
    # 'Unnamed: 15' = Side Filename (View 2)
    # 'Unnamed: 16' = X-Ray Filename...""
    if index == 0:
        continue
    else:
        print(f"Current row: {str(index)}\n", row['Overall'], row['Top View (view 1)'], row['Side View (view 2)'], row['Filename'], row['Unnamed: 15'], row['Unnamed: 16'], row['Legend/Notes'])
        info_about_images.append((row['Overall'], row['Top View (view 1)'], row['Side View (view 2)'], row['Filename'], row['Unnamed: 15'], row['Unnamed: 16'], row['Legend/Notes']))

# From excel, extract filenames as tuple: EYEFOX_TOP_DETECTED?, EYEFOX_SIDE_DETECTED?, EYEFOX_TOP, EYEFOX_SIDE, XRAY
# Iterate through each tuple, first 2 index take from EYEFOX folder, last index take from XRAY folder
#   # Go through XRAY first
#   Append XRAY_0_A into slide
#   If tuple(0)==True:
#       Append in EYEFOX images

EYEFOX_IMG_DIR = "D:\\Eyefox Data\\CAG\\Staged Scene\\test"
XRAY_IMG_DIR = "D:\\Eyefox Data\\CAG\\Staged Scene\\test TIF"

for detail in info_about_images:
    
    # Detail is a tuple with information stored in this order:
    # 0: Overall detection
    # 1: Top view detection
    # 2: Side view detection
    # 3: Eyefox top filename
    # 4: Eyefox side filename
    # 5: XRAY filename
    # 6: Legend/notes
    eyefox_detection = ""
    top_view_present = False
    
    # Create first slide

    # If overall detection is True
    if detail[0] == 1:
        eyefox_detection = eyefox_detection + "YES"
    else:
        eyefox_detection = "NO"
    top_xray_image = XRAY_IMG_DIR + "\\" + detail[5] + "_0_B"
    side_xray_image = XRAY_IMG_DIR + "\\" + detail[5] + "_0_A"
    slides.append([f"File: {detail[5]}\nEyefox detection:{eyefox_detection}", f"Remarks: {detail[6]}", side_xray_image, 8])
    slides.append([f"File: {detail[5]}\nEyefox detection:{eyefox_detection}", f"Remarks: {detail[6]}", top_xray_image, 8])

    # If Side View detection is True
    if detail[2] == 1:
        side_eyefox_image = EYEFOX_IMG_DIR + "\\" + detail[4]
        slides.append([f"File: {detail[4]}\nEyefox detection:{eyefox_detection}", f"Remarks: {detail[6]}", side_eyefox_image, 8])
    # If Top View detection is True
    if detail[1] == 1:
        top_eyefox_image = EYEFOX_IMG_DIR + "\\" + detail[3]
        slides.append([f"File: {detail[3]}\nEyefox detection:{eyefox_detection}", f"Remarks: {detail[6]}", top_eyefox_image, 8])


for each_slide in slides:
    MySlide(each_slide)

prs.save("test.pptx")
os.startfile("stack.pptx")