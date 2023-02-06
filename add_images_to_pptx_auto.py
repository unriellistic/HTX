"""
The script automates the creation of multiple slides with a specific content:
- One picture
- Two subtexts (Filename, Remarks)
"""
from pptx import Presentation
import os

prs = Presentation()

for slide in prs.slide_layouts:
    for shape in slide.placeholders:
        print('%d %d %s' % (prs.slide_layouts.index(slide), shape.placeholder_format.idx, shape.name))
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
df = pd.read_excel(r'Path of Excel file\File name.xlsx', sheet_name='your Excel sheet name')
counter = 0 # Assuming that files are already in order, else need to find name by name.

for images in image_folder:
    # image just needs to be image name
    if images.endswith(".png"):
        slides.append([f"File: {filename[:-3]}", f"Remarks: {df['Remarks'].iloc[counter]}", images, 8])
        counter += 1


for each_slide in slides:
    MySlide(each_slide)

prs.save("stack.pptx")
os.startfile("stack.pptx")

# python_pptx-0.6.21.dist-info