"""
A place to store common functions:
- load_images_from_folder: Loading all images from a directory
- save_to_excel: saves information to an excel file in the current directory
"""

"""
Function returns a list of images found in the directory

folder: A variable that contains the full path to the directory of interest
file_type: A variable that specifies what file type to look for. Default = "all"
"""
import os
def load_images_from_folder(folder, file_type="all"):
    images = []
    for filename in os.listdir(folder):
        if file_type == "all":
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) and filename is not None:
                images.append(filename)
        else:
            if filename.lower().endswith(file_type) and filename is not None:
                images.append(filename)
    return images


"""
Function saves the info into an excel.

info: A variable that contains a list of a list of the data
    - e.g.: info=[['01Feb', '14.5', 'True', 'Has 2 objects']
columns: A variable that contains a list of the column headers
    - e.g.: columns=['filename', 'detected_threshold', 'blurry', 'Remarks']
file_name: A variable that specifies what filename to save as. Default is 'test'
    - e.g.: 'stats_01FEB'
sheet_name: A variable that specifies what sheetname to save as. Default is sheet1
    - e.g.: 'stats_01FEB'
index: A variable that specifies whether you want to index or not. Default is False
    - e.g.: 'stats_01FEB'
"""
import pandas as pd
def save_to_excel(info, columns, file_name='test', sheet_name='sheet1', index=False):
    df = pd.DataFrame(info, columns=columns)
    print("df:", df)
    df.to_excel(f'{file_name}.xlsx', sheet_name=sheet_name, index=index)
    print(f"Excel file saved to {info}")
