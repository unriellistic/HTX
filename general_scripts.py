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
    list_of_image_file_format = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')
    for filename in os.listdir(folder):
        if file_type == "all":
            if filename.lower().endswith(list_of_image_file_format) and filename is not None:
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
    print(f"Excel file saved as {file_name}")

"""
Function that replaces file extension names
"""

def change_file_extension(filename, new_file_extension):

    # Returns a list of "obj" element, which in our case, is the "."
    def indexes(iterable, obj):
        return (index for index, elem in enumerate(iterable) if elem == obj)
    
    # Get a list of "."
    list_of_index_of_element = list(indexes(filename, "."))

    # Find the last ".", minus 1 to get the index before the "."
    filename = filename[0:list_of_index_of_element[-1]] + new_file_extension

    return filename

