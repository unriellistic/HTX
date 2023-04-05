"""
A place to store common functions:
- load_images_from_folder: Loading all images from a directory
    - function: def load_images_from_folder(folder, file_type="all"):
    - e.g. To load all image file type:
        load_images_from_folder(<filepath_to_folder>, file_type="all")
    - e.g. To load only png file type:
        load_images_from_folder(<filepath_to_folder>, file_type=".png")

- save_to_excel: saves information to an excel file in the current directory
    - function: def save_to_excel(info, columns, file_name='test', sheet_name='sheet1', index=False):
    - The details can be found in the comments above the function below
    - e.g. save_to_excel(info=[['01feb', true, xray_012345],['02feb',false, xray_12345],...,[...]],
                        columns=[date, detected, filename],
                        file_name='name_of_excel_file_you_want_to_save_as',
                        sheet_name='name_of_excel_sheet_you_want_to_save_as',
                        index=False)

- change_file_extension: Finds the last '.' and changes the extension to whatever u input instead.
    - function: change_file_extension(filename, new_file_extension):
    - e.g. change_file_extension("xray_scan.tiff", ".jpg")
"""

import os
def load_images_from_folder(folder, file_type="all"):
    """
    Function returns a list of images found in the directory

    folder: A variable that contains the full path to the directory of interest
    file_type: A variable that specifies what file type to look for. Default = "all"
    """
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

def save_to_excel(info, columns, file_name='test', sheet_name='sheet1', index=False):
    """
    Function saves the info into an excel.

    info: A variable that contains a list of a list of the data
        - e.g.: info=[['01Feb', '14.5', 'True', 'Has 2 objects'], [...] ... ]
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
    df = pd.DataFrame(info, columns=columns)
    print("df:", df)
    df.to_excel(f'{file_name}.xlsx', sheet_name=sheet_name, index=index)
    print(f"Excel file saved as {file_name}")



def change_file_extension(filename, new_file_extension):
    """
    Function that replaces file extension names.

    filename: A variable that contains the name of the file including the extension
        - e.g.: image_2023.png
    new_file_extension: the new string that you would like the file to end with instead
        - e.g.: tif (image_2023.png -> image_2023.tif)
    """

    # Returns a list of "obj" element, which in our case, is the "."
    def indexes(iterable, obj):
        return (index for index, elem in enumerate(iterable) if elem == obj)
    
    # Get a list of "."
    list_of_index_of_element = list(indexes(filename, "."))

    # Find the last ".", minus 1 to get the index before the ".", which is the filename without the extension
    # I didn't add the "." is so that people have to put it in by themselves. Rationale being, sometimes I want to get rid of the extension totally.
    filename = filename[0:list_of_index_of_element[-1]] + new_file_extension

    return filename

import ntpath # Rename path variables
def path_leaf(path):
    """
    Function that returns the path that leads to the filename, and the filename itself.
    path: A variable that contains a path of the file
        - e.g.: c:\\user\\alp.txt
        - head: c:\\user ; tail: alp.txt
    """
    head, tail = ntpath.split(path)
    return head, (tail or ntpath.basename(head))

