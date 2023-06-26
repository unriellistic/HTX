"""
A place to store common functions:
1) load_files: Loading all images from a directory
    - function: def load_files(folder, file_type="images"):
    - e.g. To load all image file type:
        load_files(<filepath_to_folder>, file_type="images")
    - e.g. To load only png file type:
        load_files(<filepath_to_folder>, file_type=(".png"))

2) save_to_excel: saves information to an excel file in the current directory
    - function: def save_to_excel(info, columns, file_name='test', sheet_name='sheet1', index=False):
    - The details can be found in the comments above the function below
    - e.g. save_to_excel(info=[['01feb', true, xray_012345],['02feb',false, xray_12345],...,[...]],
                        columns=[date, detected, filename],
                        file_name='name_of_excel_file_you_want_to_save_as',
                        sheet_name='name_of_excel_sheet_you_want_to_save_as',
                        index=False)

3) change_file_extension: Finds the last '.' and changes the extension to whatever u input instead.
    - function: change_file_extension(filename, new_file_extension):
    - e.g. change_file_extension("xray_scan.tiff", ".jpg")

4) path_leaf(path): Returns the basepath, filename.
5) print_nice_lines(character="=", height=1): prints a nice terminal width sized line.

@author: alp
@last modified: 31/5/23 10:00am
"""

import os
from pathlib import Path
from typing import Union, Tuple


def load_files(path_to_files: Union[str, Path], file_type: str="images", recursive: bool=False) -> list:
    """
    Function returns a list of full paths to images found in the path_to_image user specify.

    Args:
        path_to_files: A path variable that contains the full path to the directory of interest, which contains images. 
        file_type: A tuple or a string that specifies what file type to look for. e.g. (".jpg", ".tiff") or ".jpg". Default = "images"
        recursive: A boolean variable that specifies whether the function should look recursively into each folder in the directory specified. default=False
        exclude_string: A string variable that specifies the pattern of string to avoid collecting

    Returns:
        images: A list containing the full paths to the images.
    """
    p = str(Path(path_to_files).absolute())  # os-agnostic absolute path

    images_path = []
    list_of_image_file_format = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')

    def check_if_file_is_an_image(path_to_files):
        # Append only images
        if file_type == "images":
            if path_to_files.lower().endswith(list_of_image_file_format) and path_to_files is not None:
                images_path.append(path_to_files)
        # Append every file type
        elif file_type == "all":
            if path_to_files is not None:
                images_path.append(path_to_files)
        # Append specific file type
        else:
            if path_to_files.lower().endswith(file_type) and path_to_files is not None:
                images_path.append(path_to_files)
                
    # Checks if path specified is a file, folder, or a directory of subdirectories
    if os.path.isfile(p):
        check_if_file_is_an_image(p)
    
    # Checks for current directory's files
    elif os.path.isdir(p) and not recursive:
        print("Collecting a list of images from", p)
        for file in os.listdir(p):
            if os.path.isfile(os.path.join(p, file)):
                check_if_file_is_an_image(os.path.join(p, file))

    # Checks for current directory file and recursively searches each folder
    elif os.path.isdir(p) and recursive:
        print("Collecting a list of images from", p)
        for root, _, files in os.walk(p):
            for file in files:
                check_if_file_is_an_image(os.path.join(root, file))
    else:
        raise Exception(f'ERROR: {p} does not exist')

    return images_path

def save_to_excel(info, columns, file_name='test', sheet_name='sheet1', index=False):
    """
    Function saves the info into an excel.

    Args:
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
    Returns:
        None: file is saved within function call itself
    """
    import pandas as pd
    df = pd.DataFrame(info, columns=columns)
    print("df:", df)
    df.to_excel(f'{file_name}.xlsx', sheet_name=sheet_name, index=index)
    print(f"Excel file saved as {file_name}")



def change_file_extension(filename: Union[str, Path], new_file_extension:str) -> str:
    """
    Function that replaces file extension names.

    Args:
        filename: A variable that contains the name of the file including the extension
            - e.g.: image_2023.png
        new_file_extension: the new string that you would like the file to end with instead
            - e.g.: tif (image_2023.png -> image_2023.tif)
    
    Returns:
        filename: new modified name of the file
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

def path_leaf(path: Union[str, Path]) -> Tuple[Union[str, Path], str]:
    """
    Function that returns the path that leads to the filename, and the filename itself.
    Args:
        path: A variable that contains a path of the file
            - e.g.: c:\\user\\alp.txt
            - head: c:\\user ; tail: alp.txt
    Returns:
        head: root path to current file
        tail: name of current file
    """
    import ntpath # Rename path variables
    head, tail = ntpath.split(path)
    return head, (tail or ntpath.basename(head))

def print_nice_lines(character: str = "=", height: int = 1) -> None:
    """
    Prints a line of characters until the end of the terminal window.
    
    Args:
        character (str, optional): The character to print. Defaults to "=".
    Returns:
        None
    """
    import shutil
    # Get the size of the terminal window
    terminal_width, _ = shutil.get_terminal_size()

    # Adjust the line_length based on the terminal width
    line_length = int(terminal_width / len(character))  # Adjust this value if needed

    for _ in range(height):
        for _ in range(line_length):
            print(character, end="")

    # Move to the next line
    print()




