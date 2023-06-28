"""
A script to check if the YOLO annotations are within the bounds of the image. Specify the directory containing the .txt files. 
(Can be in subdirectories as well, the script will search recursively)

If there is an out-of-bound error, script will save a list of the files with errors as error_list.txt in the current working directory.
Note: The script does not check if the annotations are in the correct format, only if the assumed correctly-formatted values are within the bounds of the image.
@Created: 28/6/2023
@Author: Alphaeus
"""
import os
import general_scripts as gs
from tqdm import tqdm

def check_annotations(directory):
    error_list = []
    txt_files = gs.load_files(directory, ".txt", recursive=True)
    for filename in tqdm(txt_files):
        error_found_question_mark = False
        with open(filename, "r") as file:
            lines = file.readlines()
            for line in lines:
                values = line.split()
                for value in values[1:]:
                    if float(value) < 0 or float(value) > 1:
                        error_list.append(filename)
                        error_found_question_mark = True
                        break
                if error_found_question_mark:
                    break

    if error_list:
        with open("error_list.txt", "w") as file:
            for error in error_list:
                file.write(error + "\n")
        print(f"Error list saved as error_list.txt at {os.getcwd()}")
    else:
        print("No errors found.")

# Specify the directory containing the .txt files
directory = r"C:\dataset_8_dualenergy\labels"

check_annotations(directory)
