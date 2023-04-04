"""
This scripts goes through each folder in the "exp" folder and looks for the files with "annotated" (both .xml and .jpg) and copies 
them into a folder called "annotated", and relabels them according to their annotations.
"""

import os
import shutil

# Constants to change for differing paths
ROOT_DIR = r"D:\leann\busxray_woodlands\exp"
ANNOTATION_DIR = r"D:\leann\busxray_woodlands\annotations"

def listdirs(ROOT_DIR):
    for root, dirs, _ in os.walk(ROOT_DIR):

        # Go through the list of subdirectories
        for subdir in dirs:
         
            # Go through each file in the list
            for file in os.listdir(os.path.join(root, subdir)):
        
                # Check if it's the annotation file
                if "annotated" in file:
                    _, file_type = os.path.splitext(file)
                    old_name = os.path.join(os.path.abspath(root), os.path.join(subdir, file))
                    new_name = os.path.join(ANNOTATION_DIR, f"{subdir}_annotated{file_type}")
                    shutil.copy(old_name, new_name)
                    print(subdir)

if __name__ == "__main__":
    listdirs(ROOT_DIR)