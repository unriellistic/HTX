"""
This script works only for the original "exp" directory-structure.

It does 2 function:
1) It goes through each folder in the "exp" folder and looks for the files with "annotated" (both .xml and .jpg) and copies 
them over into a folder called "annotated". 
2) It saves both the annotated.jpg as well as the annotated.xml, and relabels them according to the original exp folder annotation. 
E.g. if image and xml were saved in subdirectory named 355, it'll re-save the image and xml as 355_annotated.jpg and 355_annotated.xml

To alter the script to compile different files,
look at line 43 and update the <if "annotated" in file:> to contain whatever unique description the file name has, 
else if you want to compile all the files in the subdirectory, uncomment that line out, and alt-tab the function.

Input Arguments:
--root-dir: specifies the folder that contains a subdirectories of image and XML.
e.g. python compile_annotations_busxray.py --root-dir r"D:\leann\busxray_woodlands\exp"
--annotation-dir: specifies the folder that you want to save the compiled image and XML files in. If directory wasn't previously created, it'll create a new one at the path you specified.
e.g. python compile_annotations_busxray.py --annotation-dir r"D:\leann\busxray_woodlands\annotations"

Full example:
python compile_annotations_busxray.py --root-dir "D:\leann\busxray_woodlands\exp" --target-dir "D:\leann\busxray_woodlands\annotations"

This will cause the function to look at root directory at <exp> and saves the file at <annotations>.
@current_author: alp
@last modified: 12/4/2023 2:29pm
"""

import os, shutil, argparse

def compile_annotations(ROOT_DIR, ANNOTATION_DIR):
    
    # Create the output directory if it does not exist
    if not os.path.exists(ANNOTATION_DIR):
        os.makedirs(ANNOTATION_DIR)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", help="folder containing the X-ray images", default=r"exp")
    parser.add_argument("--target-dir", help="folder containing the annotation files", default=r"annotations")

    args = parser.parse_args()
    compile_annotations(args.root_dir, args.target_dir)