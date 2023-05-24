"""
This script works only for the original "exp" directory-structure.
Note: the .xml annotation must have the same name as the image file

It does 2 function:
1) It goes through each folder in the "exp" folder and looks for the files with "annotated" (both .xml and .jpg) and copies 
them over into a folder called "annotated". 
2) It saves both the annotated.jpg as well as the annotated.xml, and relabels them according to the original exp folder annotation. 
E.g. if image and xml were saved in subdirectory named 355, it'll re-save the image and xml as 355_annotated.jpg and 355_annotated.xml

To alter the script to compile different files,
look at line 36 and update the <if "annotated" in file:> to contain whatever unique description the file name has, 
else if you want to compile all the files in the subdirectory, delete that portion.

Input Arguments:
--root-dir: specifies the folder that contains a subdirectories of image and XML.
e.g. python compile_annotations_busxray.py --root-dir r"D:\leann\busxray_woodlands\exp"
--annotation-dir: specifies the folder that you want to save the compiled image and XML files in. If directory wasn't previously created, it'll create a new one at the path you specified.
e.g. python compile_annotations_busxray.py --annotation-dir r"D:\leann\busxray_woodlands\annotations"

Full example:
python compile_annotations_busxray.py --root-dir "D:\leann\busxray_woodlands\exp" --target-dir "D:\leann\busxray_woodlands\annotations"

This will cause the function to look at root directory at <exp> and saves the file at <annotations>.
@current_author: alp
@last modified: 12/5/2023 3:29pm
"""

import os, shutil, argparse, general_scripts as gs
from tqdm import tqdm

def compile_annotations(ROOT_DIR, ANNOTATION_DIR):
    
    # Create the output directory if it does not exist
    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    image_files = [file for file in gs.load_files(ROOT_DIR, recursive=True, file_type="all") if ".jpg" in file and "annotated" not in file]
    annotation_files = [file for file in gs.load_files(ROOT_DIR, recursive=True, file_type=".xml") if "annotated" in file]
    files = image_files + annotation_files

    print(f"Copying over from {ROOT_DIR} ...")
    for file in tqdm(files):
        file_path, file_name = gs.path_leaf(file)
        _, subdir_name = gs.path_leaf(file_path)
        _, file_type = os.path.splitext(file_name)
        new_name = os.path.join(ANNOTATION_DIR, f"{subdir_name}_annotated{file_type}")
        shutil.copy(file, new_name)
        print(f"Copied over '{file}' to '{ANNOTATION_DIR}'")                   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", help="folder containing the X-ray images", default=r"exp")
    parser.add_argument("--target-dir", help="path to folder to store consolidated image + annotation files", default=r"compiled_annotations")
    args = parser.parse_args() 
    """
    Just to clear some bug that occurs if target-dir default options settings is left default.
    """
    # Get path to root directory folder
    # Check if default parameter is applied, if so get full path.
    if args.target_dir == "compiled_annotations":
        path_to_target_dir = os.path.join(os.getcwd(), args.target_dir)
    # Else, use path specified by user
    else:
        path_to_target_dir = args.target_dir

    compile_annotations(args.root_dir, path_to_target_dir)