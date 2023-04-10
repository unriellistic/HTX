"""
This scripts goes through each folder in the "exp" folder and looks for the files with "annotated" (both .xml and .jpg) and copies 
them into a folder called "annotated", and relabels them according to their annotations.
"""

import os, shutil, argparse

def compile_annotations(ROOT_DIR, ANNOTATION_DIR):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", help="folder containing the X-ray images", default=r"D:\leann\busxray_woodlands\exp")
    parser.add_argument("--annotation-dir", help="folder containing the annotation files", default=r"D:\leann\busxray_woodlands\annotations")

    args = parser.parse_args()
    compile_annotations(args.root_dir, args.annotation_dir)