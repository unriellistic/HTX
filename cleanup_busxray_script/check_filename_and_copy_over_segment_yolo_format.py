"""
The script finds the corresponding file name in each train/test/val sub-folders in image/label folders, and copies over the
segmented version of it into the directory of the sub-folders in the form of:
    segmented_train
    segmented_test
    segmented_val
"""

from general_scripts import general_scripts as gs
from tqdm import tqdm
import os, shutil

SEGMENTED_DIR = r"D:\BusXray\scanbus_training\segmented files for monochrome images"
SCAN_BUS_DATASET_DIR_IMAGE = r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\images"
SCAN_BUS_DATASET_DIR_LABEL = r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\labels"

# identify all the files in the scanbus_dataset
dataset_files = gs.load_files(path_to_files=SCAN_BUS_DATASET_DIR_IMAGE, file_type=('jpg', '.tiff'), recursive=True)

# Gather segmented file name
segmented_file_name = [item[item.find('_')+1:item.rfind('_')] for item in os.listdir(SEGMENTED_DIR)]
for index, item in enumerate(segmented_file_name):
    if "Mono" in item:
        item = item[:-11]
    elif "temp_image_low" in item:
        item = item[:-14]
    segmented_file_name[index] = item
segmented_file_dir = [item for item in os.listdir(SEGMENTED_DIR)]
list_of_filenames_not_found = []

# Go through each file in dataset
for file in tqdm(dataset_files):
    basepath, filename = gs.path_leaf(file)
    # Get the unique number of image
    filename = gs.change_file_extension(filename, "")
    if "Dual" in filename:
        base_filename = filename[:-11]
    elif "final" in filename:
        base_filename = filename[:-11]

    # Check if the dataset file name exist in the segmented folder
    if base_filename not in segmented_file_name:
        print(f"{filename} not found")
        list_of_filenames_not_found.append(filename)
    else:
        if "Dual" in filename:
            # Get the path to all the segmented images in the segmented image folder. Excluding the annotations.
            temp_files = gs.load_files(path_to_files=os.path.join(SEGMENTED_DIR, f"adjusted_{base_filename} Monochrome_segmented"), file_type="images")
        elif "final" in filename:
            temp_files = gs.load_files(path_to_files=os.path.join(SEGMENTED_DIR, f"adjusted_{base_filename}temp_image_low_segmented"), file_type="images")
        clean_folder = False
        # Check if name of folder has clean in it, if it does, then perform copying for all images in folder
        if "clean" in base_filename:
            clean_folder = True
        # Get the folder names
        path_to_test_train_valid_image_folder, test_train_valid_image_folder = gs.path_leaf(basepath)
        # New folder for image
        new_image_folder = os.path.join(path_to_test_train_valid_image_folder, "segmented_"+test_train_valid_image_folder)
        # Get new label folder name
        new_label_folder = os.path.join(SCAN_BUS_DATASET_DIR_LABEL, "segmented_"+test_train_valid_image_folder)

        # Make new dir path for labels if doesn't exist
        if not os.path.exists(new_label_folder):
            os.makedirs(new_label_folder)
        # Make new dir path for images if doesn't exist            
        if not os.path.exists(new_image_folder):
            os.makedirs(new_image_folder)

        # Copy over the temp_files into a folder equivalent to it's basepath.
        for segmented_file in temp_files:
            # If clean in folder name, means their images won't have "clean" in the name, just perform for all images found.
            # OR check for "clean" in images and only extract those
            if clean_folder or "cleaned" in segmented_file:
                _, segmented_filename = gs.path_leaf(segmented_file)
                # src and dst path for image
                src_path = segmented_file
                dst_path = os.path.join(new_image_folder, f"{base_filename}_{segmented_filename}")
                shutil.copy(src_path, dst_path)
                # src and dst path for label if it exists
                src_path = os.path.join(gs.change_file_extension(segmented_file, ".txt"))
                if os.path.exists(src_path):
                    dst_path = os.path.join(new_label_folder, f"{base_filename}_{gs.change_file_extension(segmented_filename, '.txt')}")
                    shutil.copy(src_path, dst_path)
                # Else, create empty txt file
                else:
                    with open(os.path.join(new_label_folder, f"{base_filename}_{gs.change_file_extension(segmented_filename, '.txt')}"), 'x') as f:
                        pass

print("list_of_filenames_not_found:", list_of_filenames_not_found)
print("len(list_of_filenames_not_found):", len(list_of_filenames_not_found))