"""
The script consolidates all the images and txt files from the adjusted folder.
"""
import os, shutil
import general_scripts as gs
from tqdm import tqdm

ROOT_DIR = r"D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_monochrome"
TARGET_DIR = r"D:\BusXray\scanbus_training\segmented_master_file_for_both_clean_and_threat_images_monochrome"

# Create the output directory if it does not exist
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

files = [file for file in gs.load_images(ROOT_DIR, recursive=True) if "cleaned" in file]

# Copy over the temp_files into a folder equivalent to it's basepath.
for segmented_file in tqdm(files):

    segmented_filepath, segmented_filename = gs.path_leaf(segmented_file)
    _, segmented_filepath_name = gs.path_leaf(segmented_filepath)

    # Get rid of adjusted and segmented at front and back of file name
    segmented_filepath_name = segmented_filepath_name[9:-10]

    # src and dst path for image
    src_path = segmented_file
    dst_path = os.path.join(TARGET_DIR, f"{segmented_filepath_name}_{segmented_filename}")
    shutil.copy(src_path, dst_path)
    # src and dst path for label
    src_path = os.path.join(gs.change_file_extension(segmented_file, ".txt"))
    if os.path.exists(src_path):
        dst_path = os.path.join(TARGET_DIR, f"{segmented_filepath_name}_{gs.change_file_extension(segmented_filename, '.txt')}")
        shutil.copy(src_path, dst_path)
    # Else, create empty txt file
    else:
        print("creating empty txt file for:", f"{segmented_filepath_name}_{segmented_filename}")
        with open(os.path.join(TARGET_DIR, f"{segmented_filepath_name}_{gs.change_file_extension(segmented_filename, '.txt')}"), 'x') as f:
            pass
