"""
Just a mini testing file for me to try out python logic for debugging.
"""
import os
import general_scripts as gs
from tqdm import tqdm
import os, shutil

SEGMENTED_DIR = r"D:\BusXray\scanbus_training\Segmented files"
SCAN_BUS_DATASET_DIR_IMAGE = r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\images"
SCAN_BUS_DATASET_DIR_LABEL = r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\labels"

# identify all the files in the scanbus_dataset
dataset_files = gs.load_images(path_to_images=SCAN_BUS_DATASET_DIR_IMAGE, file_type=['jpg'], recursive=True)

# Gather segmented file name
segmented_file_name = []
for item in os.listdir(SEGMENTED_DIR):
    if os.path.isdir(item):
        start_index = item.find('_') + 1
        end_index = item.rfind('_') - 1
        name = item[start_index:end_index]
        segmented_file_name.append(name)