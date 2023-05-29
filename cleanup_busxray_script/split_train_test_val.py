"""
Script splits files from from target directory to output folder in the form of:
    images
    |--->
        train
        test
        validation
    labels
    |--->
        train
        test
        validation
"""

import os
import random
import shutil
import general_scripts as gs
from tqdm import tqdm

def split_data(input_folder, output_folder, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, seed=42):
    random.seed(seed)

    # Create output directories
    images_dir = os.path.join(output_folder, 'images')
    labels_dir = os.path.join(output_folder, 'labels') 
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    images_train_dir = os.path.join(images_dir, 'train')
    images_test_dir = os.path.join(images_dir, 'test')
    images_val_dir = os.path.join(images_dir, 'validation')
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_test_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)

    labels_train_dir = os.path.join(labels_dir, 'train')
    labels_test_dir = os.path.join(labels_dir, 'test')
    labels_val_dir = os.path.join(labels_dir, 'validation')
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_test_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    # Get a list of subdirs
    subdirectories = []
    for item in os.listdir(input_folder):
        item_path = os.path.join(input_folder, item)
        if os.path.isdir(item_path):
            subdirectories.append(item_path)

    # Shuffle the subdir list
    random.shuffle(subdirectories)

    # Calculate number of samples for each split
    num_samples = len(subdirectories)
    num_train = int(num_samples * train_ratio)
    num_test = int(num_samples * test_ratio)

    # Split the data
    train_files = subdirectories[:num_train]
    test_files = subdirectories[num_train:num_train + num_test]
    val_files = subdirectories[num_train + num_test:]

    """
    Copy subdirs to respective directories
    """
    # Copy for train folder
    for subdir in tqdm(train_files):
        # Get list of image files
        image_files = gs.load_files(subdir, file_type="images")
        segmented_subdir_path, segmented_subdir_name = gs.path_leaf(subdir)
        # Get rid of segmented at back of file name
        segmented_subdir_name = segmented_subdir_name[:-10]

        # Copy each image to target folder
        for file in image_files:
            # Clean up image name
            segmented_filename = f"{segmented_subdir_name}_{file}"
            dst_path = f"{segmented_subdir_path}_{segmented_filename}"
            try:
                src_image_path = os.path.join(subdir, file)
                shutil.copy(src_image_path, os.path.join(images_train_dir, dst_path))
            except:
                print(f"{src_image_path} already exists")
            try:
                annotation_path = gs.change_file_extension(src_image_path, '.txt')
                shutil.copy(annotation_path, os.path.join(labels_train_dir, dst_path))
            except:
                print(f"{annotation_path} already exists")

    # Copy for test folder
    for subdir in tqdm(test_files):
        # Get list of image files
        image_files = gs.load_files(subdir, file_type="images")
        segmented_subdir_path, segmented_subdir_name = gs.path_leaf(subdir)
        # Get rid of segmented at back of file name
        segmented_subdir_name = segmented_subdir_name[:-10]

        # Copy each image to target folder
        for file in image_files:
            # Clean up image name
            segmented_filename = f"{segmented_subdir_name}_{file}"
            dst_path = f"{segmented_subdir_path}_{segmented_filename}"
            try:
                src_image_path = os.path.join(subdir, file)
                shutil.copy(src_image_path, os.path.join(images_test_dir, dst_path))
            except:
                print(f"{src_image_path} already exists")
            try:
                annotation_path = gs.change_file_extension(src_image_path, '.txt')
                shutil.copy(annotation_path, os.path.join(labels_test_dir, dst_path))
            except:
                print(f"{annotation_path} already exists")
    # Copy for val folder
    for subdir in tqdm(val_files):
        # Get list of image files
        image_files = gs.load_files(subdir, file_type="images")
        segmented_subdir_path, segmented_subdir_name = gs.path_leaf(subdir)
        # Get rid of segmented at back of file name
        segmented_subdir_name = segmented_subdir_name[:-10]

        # Copy each image to target folder
        for file in image_files:
            # Clean up image name
            segmented_filename = f"{segmented_subdir_name}_{file}"
            dst_path = f"{segmented_subdir_path}_{segmented_filename}"
            try:
                src_image_path = os.path.join(subdir, file)
                shutil.copy(src_image_path, os.path.join(images_val_dir, dst_path))
            except:
                print(f"{src_image_path} already exists")
            try:
                annotation_path = gs.change_file_extension(src_image_path, '.txt')
                shutil.copy(annotation_path, os.path.join(labels_val_dir, dst_path))
            except:
                print(f"{annotation_path} already exists")

    print('Data split completed successfully.')

if __name__ == "__main__":
    print("This script is not meant to be run directly.")
    print("Please import it as a module and call the split_data() function, unless you're debugging.")
    input_folder = r"E:\alp\segmented_master_file_for_both_clean_and_threat_images_dualenergy"
    output_folder = r"E:\alp\output_dualenergy"
    # Usage example:
    split_data(input_folder, output_folder)

