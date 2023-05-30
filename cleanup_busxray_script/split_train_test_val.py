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
    images_val_dir = os.path.join(images_dir, 'val')
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_test_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)

    labels_train_dir = os.path.join(labels_dir, 'train')
    labels_test_dir = os.path.join(labels_dir, 'test')
    labels_val_dir = os.path.join(labels_dir, 'val')
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

    # Dict structure to reuse code
    ttv_dict = {"test": {"files": test_files, "image_dir": images_test_dir, "label_dir": labels_test_dir},
                "val": {"files": val_files, "image_dir": images_val_dir, "label_dir": labels_val_dir},
                "train": {"files": train_files, "image_dir": images_train_dir, "label_dir": labels_train_dir}
                }
    """
    Copy subdirs to respective directories
    """
    # Iterate through each folder
    for ttv_folder, ttv_data in ttv_dict.items():
        print(f"Transferring images to {ttv_folder.upper()} folder...")
        for subdir in tqdm(ttv_data["files"]):
            # Get list of image files
            image_files = [i for i in gs.load_files(subdir, file_type="images") if "cleaned" in i]
            segmented_subdir_path, segmented_subdir_name = gs.path_leaf(subdir)
            # Get rid of segmented at back of file name
            segmented_subdir_name = segmented_subdir_name[:-10]

            # Copy each image to target folder
            for file in image_files:
                # Clean up image name
                _, filename = gs.path_leaf(file)
                segmented_filename = f"{segmented_subdir_name}_{filename}"
                dst_path = os.path.join(ttv_data["image_dir"], segmented_filename)
                try:
                    src_image_path = os.path.join(subdir, file)
                    shutil.copy(src_image_path, dst_path)
                except:
                    print(f"{src_image_path} already exists")
                try:
                    src_annotation_path = gs.change_file_extension(src_image_path, '.txt')
                    dst_annotation_path = os.path.join(ttv_data["label_dir"], gs.change_file_extension(segmented_filename, '.txt'))
                    shutil.copy(src_annotation_path, dst_annotation_path)
                except:
                    print(f"{src_annotation_path} already exists")

    print('Data split completed successfully.')

if __name__ == "__main__":
    print("This script is not meant to be run directly.")
    print("Please import it as a module and call the split_data() function, unless you're debugging.")
    input_folder = r"annotations_adjusted"
    output_folder = rf"{input_folder}_output"
    # Usage example:
    split_data(input_folder, output_folder)

