import os
import random
import shutil
import general_scripts as gs
from tqdm import tqdm

input_folder = r"E:\alp\segmented_master_file_for_both_clean_and_threat_images_dualenergy"
output_folder = r"E:\alp\output_dualenergy"

minput_folder = r"E:\alp\segmented_master_file_for_both_clean_and_threat_images_monochrome"
moutput_folder = r"E:\alp\output_monochrome"


def split_data(input_folder, output_folder, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, seed=42):
    random.seed(seed)

    # Create output directories
    images_dir = os.path.join(output_folder, 'images')
    labels_dir = os.path.join(output_folder, 'labels') 
    images_train_dir = os.path.join(images_dir, 'train')
    images_test_dir = os.path.join(images_dir, 'test')
    images_val_dir = os.path.join(images_dir, 'validation')
    labels_train_dir = os.path.join(labels_dir, 'train')
    labels_test_dir = os.path.join(labels_dir, 'test')
    labels_val_dir = os.path.join(labels_dir, 'validation')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_test_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_test_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    # Get list of image files
    image_files = [file for file in os.listdir(input_folder) if file.endswith(('.tif', '.tiff', '.jpg'))]

    # Shuffle the list
    random.shuffle(image_files)

    # Calculate number of samples for each split
    num_samples = len(image_files)
    num_train = int(num_samples * train_ratio)
    num_test = int(num_samples * test_ratio)

    # Split the data
    train_files = image_files[:num_train]
    test_files = image_files[num_train:num_train+num_test]
    val_files = image_files[num_train+num_test:]

    # Move files to respective directories
    for file in tqdm(train_files):
        try:
            image_path = os.path.join(input_folder, file)
            shutil.move(image_path, images_train_dir)
        except:
            print(f"{image_path} already exists")
        try:
            annotation_path = os.path.join(input_folder, gs.change_file_extension(image_path, '.txt'))
            shutil.move(annotation_path, labels_train_dir)
        except:
            print(f"{annotation_path} already exists")

    for file in tqdm(test_files):
        try:
            image_path = os.path.join(input_folder, file)
            shutil.move(image_path, images_test_dir)
        except:
            print(f"{image_path} already exists")
        try:
            annotation_path = os.path.join(input_folder, gs.change_file_extension(image_path, '.txt'))
            shutil.move(annotation_path, labels_test_dir)
        except:
            print(f"{annotation_path} already exists")

    for file in tqdm(val_files):
        try:
            image_path = os.path.join(input_folder, file)
            shutil.move(image_path, images_val_dir)
        except:
            print(f"{image_path} already exists")
        try:
            annotation_path = os.path.join(input_folder, gs.change_file_extension(image_path, '.txt'))
            shutil.move(annotation_path, labels_val_dir)   
        except:
            print(f"{annotation_path} already exists")

    print('Data split completed successfully.')

# Usage example:
split_data(input_folder, output_folder)
split_data(minput_folder, moutput_folder)

