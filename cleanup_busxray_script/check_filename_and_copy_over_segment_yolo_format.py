import general_scripts as gs
from tqdm import tqdm
import os, shutil

SEGMENTED_DIR = r"D:\BusXray\scanbus_training\Segmented files"
SCAN_BUS_DATASET_DIR_IMAGE = r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\images"
SCAN_BUS_DATASET_DIR_LABEL = r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\labels"

# identify all the files in the scanbus_dataset
dataset_files = gs.load_images(path_to_images=SCAN_BUS_DATASET_DIR_IMAGE, file_type=['jpg', 'tiff'], recursive=True)

# Gather segmented file name
segmented_file_name = [item[item.find('_')+1:item.rfind('_')-1] for item in os.listdir(SEGMENTED_DIR) if os.path.isdir(item)]
segmented_file_dir = [item for item in os.listdir(SEGMENTED_DIR) if os.path.isdir(item)]

# Go through each file in dataset
for file in tqdm(dataset_files):
    basepath, filename = gs.path_leaf(file)
    # Get the root filename without the .tiff or .jpg
    base_filename = gs.change_file_extension(filename, "")
    # Check if it exists
    if base_filename not in segmented_file_name:
        print(f"{base_filename} not found")
    else:
        # Get the path to all the segments in the segmented image folder
        temp_files = gs.load_images(path_to_images=os.join(SEGMENTED_DIR, base_filename))

        # Get the folder names
        path_to_test_train_valid_image_folder, test_train_valid_image_folder = gs.path_leaf(basepath)
        # New folder for image
        new_folder = os.path.join(path_to_test_train_valid_image_folder, "segmented_"+test_train_valid_image_folder)

        # Get label folder names
        new_label_folder = os.path.join(SCAN_BUS_DATASET_DIR_LABEL, test_train_valid_image_folder)

        # Make new dir path for labels if doesn't exist
        if not os.path.exists(new_label_folder):
            os.makedirs(new_folder)
        # Make new dir path for images if doesn't exist            
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        # Copy over the temp_files into a folder equivalent to it's basepath.
        for segmented_file in temp_files:
            # Get the cleaned image
            if "cleaned" in segmented_file and [".jpg", ".tiff"] in segmented_file:
                # src and dst path for image
                src_path = os.path.join(SEGMENTED_DIR, f"adjusted_{segmented_file_name}_segmented", segmented_file)
                dst_path = os.path.join(new_folder, f"{segmented_file_name}_{segmented_file}")
                shutil.copy(src_path, dst_path)
                # src and dst path for label if it exists
                src_path = os.path.join(SEGMENTED_DIR, f"adjusted_{segmented_file_name}_segmented", gs.change_file_extension(segmented_file, ".txt"))
                if os.path.exists(src_path):
                    dst_path = os.path.join(new_label_folder, f"{segmented_file_name}_{gs.change_file_extension(segmented_file, '.txt')}")
                    shutil.copy(src_path, dst_path)




