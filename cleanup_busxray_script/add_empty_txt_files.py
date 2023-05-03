import os
from tqdm import tqdm
import general_scripts as gs

"""
To create empty txt files for clean images
"""

clean_images_in_segmented_test = gs.load_images(r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\images\segmented_mono_test")
clean_images_in_segmented_train = gs.load_images(r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\images\segmented_mono_train")
clean_images_in_segmented_val = gs.load_images(r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\images\segmented_mono_val")
corresponding_label_dir_test = r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\labels\segmented_mono_test"
corresponding_label_dir_train = r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\labels\segmented_mono_train"
corresponding_label_dir_val = r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\labels\segmented_mono_val"

for image in clean_images_in_segmented_test:
    if "-clean-" in image:
        _, filename = gs.path_leaf(image)
        # Write the XML string to a file
        with open(os.path.join(corresponding_label_dir_test, gs.change_file_extension(filename, ".txt")), 'x') as f:
            pass

for image in clean_images_in_segmented_train:
    if "-clean-" in image:
        _, filename = gs.path_leaf(image)
        # Write the XML string to a file
        with open(os.path.join(corresponding_label_dir_train, gs.change_file_extension(filename, ".txt")), 'x') as f:
            pass
     
for image in clean_images_in_segmented_val:
    if "-clean-" in image:
        _, filename = gs.path_leaf(image)
        # Write the XML string to a file
        with open(os.path.join(corresponding_label_dir_val, gs.change_file_extension(filename, ".txt")), 'x') as f:
            pass