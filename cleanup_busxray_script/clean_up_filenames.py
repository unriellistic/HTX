"""
The script converts the images in "D:\BusXray\scanbus_training\Compiled_Clean_Images" folder to the running order specified in the 
"D:\BusXray\scanbus_training\Compiled_Threat_Images\removeThreat_images"
"""

import os, re, shutil
import general_scripts as gs
from tqdm import tqdm

NUMBER_OF_IMAGES_IN_ORIGINAL = 609
NEW_DIR = r"D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images"
def extract_unique_number(filename):
    pattern = r'-(\d+)-'
    matches = re.findall(pattern, filename)
    if len(matches) > 1:
        print('Warning: multiple matches found for pattern, selecting the first one found')
    if matches:
        return matches[0]
    else:
        return None

clean_images = gs.load_images(r"D:\BusXray\scanbus_training\Compiled_Clean_Images")
# Start from 610 since there's a 609 copy
NUMBER_OF_IMAGES_IN_ORIGINAL += 1
for image in tqdm(clean_images):
    filepath, filename = gs.path_leaf(image)
    if "Dual" in filename:
        new_filename = filename.split("-")[0] + '-' + filename.split("-")[1] + '-' + str(NUMBER_OF_IMAGES_IN_ORIGINAL) + "-" + filename.split("-")[-1]
        shutil.copy(image, os.path.join(NEW_DIR, new_filename))
    elif "Monochrome" in filename:
        new_filename = filename.split("-")[0] + '-' + filename.split("-")[1] + '-' + str(NUMBER_OF_IMAGES_IN_ORIGINAL) + "-" + filename.split("-")[-1]
        shutil.copy(image, os.path.join(NEW_DIR, new_filename))
        NUMBER_OF_IMAGES_IN_ORIGINAL += 1