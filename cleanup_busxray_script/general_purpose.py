"""
This script doesn't contain any constant info, just for me to run a process to clean up a customised older
"""

import os, shutil, re
from tqdm import tqdm
import general_scripts as gs

images = gs.load_files(r"D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images_monochrome", ".xml")
image_file = gs.load_files(r"D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images_monochrome")
def extract_unique_number(filename):
    pattern = r'-(\d+)-'
    matches = re.findall(pattern, filename)
    if len(matches) > 1:
        print('Warning: multiple matches found for pattern, selecting the first one found')
    if matches:
        return matches[0]
    else:
        return None
    
new_dir = r"D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images_dualenergy"
os.chdir(r"D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images_monochrome")
for image in images:
    filepath, filename = gs.path_leaf(image)
    unique_id = extract_unique_number(image)
    for real_image in image_file:
        _, imagename = gs.path_leaf(real_image)
        if unique_id == extract_unique_number(real_image):
            os.rename(filename, gs.change_file_extension(imagename, ".xml"))
        