"""
A script that compares two images and returns their SSIM and MSE values, saved in an excel file.
in command line, specify --dir <full absolute path to folder that contains the images>
e.g. python ssim_script_modified.py --dir "D:\\BusXray\\Compiling_All_subfolder_images\\test_compiled_clean_images"

To compare clean images, use the 'compare_clean_images_ssim_mse_value()' function
To compare clean and threat images, use the 'compare_threat_images_with_clean_images()' function
If compare clean and threat images function is used, need to specify 2 directories. 1 contains the clean images, the other contains the threat images.
e.g. python ssim_script_modified.py --dir "D:\\BusXray\\Compiling_All_subfolder_images\\test_compiled_clean_images" --dir2 "D:\BusXray\Compiling_All_subfolder_images\Compiled_Threat_Images\removeThreat_images"

Note: 
- It'll skip pass DualEnergy.tiff files because my primary objective was to compare the MonoChrome.tiff files.
- It'll be able to calculate number of images or threat images in their respective folders. However, the files need to be saved in this format <bus model>-<ANYTHING>-<index>
Made on: 16/2/2023
Last updated: 17/2/2023
@Author: Alphaeus
"""

from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import argparse
import general_scripts as gs
import os
import pandas as pd
from datetime import datetime, timedelta
import time

# CONSTANTS to alter
DIR_TO_SAVE_IN = "D:\BusXray\Compiling_All_subfolder_images"
NUM_OF_SCANS = 3

current_datetime = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

def options():
  parser = argparse.ArgumentParser(description="Read image metadata")
  parser.add_argument("-d", "--dir", help="Input clean image file folder.", required=True)
  parser.add_argument("-m", "--mode", help="Sets the mode for functions to use, 0: compare clean images | 1: compare clean and threat images", default=0)
  parser.add_argument("-i", "--dir2", help="Input threat image file folder", default="")
  args = parser.parse_args()
  return args

def mse(imageA, imageB):
  # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
  mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
  mse_error /= float(imageA.shape[0] * imageA.shape[1])

  # return the MSE. The lower the error, the more "similar" the two images are.
  return mse_error

def compare(imageA, imageB):
  # Calculate the MSE and SSIM
  m = mse(imageA, imageB)
  s = ssim(imageA, imageB)

  # Return the SSIM. The higher the value, the more "similar" the two images are.
  return s

def compare_ssim_mse_value(image1, image2):
  image1 = cv2.imread(image1)
  image2 = cv2.imread(image2)

  # Convert the images to grayscale
  gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

  # Check for same size and ratio and report accordingly
  ho, wo, _ = image1.shape
  hc, wc, _ = image2.shape
  ratio_orig = ho/wo
  ratio_comp = hc/wc
  dim = (wc, hc)

  if round(ratio_orig, 2) != round(ratio_comp, 2):
    print("\nImages not of the same dimension. Check input.")
    exit()

  # Resize first image if the second image is smaller
  elif ho > hc and wo > wc:
    print("\nResizing original image for analysis...")
    gray1 = cv2.resize(gray1, dim)

  elif ho < hc and wo < wc:
    print("\nCompressed image has a larger dimension than the original. Check input.")
    exit()

  if round(ratio_orig, 2) == round(ratio_comp, 2):
    mse_value = mse(gray1, gray2)
    ssim_value = compare(gray1, gray2)
    print("MSE:", mse_value)
    print("SSIM:", ssim_value)
    return ssim_value, mse_value

# Compares images in the directory specified and returns their SSIM, MSE values and saves it in an excel file in the same directory.
def compare_clean_images_ssim_mse_value(): 

  # Timer
  start = time.time()
  # Get options
  args = options()
  info = {}
  num_of_images_per_bus_model = {}
  # Import images and store as a list
  images = gs.load_images_from_folder(str(args.dir))
  images_with_visited_flag = [[i, False] for i in images]
  number_of_images_left = len(images)
  
  os.chdir(args.dir)
  bus_model_info = []
  for first_image_index, first_image in enumerate(images_with_visited_flag):
    
    bus_model_count = 0
    # Check if reaches 20, can exit, because there's only 20 images per bus model
    counter = 0
    bus_model_1 = ' '.join(first_image[0].split(" ")[0:-3])
    bus_image_index_1 = (first_image[0].split(" ")[-2]).split("-")[-1]
    picture_type_1 = first_image[0].split(" ")[-1] # monochrome or dualenergy

    # Keep track of smallest index for that specific bus model
    smallest_bus_model_index = 1000000000000

    if picture_type_1 == "DualEnergy.tiff":
      continue #skip, we only want to test the monochrome images

    for second_image in images_with_visited_flag:
      if counter == NUM_OF_SCANS:
        break
      if second_image[1] == True:
        continue
      # value will be 1, no point comparing
      if first_image[0] == second_image[0]:
        continue
      
      bus_model_2 = ' '.join(second_image[0].split(" ")[0:-3])
      picture_type_2 = second_image[0].split(" ")[-1]
      bus_image_index_2 = (second_image[0].split(" ")[-2]).split("-")[-1]
      if picture_type_2 == "DualEnergy.tiff":
        continue #skip, we only want to test the monochrome images

      # Check if they're the same model, if they are, compare them
      if bus_model_1 == bus_model_2:
        counter += 1
        bus_model_count += 1
        print(f"\nProcessing '{first_image[0]}' and '{second_image[0]}'...")
        temp_ssim, temp_mse = compare_ssim_mse_value(first_image[0], second_image[0])
        bus_model_info.append([bus_image_index_1, bus_image_index_2, temp_ssim, temp_mse])

        # Update smallest_bus_model_index
        if smallest_bus_model_index > int(bus_image_index_2):
          smallest_bus_model_index = int(bus_image_index_2)

    # This will be repeated NUM_OF_SCANS's times, will override the same values.
    temp_info_bus_count = {f'{bus_model_1}': [bus_model_count, smallest_bus_model_index]}
    num_of_images_per_bus_model.update(temp_info_bus_count)

    # Set flag as visited
    first_image[1] = True
    print(f"Number of scans left: {number_of_images_left-1} out of {len(images)}")
    number_of_images_left -= 1
    
    # If first_image index reaches NUM_OF_SCANS, it means it has finished going through all the various images. Update information into info dictionary, and wipe bus_model_info clean for the next model.
    if (first_image_index+1) % NUM_OF_SCANS == 0:
      temp_info = {f'{bus_model_1}': bus_model_info}
      info.update(temp_info)
      bus_model_info = []
    
  # Organise information into lists
  print("#"*100)
  print("info:", info)
  writer = pd.ExcelWriter(f'{DIR_TO_SAVE_IN}\Clean_vs_Clean_SSIM_MSE_stats_{current_datetime}.xlsx', engine='xlsxwriter')
  for bus_model, data in info.items():
    final_ssim_info = []
    final_mse_info = []

    # Create an array-like structure and populate it with zeros
    for i in range(NUM_OF_SCANS):
      final_ssim_info.append([0 for i in range(NUM_OF_SCANS)])
      final_mse_info.append([0 for i in range(NUM_OF_SCANS)])

    # Add in data into indexes with same row and column, i.e. 1 for SSIM, 0 for MSE.
    for i in range(NUM_OF_SCANS):
      for j in range(NUM_OF_SCANS):
        if i == j:
          final_ssim_info[i][j] = 1
          final_mse_info[i][j] = 0

    # pair_of_data = ['2', '3', 0.649719720844514, 1021.8158662027311], index, index, SSIM, MSE values
    for pair_of_data in data:
      # Populate SSIM information
      final_ssim_info[int(pair_of_data[0])-1][int(pair_of_data[1])-1] = pair_of_data[2]
      final_ssim_info[int(pair_of_data[1])-1][int(pair_of_data[0])-1] = pair_of_data[2] # Populate the reversed side
      # Populate MSE information
      final_mse_info[int(pair_of_data[0])-1][int(pair_of_data[1])-1] = pair_of_data[3]
      final_mse_info[int(pair_of_data[1])-1][int(pair_of_data[0])-1] = pair_of_data[3] # Populate the reversed side

    print("final_ssim_info:", final_ssim_info)
    print("final_mse_info:", final_mse_info)
    df_ssim = pd.DataFrame(final_ssim_info, columns=[str(i+1) for i in range(NUM_OF_SCANS)])
    df_mse = pd.DataFrame(final_mse_info, columns=[str(i+1) for i in range(NUM_OF_SCANS)])

    # Excel sheet can't have > 31 characters, shorten it
    sheet_name = (f'{bus_model}')[0:26]
    df_ssim.to_excel(writer, sheet_name=f'{sheet_name}_SSIM', index=False)
    sheet_name = (f'{bus_model}')[0:26]
    df_mse.to_excel(writer, sheet_name=f'{sheet_name}_MSE', index=False)
    current = time.time()
    print(f"Total time taken thus far: {str(timedelta(seconds=(current-start)))}")
  writer.save()
  end = time.time()
  print(f"Total time taken: {str(timedelta(seconds=(end-start)))}")
      
def compare_threat_images_with_clean_images(): 
  start = time.time()
  # Log info to see which files did not complete successfully
  log = []
  # Get options
  args = options()
  info = {}
  num_of_images_per_bus_model = {}
  # Import images and store as a list
  clean_images = gs.load_images_from_folder(str(args.dir))
  threat_images = gs.load_images_from_folder(str(args.dir2))
  number_of_images_left = len(clean_images)
  
  os.chdir(args.dir)
  bus_model_info = []
  for first_image_index, first_image in enumerate(clean_images):
    
    # Keep track of how many images found for that specific bus model
    bus_model_count = 0
    # Keep track of smallest index for that specific bus model
    smallest_bus_model_index = 1000000000000
    # Extract clean image details
    bus_model_1 = ' '.join(first_image.split(" ")[0:-3])
    bus_image_index_1 = (first_image.split(" ")[-2]).split("-")[-1]
    picture_type_1 = first_image.split(" ")[-1] # monochrome or dualenergy
    if picture_type_1 == "DualEnergy.tiff":
      continue #skip, we only want to test the monochrome images

    for index_of_threat_images, second_image in enumerate(threat_images):
      
      # Extract threat image details
      bus_model_2 = ' '.join(second_image.split(" ")[0:-2])
      picture_type_2 = (second_image.split(" ")[-1]).split("-")[-1]
      bus_image_index_2 = (second_image.split(" ")[-1]).split("-")[-2]
      if picture_type_2 == "final_color.jpg":
        continue #skip, we only want to test the monochrome images

      # Check if they're the same model, if they are, compare them
      if bus_model_1 == bus_model_2:
        bus_model_count += 1
        print(f"\nProcessing '{first_image}' and '{second_image}'...")
        temp_ssim, temp_mse = compare_ssim_mse_value(first_image, f"{args.dir2}\{second_image}")
        bus_model_info.append([bus_image_index_1, bus_image_index_2, temp_ssim, temp_mse])

        # Update smallest_bus_model_index
        if smallest_bus_model_index > int(bus_image_index_2):
          smallest_bus_model_index = int(bus_image_index_2)

    # This will be repeated NUM_OF_SCANS's times, will override the same values.
    temp_info_bus_count = {f'{bus_model_1}': [bus_model_count, smallest_bus_model_index]}
    num_of_images_per_bus_model.update(temp_info_bus_count)

    print("num_of_images_per_bus_model:", num_of_images_per_bus_model)
    print("first_image_index:", first_image_index)
    print("num_of_images_per_bus_model[bus_model_1]:", num_of_images_per_bus_model[bus_model_1])
    
    # check that for every 3 images checked, it'll save and reset. Have to manually change NUM_OF_SCANS constant at the top of the code to reflect how many clean images you're using
    if (first_image_index+1) % NUM_OF_SCANS == 0:
      temp_info = {f'{bus_model_1}': bus_model_info}
      info.update(temp_info)
      bus_model_info = []
      current = time.time()
      print(f"Total time taken thus far: {str(timedelta(seconds=(current-start)))}")
    
    print(f"Number of scans left: {number_of_images_left-1} out of {len(clean_images)}")
    number_of_images_left -= 1
    
    
  # Organise information into lists
  print("")
  print("-"*150)
  print("")
  print("info:", info)

  # Check how much time has passed
  current = time.time()
  print(f"Total time taken thus far: {str(timedelta(seconds=(current-start)))}")

  # Change directory to the directory you want it to be saved in
  writer = pd.ExcelWriter(f'{DIR_TO_SAVE_IN}\Clean_vs_Threat_SSIM_MSE_stats_{current_datetime}.xlsx', engine='xlsxwriter')
  for bus_model, data in info.items():

    # Check if model has any data recorded, if nil, then skip to next model.
    if len(data) == 0:
      print(f"{bus_model} doesn't have a copy of clean and threat images to compare")
      log.append(bus_model)
      continue

    num_of_bus_model_image = num_of_images_per_bus_model[bus_model][0] # I can do this because i saved the same bus_model name in the num_of_images_per_bus_model as well. 0 specifies bus_model_count, 1 specifies smallest_bus_model_index

    final_ssim_info = []
    final_mse_info = []
    # Create an array-like structure and populate it with zeros, hardcoded with NUM_OF_SCANS
    for i in range(NUM_OF_SCANS):
      
      # +1 because include one more column for the index later
      final_ssim_info.append([0 for i in range(num_of_bus_model_image+1)])
      final_mse_info.append([0 for i in range(num_of_bus_model_image+1)])

    print("final_ssim_info:", final_ssim_info)
    print("final_mse_info:", final_mse_info)
    
    # pair_of_data = ['2', '3', 0.649719720844514, 1021.8158662027311], index of clean image, index of, SSIM, MSE values
    print("data:", data)

    # Start at 1 because we want to skip the first column. For index.
    lame_method_counter = 1
    i = 0
    for pair_of_data in data:
      # Will store information according to the order they're packaged in data. Which may not be numerical ordered correctly.
      # Can do post-processing in excel to arrange them correctly.

      # Debugging purposes
      # print("current bus model:", bus_model)
      # print("num_of_bus_model_image:", num_of_bus_model_image)
      # print("pair_of_data[1]:", pair_of_data[1])
      # print("index:", index)
      #       
      # Populate SSIM information
      final_ssim_info[i][lame_method_counter] = pair_of_data[2]
      # Populate MSE information
      final_mse_info[i][lame_method_counter] = pair_of_data[3]

      # Means we should move on to the next row
      if lame_method_counter == num_of_bus_model_image:
        # Before we go next row, populate SSIM information for first column
        final_ssim_info[i][0] = pair_of_data[0]
        # Before we go next row, populate MSE information for first column
        final_mse_info[i][0] = pair_of_data[0]
        
        # Go next row
        i += 1
        lame_method_counter = 0
      
      # To check how many times a column has been updated. Lame because I do not wanna put more effort to think of something smarter.
      lame_method_counter += 1
        

    # Populate first column with the corresponding clean image that was tested against the threat images
    # data is a list of lists
    # e.g. [['2', '3', 0.649719720844514, 1021.8158662027311], ['2', '4', 0.6719720844514, 1011.8158662027311], ...]
    print("data:", data)

    # df = pd.DataFrame({f'{pair_of_data[0]}': [i for i in ()'red', 'yellow', 'blue'], 'b': [0.5, 0.25, 0.125]})

    print("final_ssim_info:", final_ssim_info)
    print("final_mse_info:", final_mse_info)
    temp_column_header = ['Clean Image indexes vs Threat Image indexes']

    # Get list of all the threat images found
    list_of_threat_image_indexes = [str(pair_of_data[1]) for pair_of_data in data]
    print("list_of_threat_image_indexes:", list_of_threat_image_indexes)
    # Remove duplicates
    temp_column__header_2 = []
    [temp_column__header_2.append(x) for x in list_of_threat_image_indexes if x not in temp_column__header_2]
    joined_column_header = temp_column_header + temp_column__header_2
    print("joined_column_header:", joined_column_header)
    # # rearrange the array value
    final_ssim_info = np.array(final_ssim_info)
    final_ssim_info = np.resize(final_ssim_info, (NUM_OF_SCANS, len(joined_column_header)))
    final_mse_info = np.array(final_mse_info)
    final_mse_info = np.resize(final_mse_info, (NUM_OF_SCANS, len(joined_column_header)))
    print("final_ssim_info:", final_ssim_info)
    print("final_mse_info:", final_mse_info)

    # Save it as df
    df_ssim = pd.DataFrame(final_ssim_info, columns=joined_column_header)
    df_mse = pd.DataFrame(final_mse_info, columns=joined_column_header)

    # Excel sheet can't have > 31 characters, shorten it
    sheet_name = (f'{bus_model}')[0:26]
    df_ssim.to_excel(writer, sheet_name=f'{sheet_name}_SSIM', index=False)
    sheet_name = (f'{bus_model}')[0:26]
    df_mse.to_excel(writer, sheet_name=f'{sheet_name}_MSE', index=False)
    current = time.time()
    print(f"Total time taken thus far: {str(timedelta(seconds=(current-start)))}")
  writer.save()
  print("These files, bus models, did not have a copy of Clean vs Threat:", log)
  end = time.time()
  print(f"Total time taken: {str(timedelta(seconds=(end-start)))}")

if __name__ == '__main__':
  # python ssim_script_modified.py --dir "D:\BusXray\Compiling_All_subfolder_images\Compiled_Clean_Images"
  # compare_clean_images_ssim_mse_value()

  # python ssim_script_modified.py --dir "D:\BusXray\Compiling_All_subfolder_images\test_compiled_clean_images" --mode 1 --dir2 "D:\BusXray\Compiling_All_subfolder_images\Compiled_Threat_Images\removeThreat_images"
  # python ssim_script_modified.py --dir "D:\BusXray\Compiling_All_subfolder_images\test_compiled_clean_images" --mode 1 --dir2 "D:\BusXray\Compiling_All_subfolder_images\Compiled_Threat_Images\images_for_ssim_test"
  compare_threat_images_with_clean_images()