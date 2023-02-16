"""
A script that compares two images and returns their SSIM and MSE values, saved in an excel file.
in command line, specify --dir <full absolute path to folder that contains the images>
e.g. python ssim_script_modified.py --dir "D:\\BusXray\\Compiling_All_subfolder_images\\test_compiled_clean_images"

To compare clean images, use the 'compare_clean_images_ssim_mse_value()' function
To compare clean and threat images, use the 'compare_threat_images_with_clean_images()' function
If compare clean and threat images function is used, need to specify 2 directories. 1 contains the clean images, the other contains the threat images.
e.g. python ssim_script_modified.py --dir "D:\\BusXray\\Compiling_All_subfolder_images\\test_compiled_clean_images" --dir2 "D:\BusXray\Compiling_All_subfolder_images\Compiled_Threat_Images\removeThreat_images"

Note: It'll skip pass DualEnergy.tiff files because my primary objective was to compare the MonoChrome.tiff files
Made on: 16/2/2023
Last updated: 16/2/2023
By: Alphaeus
"""

from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import argparse
import general_scripts as gs
import os
import pandas as pd

# A constant to change depending on how many clean photos there are
NUM_OF_SCANS = 20


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

  # Get options
  args = options()
  info = {}
  # Import images and store as a list
  images = gs.load_images_from_folder(str(args.dir))
  images_with_visited_flag = [[i, False] for i in images]
  
  os.chdir(args.dir)
  bus_model_info = []
  for index, first_image in enumerate(images_with_visited_flag):
    
    # Check if reaches 20, can exit, because there's only 20 images per bus model
    counter = 0
    bus_model_1 = ' '.join(first_image[0].split(" ")[0:-3])
    bus_image_index_1 = (first_image[0].split(" ")[-2]).split("-")[-1]
    picture_type_1 = first_image[0].split(" ")[-1] # monochrome or dualenergy
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
        print(f"\nProcessing '{first_image[0]}' and '{second_image[0]}'...")
        temp_ssim, temp_mse = compare_ssim_mse_value(first_image[0], second_image[0])
        bus_model_info.append([bus_image_index_1, bus_image_index_2, temp_ssim, temp_mse])


    # Set flag as visited
    first_image[1] = True
    
    # If first_image index reaches NUM_OF_SCANS, it means it has finished going through all the various images. Update information into info dictionary, and wipe bus_model_info clean for the next model.
    if bus_image_index_1 == str(NUM_OF_SCANS):
      temp_info = {f'{bus_model_1}': bus_model_info}
      info.update(temp_info)
      bus_model_info = []
    
  # Organise information into lists
  print("@"*20)
  print("info:", info)
  writer = pd.ExcelWriter('SSIM_MSE_stats.xlsx', engine='xlsxwriter')
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
    sheet_name = (f'{bus_model}')[0:26]
    df_ssim.to_excel(writer, sheet_name=f'{sheet_name}_SSIM', index=False)
    sheet_name = (f'{bus_model}')[0:26]
    df_mse.to_excel(writer, sheet_name=f'{sheet_name}_MSE', index=False)
  writer.save()
      
def compare_threat_images_with_clean_images(): 

  # Get options
  args = options()
  info = {}
  num_of_images_per_bus_model = {}
  # Import images and store as a list
  clean_images = gs.load_images_from_folder(str(args.dir))
  threat_images = gs.load_images_from_folder(str(args.dir2))
  
  os.chdir(args.dir)
  bus_model_info = []
  for first_image in clean_images:
    
    # Keep track of how many images found for that specific bus model
    bus_model_count = 0
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
        temp_ssim, temp_mse = compare_ssim_mse_value(first_image, f"D:\\BusXray\\Compiling_All_subfolder_images\\Compiled_Threat_Images\\removeThreat_images\\{second_image}")
        bus_model_info.append([bus_image_index_1, bus_image_index_2, temp_ssim, temp_mse])
    
    # Exited from threat_image loop, save info and reset bus_model_info statistics and bus_model_count
      temp_info = {f'{bus_model_1}': bus_model_info}
      temp_info_bus_count = {f'{bus_model_1}': bus_model_count}
      info.update(temp_info)
      num_of_images_per_bus_model.update(temp_info_bus_count)
      bus_model_info = []
    
  # Organise information into lists
  print("@"*20)
  print("info:", info)
  writer = pd.ExcelWriter('Clean_vs_Threat_SSIM_MSE_stats.xlsx', engine='xlsxwriter')
  for bus_model, data in info.items():
    final_ssim_info = []
    final_mse_info = []
    num_of_bus_model_image = num_of_images_per_bus_model[bus_model]

    # Create an array-like structure and populate it with zeros
    for i in range(num_of_bus_model_image):
      final_ssim_info.append([0 for i in range(num_of_bus_model_image)])
      final_mse_info.append([0 for i in range(num_of_bus_model_image)])

    # Add in data into indexes with same row and column, i.e. 1 for SSIM, 0 for MSE.
    for i in range(num_of_bus_model_image):
      for j in range(num_of_bus_model_image):
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


    # df = pd.DataFrame({f'{pair_of_data[0]}': [i for i in ()'red', 'yellow', 'blue'], 'b': [0.5, 0.25, 0.125]})

    print("final_ssim_info:", final_ssim_info)
    print("final_mse_info:", final_mse_info)
    df_ssim = pd.DataFrame(final_ssim_info, columns=[str(i+1) for i in range(NUM_OF_SCANS)])
    df_mse = pd.DataFrame(final_mse_info, columns=[str(i+1) for i in range(NUM_OF_SCANS)])
    df_ssim.to_excel(writer, sheet_name=f'{bus_model}_{bus_image_index_1}_SSIM', index=False)
    df_mse.to_excel(writer, sheet_name=f'{bus_model}_{bus_image_index_1}_MSE', index=False)
  writer.save()

if __name__ == '__main__':
  # python ssim_script_modified.py --dir "D:\BusXray\Compiling_All_subfolder_images\Compiled_Clean_Images"
  compare_clean_images_ssim_mse_value()

  # python ssim_script_modified.py --dir "D:\BusXray\Compiling_All_subfolder_images\test_compiled_clean_images" --mode 1 --dir2 "D:\BusXray\Compiling_All_subfolder_images\Compiled_Threat_Images\removeThreat_images"
  # compare_threat_images_with_clean_images()