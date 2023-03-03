"""
A script to perform histogram equaliser on images.
CLEAN_IMR_DIR is the path to the folder that stores the clean images
FOLDER_FOR_HISTOGRAMISED_IMG is the path to the folder that stores the histogramised image
THREAT_IMG_DIR is the path to the folder that stores the threat image
DIR_FOR_IMAGE_INPUT contains the either the path to the clean image folder, or can set to the path to the threat image folder
"""
import cv2
import matplotlib.pyplot as plt
import general_scripts as gs
import os, numpy as np

# python ssim_script_modified.py --dir "D:\BusXray\Compiling_All_subfolder_images\test_compiled_clean_images" --dir2 "D:\BusXray\Compiling_All_subfolder_images\Compiled_Threat_Images\removeThreat_images"
CLEAN_IMR_DIR = r"D:\BusXray\Compiling_All_subfolder_images\compiled_clean_images"
FOLDER_FOR_HISTOGRAMISED_IMG = r"D:\BusXray\Compiling_All_subfolder_images\test_folder_for_threat_histogramised"
THREAT_IMG_DIR = r"D:\BusXray\Compiling_All_subfolder_images\Compiled_Threat_Images\removeThreat_images"
DIR_FOR_IMAGE_INPUT = THREAT_IMG_DIR
BUS_MODEL = 'PC603E Volvo'
TYPE_OF_SCAN = 'temp_image_low' # or 'Monochrome' for clean image

def equalize_histogram_16bit(src_img):
    # Create a copy of the input image
    dst_img = src_img.copy()

    # Compute histogram
    hist_size = 2**16
    hist = np.zeros(hist_size, dtype=int)
    for pixel_val in src_img.flat:
        hist[pixel_val] += 1

    # Check if image is already equalized
    i = 0
    while hist[i] == 0:
        i += 1
    total_pixels = src_img.size
    if hist[i] == total_pixels:
        dst_img[:, :] = i
        return dst_img

    # Compute lookup table
    scale = float(hist_size - 1) / (total_pixels - hist[i])
    lut = np.zeros(hist_size, dtype=np.float32)
    lut[i:] = (np.cumsum(hist[i:]) - hist[i]/2.0) * scale
    lut = np.round(lut)
    lut = np.clip(lut, 0, hist_size-1).astype(np.uint16)

    # Apply lookup table to input image
    dst_img = lut[src_img]

    return dst_img

import cv2
import numpy as np

# def equalize_histogram_16bit_CLAHE(image):
#     # Convert the input image to grayscale if it is not already in grayscale
#     if len(image.shape) == 3 and image.shape[2] == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Create a CLAHE object
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

#     # Apply CLAHE to the input image
#     equalized = clahe.apply(image)

#     return equalized


list_of_clean_images = gs.load_images_from_folder(DIR_FOR_IMAGE_INPUT)
one_set_of_bus_model = [i for i in list_of_clean_images if BUS_MODEL in i and TYPE_OF_SCAN in i]
print(one_set_of_bus_model)

for image in one_set_of_bus_model:
  img = cv2.imread(os.path.join(DIR_FOR_IMAGE_INPUT, image), cv2.IMREAD_ANYDEPTH)
  # Find max intensity value of pixel in image
  biggest_pixel_value = np.amax(img)

  # Normalisation. Tried on 24/2/23, doesn't seem to do anything. Compared individual pixels but like never normalise.
  # img_2 = cv2.normalize(img, None, 0, biggest_pixel_value, cv2.NORM_MINMAX)
  # Histogram
  if img.dtype == np.uint8:
    img_2 = cv2.equalizeHist(img)
  elif img.dtype == np.uint16:
    img_2 = equalize_histogram_16bit(img)
  else:
    print("bit depth not 8 or 16. Help please.")
    continue
  
  hist1 = cv2.calcHist([img],[0],None,[int(biggest_pixel_value+1)],[0,int(biggest_pixel_value+1)])
  hist2 = cv2.calcHist([img_2],[0],None,[int(biggest_pixel_value+1)],[0,int(biggest_pixel_value+1)])

  # Setting for plots
  (fig, ax) = plt.subplots(2, 2, )
  # Display the Original image + spectrum
  ax[0][0].imshow(img, cmap="gray")
  ax[0][0].set_title("Original image")
  ax[0][1].plot(hist1)
  ax[0][1].set_title("Original pixel value range")
  # Display the histogramised image + spectrum
  ax[1][0].imshow(img_2, cmap="gray")
  ax[1][0].set_title("Histogramised image")
  ax[1][1].plot(hist2)
  ax[1][1].set_title("Histogramised pixel value range")

  # Save plot
  image_comparison_name = gs.change_file_extension(image, "") + '_comparison.png'
  print(f"Saving {image_comparison_name}...")
  plt.savefig(f"{FOLDER_FOR_HISTOGRAMISED_IMG}\{image_comparison_name}",
              bbox_inches="tight", 
              dpi=1800)
  plt.close()
  print(f"{image_comparison_name} saved")

  # Save histogramised image
  image_histogramised_name = gs.change_file_extension(image, "") + '_histogramised.png'
  print(f"Saving {image_histogramised_name}...")
  cv2.imwrite(f"{FOLDER_FOR_HISTOGRAMISED_IMG}\{image_histogramised_name}", img_2)
  print(f"{image_histogramised_name} saved")
  # plt.show()


