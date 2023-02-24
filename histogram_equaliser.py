"""
A script to perform histogram equaliser on images.
"""
import cv2
import matplotlib.pyplot as plt
import general_scripts as gs
import os, numpy as np

# python ssim_script_modified.py --dir "D:\BusXray\Compiling_All_subfolder_images\test_compiled_clean_images" --dir2 "D:\BusXray\Compiling_All_subfolder_images\Compiled_Threat_Images\removeThreat_images"
CLEAN_IMR_DIR = r"D:\BusXray\Compiling_All_subfolder_images\compiled_clean_images"
FOLDER_FOR_HISTOGRAMISED_IMG = r"D:\BusXray\Compiling_All_subfolder_images\test_compiled_clean_images_storage"
THREAT_IMG_DIR = r"D:\BusXray\Compiling_All_subfolder_images\Compiled_Threat_Images\removeThreat_images"

# def equalize_histogram_16bit(src_img):
#     dst_img = src_img.copy()

#     # Create histogram and lookup tables
#     hist_size = 65536
#     hist = np.zeros(hist_size, dtype=np.int32)
#     lut = np.zeros(hist_size, dtype=np.int32)

#     # Compute histogram
#     for y in range(src_img.shape[0]):
#         for x in range(src_img.shape[1]):
#             intensity = src_img[y, x]
#             hist[intensity] += 1

#     # Check if image has a single intensity value
#     i = 0
#     while hist[i] == 0:
#         i += 1

#     total_pixels = src_img.size
#     if hist[i] == total_pixels:
#         dst_img[:, :] = i
#         return dst_img

#     # Compute lookup table
#     scale = float(hist_size - 1) / (total_pixels - hist[i])
#     sum = 0
#     for j in range(i, hist_size):
#         sum += hist[j]
#         lut[j] = int(round(sum * scale))

#     # Apply lookup table
#     for y in range(src_img.shape[0]):
#         for x in range(src_img.shape[1]):
#             intensity = src_img[y, x]
#             dst_img[y, x] = lut[intensity]

#     return dst_img

# def equalize_histogram_16bit(src_img):
#     # Create a copy of the source image to be used as the output
#     dst_img = src_img.copy()

#     # Compute the histogram of pixel intensities in the source image
#     hist_size = 2**16
#     hist, _ = np.histogram(src_img, bins=hist_size, range=(0, hist_size-1))

#     # Find the minimum non-zero value in the histogram
#     i = 0
#     while hist[i] == 0:
#         i += 1

#     # Check if the histogram is already flat (i.e., all pixels have the same value)
#     total_pixels = src_img.size
#     if hist[i] == total_pixels:
#         dst_img[:, :] = i
#         return dst_img

#     # Compute the lookup table for mapping pixel intensities
#     scale = float(hist_size - 1) / (total_pixels - hist[i])
#     lut = np.zeros(hist_size, dtype=np.int32)
#     lut[i:] = np.round((np.cumsum(hist[i:]) - hist[i]) * scale).astype(np.int32)

#     # Apply the lookup table to the source image to create the output image
#     np.take(lut, src_img, out=dst_img)

#     return dst_img

# def equalize_histogram_16bit(src_img):
#     # Create a copy of the input image
#     dst_img = src_img.copy()

#     # Compute histogram
#     hist_size = 2**16
#     hist = np.zeros(hist_size, dtype=int)
#     for pixel_val in src_img.flat:
#         hist[pixel_val] += 1

#     # Check if image is already equalized
#     i = 0
#     while hist[i] == 0:
#         i += 1
#     total_pixels = src_img.size
#     if hist[i] == total_pixels:
#         dst_img[:, :] = i
#         return dst_img

#     # Compute lookup table
#     scale = float(hist_size - 1) / (total_pixels - hist[i])
#     lut = np.zeros(hist_size, dtype=np.float32)
#     lut[i:] = (np.cumsum(hist[i:]) - hist[i]/2.0) * scale
#     lut = np.round(lut)
#     lut = np.clip(lut, 0, hist_size-1).astype(np.uint16)

#     # Apply lookup table to input image
#     dst_img = lut[src_img]

#     return dst_img

def equalize_histogram_16bit(img):
    # Scale pixel values to span the full dynamic range of the 16-bit image
    img_min = np.min(img)
    img_max = np.max(img)
    img_range = img_max - img_min
    img_stretched = np.clip((img - img_min) * (65535 / img_range), 0, 65535).astype(np.uint16)

    # Compute histogram
    hist_size = 65536
    hist = cv2.calcHist([img_stretched], [0], None, [hist_size], [0, hist_size])

    # Compute lookup table
    total_pixels = img_stretched.size
    lut = np.zeros(hist_size, dtype=np.float32)
    for i in range(hist_size):
        scale = float(hist_size - 1) / (total_pixels - hist[i])
        sum = 0
        for j in range(i, hist_size):
            sum += hist[j]
            lut[j] = np.round(sum * scale).astype(np.uint16)

    # Apply lookup table to stretched image
    img_equalized = cv2.LUT(img_stretched, lut)

    # Inverse contrast stretching to restore original dynamic range
    img_equalized = np.clip((img_equalized / 65535) * img_range + img_min, 0, 65535).astype(np.uint16)

    return img_equalized


list_of_clean_images = gs.load_images_from_folder(CLEAN_IMR_DIR)
one_set_of_bus_model = [i for i in list_of_clean_images if 'PC603E Volvo' and 'Monochrome' in i]

for image in one_set_of_bus_model:
  img = cv2.imread(os.path.join(CLEAN_IMR_DIR, image), cv2.IMREAD_ANYDEPTH)
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
  print(f"{image_comparison_name} saved")

  # Save histogramised image
  image_histogramised_name = gs.change_file_extension(image, "") + '_histogramised.png'
  print(f"Saving {image_histogramised_name}...")
  cv2.imwrite(f"{FOLDER_FOR_HISTOGRAMISED_IMG}\{image_comparison_name}", img_2)
  print(f"{image_histogramised_name} saved")
  plt.show()


