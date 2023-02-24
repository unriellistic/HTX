import cv2
import matplotlib.pyplot as plt
import general_scripts as gs
import argparse 

# python ssim_script_modified.py --dir "D:\BusXray\Compiling_All_subfolder_images\test_compiled_clean_images" --dir2 "D:\BusXray\Compiling_All_subfolder_images\Compiled_Threat_Images\removeThreat_images"
def options():
  parser = argparse.ArgumentParser(description="Read image metadata")
  parser.add_argument("-d", "--dir", help="Input clean image file folder.", required=True)
  parser.add_argument("-i", "--dir2", help="Input threat image file folder", default="")
  args = parser.parse_args()
  return args


args = options()

img = cv2.imread('/image_1.jpg',0)
hist1 = cv2.calcHist([img],[0],None,[256],[0,256])
img_2 = cv2.equalizeHist(img)
hist2 = cv2.calcHist([img_2],[0],None,[256],[0,256])
plt.subplot(221),plt.imshow(img);
plt.subplot(222),plt.plot(hist1);
plt.subplot(223),plt.imshow(img_2);
plt.subplot(224),plt.plot(hist2);