"""
This script is to compare the original image and the reconstructed image after applying FFT masking.
It includes the plots for comparison.
Parameters that can be tailored are:
weight_list: Contains the "weight" of how dark you want to mask the magnitude and fft images. Darker -> More masking
arguments:
    image: path input, can be absolute file paths to either a single image or a folder containing multiple images
    size: how many pixels width you want to mask
    save: whether do you want to save to the folder (this will save to a folder that must exist with its relevant method name)
    mask_method: how you want to mask the image
        - Name of each method can be found in the code below, roughly ~line 81

E.g. of how to call the script:
python compare_FFT_images.py --save 1 -i "D:\BusXray\Compiling_All_subfolder_images\Compiled_Threat_Images\images_for_fft_testing" -m mask_vertical_line --size 1

Made on: Roughly early Feb 2023
Last updated: 22/3/2023
@Author: Alphaeus

P.S anything not sure just ask ChatGPT explain LUL.
"""

# Constants to define
FOLDERS_TO_SAVE_IN = r"D:\BusXray\Compiling_All_subfolder_images\Compiled_Threat_Images\FFT Masking on Bus images"

# https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
# import the necessary packages
import general_scripts as gs
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os

# Initialise different arguments for command prompt
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path input image that we'll detect blur in")
ap.add_argument("-z", "--size", type=int, default=60,
                help="size of masking to FFT")
ap.add_argument("-s", "--save", type=int, default=-1,
                help="save image comparison")
ap.add_argument("-m", "--mask_method", type=str, default="center_radial",
                help="set type of masking")

# For debugging purposes on IDE rather than command prompt
def main(args):
    args = vars(ap.parse_args())
    return args

# image: Our input image for blur detection
# size: The size of the radius around the centerpoint of the image for which we will zero out the FFT shift
# thresh: A value which the mean value of the magnitudes (more on that later) will be compared to for determining whether an image is considered blurry or not blurry
# vis: A boolean indicating whether to visualize/plot the original input image and magnitude image using matplotlib
def compare_fft_image(image, size=60, save=-1, image_name="nil", mask_method='center_radial'):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)  # compute the FFT
    fftShift = np.fft.fftshift(fft)  # shift the zero frequency component (DC component) of the result to the center for easier analysis

    # check to see if we are visualizing our output

    # compute the magnitude spectrum of the transform
    magnitude = 20 * np.log(np.abs(fftShift))
    # display the original input image
    (fig, ax) = plt.subplots(2, 2, )
    ax[0][0].imshow(image, cmap="gray")
    ax[0][0].set_title("Input")
    ax[0][0].set_xticks([])
    ax[0][0].set_yticks([])
    # display the magnitude spectrum image
    ax[0][1].imshow(magnitude, cmap="gray")
    ax[0][1].set_title("Original Magnitude Spectrum")
    ax[0][1].set_xticks([])
    ax[0][1].set_yticks([])
    # display masked magnitude spectrum image
    magnitude_spectrum = magnitude

    ### The various masking methods
    # Mask the pixels in the center in a circle-shaped format
    def mask_center_radial(magnitude_spectrum, fftShift_item, radius_size=1, weight=0):
        magnitude_spectrum[cY - radius_size:cY + radius_size, cX - radius_size:cX + radius_size] = weight
        print("magnitude_spectrum:", magnitude_spectrum[cY - radius_size:cY + radius_size, cX - radius_size:cX + radius_size])
        # zero-out the center of the FFT shift (i.e., remove low frequencies),
        # apply the inverse shift such that the DC component once again becomes the top-left
        # and then apply the inverse FFT
        fftShift_item[cY - radius_size:cY + radius_size, cX - radius_size:cX + radius_size] = weight
        return magnitude_spectrum, fftShift_item
    
    # Mask the pixels in the center in a square-shaped format
    def mask_center_square(magnitude_spectrum, fftShift_item, square_size=1, weight=0):
        magnitude_spectrum[cY+square_size:cY-square_size, cX-square_size:cX+square_size] = weight
        fftShift_item[cY+square_size:cY-square_size, cX-square_size:cX+square_size] = weight
        return magnitude_spectrum,  fftShift_item

    # Not done yet. Mask the pixels in the corners in a square-shaped format.
    def mask_corner_square(magnitude_spectrum, fftShift_item, square_size=1, weight=0):
        magnitude_spectrum[cY - square_size:cY + square_size, cX - square_size:cX + square_size] = weight
        fftShift_item[cY - square_size:cY + square_size, cX - square_size:cX + square_size] = weight
        return magnitude_spectrum,  fftShift_item
    
    # Mask the pixels in the center in a horizontal-line-shaped format
    # This will cause vertical lines in the image itself to be masked
    def mask_horizontal_line(magnitude_spectrum, fftShift_item, rectangle_size=1, weight=0):
        
        # Mask all the way
        magnitude_spectrum[cY-rectangle_size:cY+rectangle_size, 0:w] = weight
        fftShift_item[cY-rectangle_size:cY+rectangle_size, 0:w] = weight

        # Mask all the way, but avoid center
        # magnitude_spectrum[cY-rectangle_size:cY+rectangle_size, 0:cX-5] = weight
        # magnitude_spectrum[cY-rectangle_size:cY+rectangle_size, cX+5:w] = weight
        # fftShift[cY-rectangle_size:cY+rectangle_size, 0:cX-5] = weight
        # fftShift[cY-rectangle_size:cY+rectangle_size, cX+5:w] = weight
        return magnitude_spectrum,  fftShift_item

    # Mask the pixels in the center in a vertical-line-shaped format
    # This will cause horizontal line in the image itself to be masked
    def mask_vertical_line(magnitude_spectrum, fftShift_item, rectangle_size=1, weight=0):
        magnitude_spectrum[0:cY-5, cX-rectangle_size:cX+rectangle_size] = weight
        magnitude_spectrum[cY+5:h, cX-rectangle_size:cX+rectangle_size] = weight
        fftShift_item[0:cY-5, cX-rectangle_size:cX+rectangle_size] = weight
        fftShift_item[cY+5:h:, cX-rectangle_size:cX+rectangle_size] = weight
        return magnitude_spectrum, fftShift_item

    # Mask the pixels in the your own custom format.
    def mask_custom_shape(magnitude_spectrum, fftShift_item, pixel_size=1, weight=0):
        magnitude_spectrum[cY-pixel_size:cY+pixel_size, cX] = weight
        magnitude_spectrum[cY, cX-pixel_size:cX+pixel_size] = weight
        fftShift_item[cY-pixel_size:cY+pixel_size, cX] = weight
        fftShift_item[cY, cX-pixel_size:cX+pixel_size] = weight
        return magnitude_spectrum, fftShift_item
    
    # Find the darkest and brightest pixel value.
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
    print(f"minVal: {minVal}, maxVal: {maxVal}, minLoc: {minLoc}, maxLoc: {maxLoc}")

    # Select which option of masking, loop through different weights set
    weight_list = [0, 200, 400]
    print("weight_list:", weight_list)

    for weight in weight_list:
        fftShift_item = np.copy(fftShift)
        if mask_method == 'mask_center_radial':
            magnitude_reduced, fftShift_item = mask_center_radial(magnitude_spectrum=magnitude_spectrum, fftShift_item=fftShift_item, radius_size=size, weight=weight)
        elif mask_method == 'mask_center_square':
            magnitude_reduced, fftShift_item = mask_center_square(magnitude_spectrum=magnitude_spectrum, fftShift_item=fftShift_item, square_size=size, weight=weight)
        elif mask_method == 'mask_corner_square':
            magnitude_reduced, fftShift_item = mask_corner_square(magnitude_spectrum=magnitude_spectrum, fftShift_item=fftShift_item, square_size=size, weight=weight)
        elif mask_method == 'mask_horizontal_line':
            magnitude_reduced, fftShift_item = mask_horizontal_line(magnitude_spectrum=magnitude_spectrum, fftShift_item=fftShift_item, rectangle_size=size, weight=weight)
        elif mask_method == 'mask_vertical_line':
            magnitude_reduced, fftShift_item = mask_vertical_line(magnitude_spectrum=magnitude_spectrum, fftShift_item=fftShift_item, rectangle_size=size, weight=weight)
        elif mask_method == 'mask_custom_shape':
            magnitude_reduced, fftShift_item = mask_custom_shape(magnitude_spectrum=magnitude_spectrum, fftShift_item=fftShift_item, pixel_size=size, weight=weight)
    
    
        ax[1][1].imshow(magnitude_reduced, cmap="gray")
        ax[1][1].set_title("Masked Magnitude Spectrum")
        ax[1][1].set_xticks([])
        ax[1][1].set_yticks([])

        fftShift_item = np.fft.ifftshift(fftShift_item)
        reconstructed_image = np.fft.ifft2(fftShift_item)

        # compute the magnitude spectrum of the reconstructed image,
        # then compute the mean of the magnitude values
        magnitude = 20 * np.log(np.abs(reconstructed_image))

        # Compare original with transformed image
        # compute the magnitude spectrum of the transform
        # display the original input image

        # Each pixel contains a magnitude value, lower it is the darker the image is.
        mean = np.mean(magnitude)
        for i in range(10):
            print("magnitude:", magnitude[i])

        # To compare the original image vs inversed image
        # Display altered image
        ax[1][0].imshow(abs(reconstructed_image), cmap="gray")
        ax[1][0].set_title(f"Recon image, weight={weight}")
        ax[1][0].set_xticks([])
        ax[1][0].set_yticks([])

        if save:
            print(f"Saving {image_name}...")
            plt.savefig(f"{FOLDERS_TO_SAVE_IN}\{mask_method}\{image_name}_{mask_method}_size={size}_weight={weight}.png",
                        bbox_inches="tight", 
                        dpi=1800)
            print(f"Saved as {image_name}_{mask_method}_size={size}_weight={weight}.png")
        else:
            plt.show()

        print("mean:", mean)
    return


if __name__ == "__main__":
    # python compare_FFT_images.py --save 1 -i "D:\BusXray\Compiling_All_subfolder_images\Compiled_Threat_Images\images_for_fft_testing\temp_image_low.tif" -m mask_vertical_line --size 1
    # uncomment below if want to debug using pycharm in alp's laptop
    # sys.argv = ['compare_FFT_images.py', '-s', '-1', '-i', "D:\BusXray\Compiling_All_subfolder_images\Compiled_Threat_Images\YOLO_removeThreat_images\PA8506K Higer 49 seats-Threat-1-final_color.jpg",
    #             '-z', '1']
    args = ap.parse_args()
    args = main(args)
    # load the input image from disk, resize it, and convert it to
    # grayscale

    if args["image"].lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')):
        print(f"Processing {args['image']}")
        image = cv2.imread(args["image"], cv2.IMREAD_ANYDEPTH)
        _, image_name = os.path.split(args['image'])
        compare_fft_image(image, args["size"], args["save"], image_name=gs.change_file_extension(image_name, ""), mask_method=args["mask_method"])
        
    else:
        imagedir = args['image']
        images = gs.load_images_from_folder(args["image"])
        for image_name in images:
            print(f"Processing {image_name}")
            # image = cv2.imread(os.path.join(imagedir, image_name), cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(os.path.join(imagedir, image_name), cv2.IMREAD_ANYDEPTH) #cv2.IMREAD_GRAYSCALE,

            # Debugging purposes
            # print(image.shape)
            # print(image[100])
            compare_fft_image(image, args["size"], args["save"], image_name=gs.change_file_extension(image_name, ""), mask_method=args["mask_method"])
    cv2.destroyAllWindows()
