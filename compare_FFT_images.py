"""
This script is to compare the original image and the reconstructed image after applying FFT masking.
It includes the plots for comparison.
"""

# https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
# import the necessary packages
from general_scripts import load_images_from_folder
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os, sys

# compare_FFT_images.py -i "../test folder" -z 1

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path input image that we'll detect blur in")
ap.add_argument("-z", "--size", type=int, default=60,
                help="size of masking to FFT")


def main(args):
    args = vars(ap.parse_args())
    return args


# image: Our input image for blur detection
# size: The size of the radius around the centerpoint of the image for which we will zero out the FFT shift
# thresh: A value which the mean value of the magnitudes (more on that later) will be compared to for determining whether an image is considered blurry or not blurry
# vis: A boolean indicating whether to visualize/plot the original input image and magnitude image using matplotlib

def compare_fft_image(image, size=60):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)  # compute the FFT
    fftShift = np.fft.fftshift(
        fft)  # shift the zero frequency component (DC component) of the result to the center for easier analysis

    # check to see if we are visualizing our output

    # compute the magnitude spectrum of the transform
    magnitude = 20 * np.log(np.abs(fftShift))
    # display the original input image
    (fig, ax) = plt.subplots(2, 3, )
    ax[0][0].imshow(image, cmap="gray")
    ax[0][0].set_title("Input")
    ax[0][0].set_xticks([])
    ax[0][0].set_yticks([])
    # display the magnitude image
    ax[0][1].imshow(magnitude, cmap="gray")
    ax[0][1].set_title("Magnitude Spectrum")
    ax[0][1].set_xticks([])
    ax[0][1].set_yticks([])
    # display size reduction image
    magnitude_reduced = magnitude
    magnitude_reduced[cY - size:cY + size, cX - size:cX + size] = 0
    ax[0][2].imshow(magnitude_reduced, cmap="gray")
    ax[0][2].set_title("Magnitude Spectrum")
    ax[0][2].set_xticks([])
    ax[0][2].set_yticks([])

    # show our plots
    # plt.show()
    # print("magnitude's shape:", np.shape(magnitude))

    # zero-out the center of the FFT shift (i.e., remove low frequencies),
    # apply the inverse shift such that the DC component once again becomes the top-left
    # and then apply the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))

    # Compare original with transformed image
    # compute the magnitude spectrum of the transform
    # display the original input image

    # Each pixel contains a magnitude value, lower it is the darker the image is.
    mean = np.mean(magnitude)
    for i in range(10):
        print("magnitude:", magnitude[i])

    # To compare the original image vs inversed image
    # Display altered image
    ax[1][0].imshow(abs(recon), cmap="gray")
    ax[1][0].set_title("Reconstructed image")
    ax[1][0].set_xticks([])
    ax[1][0].set_yticks([])
    # display the inversed image after zero-out image
    ax[1][2].imshow(magnitude, cmap="gray")
    ax[1][2].set_title("Inversed image after zero-out center")
    ax[1][2].set_xticks([])
    ax[1][2].set_yticks([])
    plt.show()

    print("mean:", mean)

    # Show recon image
    cv2.imshow("Output", abs(recon))
    cv2.waitKey(0)


if __name__ == "__main__":
    # python compare_FFT_images.py -i "test images\\PA8506K Higer 49 seats-clean-1-1 Monochrome.tiff"" -z 1
    # uncomment below if want to debug using pycharm in alp's laptop
    sys.argv = ['compare_FFT_images.py', '-i', "test images\\PA8506K Higer 49 seats-clean-1-1 Monochrome.tiff",
                '-z', '1']
    # \\autumn leaf.jpeg
    args = ap.parse_args()
    args = main(args)
    # load the input image from disk, resize it, and convert it to
    # grayscale

    if args["image"].lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        image = cv2.imread(os.path.join(args["image"]), cv2.IMREAD_GRAYSCALE)
        compare_fft_image(image, args["size"])
    else:
        imagedir = args['image']
        images = load_images_from_folder(args["image"])
        for image_name in images:
            image = cv2.imread(os.path.join(imagedir, image_name), cv2.IMREAD_GRAYSCALE)
            compare_fft_image(image, args["size"])
    cv2.destroyAllWindows()
