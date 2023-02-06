"""
This script calls on the original blur_detector.py script for blur detection
OR
compare_images_before_after_FFT_masking.py script for FFT checking.
It has a -s to save the images to the current folder
"""

# import the necessary packages
# from blur_detector import detect_blur_fft
from compare_images_before_after_FFT_masking import detect_blur_fft
import numpy as np
import argparse
import imutils
import cv2
import os, sys
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
# python detect_blur_image.py -i "C:\Users\alpha\PycharmProjects\HTX_Test_Projects\test images" -t 5 -v -1 -d 1 -s -1 -z 60
# detect_blur_image.py -i "../test folder" -t 40 -v -1 -d 1 -s -1 -z 60

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path input image that we'll detect blur in")
ap.add_argument("-t", "--thresh", type=int, default=40,
                help="threshold for our blur detector to fire")
ap.add_argument("-v", "--vis", type=int, default=-1,
                help="whether or not we are visualizing intermediary steps")
ap.add_argument("-d", "--test", type=int, default=-1,
                help="whether or not we should progressively blur the image")
ap.add_argument("-s", "--save", type=int, default=-1,
                help="whether or not we should save the image threshold results")
ap.add_argument("-z", "--size", type=int, default=60,
                help="size of masking to FFT")

# A list to store all the threshold if user enables --save 1
info = []


def main(args):
    args = vars(ap.parse_args())
    return args


image


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) and filename is not None:
            images.append(filename)
    return images


def detect_single_image(image, save):
    # orig = image
    orig = cv2.imread(image)
    # orig = imutils.resize(orig, width=500)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    # apply our blur detector using the FFT
    (mean, blurry) = detect_blur_fft(gray, size=args["size"], thresh=args["thresh"], vis=args["vis"] > 0)

    # draw on the image, indicating whether it is blurry
    # Add two more channels to our single-channel gray image, storing the result in image
    image = np.dstack([gray] * 3)

    # Set the color as red (if blurry) and green (if not blurry)
    color = (0, 0, 255) if blurry else (0, 255, 0)

    # Draw our blurry text indication and mean value in the top-left corner of our image
    text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
    text = text.format(mean)
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    print("[INFO] {}".format(text))

    # Show the output image until a key is pressed
    # cv2.imshow("Output", image)
    # cv2.waitKey(0)

    # If user sets --save 1, save threshold and output into excel file
    if save != "-1":
        info.append([image, mean, "Blurry" if blurry else "Not blurry", ""])

    # check to see if are going to test our FFT blurriness detector using
    # various sizes of a Gaussian kernel
    if args["test"] > 0:

        # loop over various blur radii in the range [0, 30]
        for radius in range(1, 30, 2):

            # clone the original grayscale image
            image = gray.copy()

            # check to see if the kernel radius is greater than zero
            if radius > 0:
                # applies OpenCV’s GaussianBlur method to intentionally introduce blurring in our image.
                # blur the input image by the supplied radius using a Gaussian kernel
                image = cv2.GaussianBlur(image, (radius, radius), 0)
                # apply our blur detector using the FFT
                (mean, blurry) = detect_blur_fft(image, size=args["size"],
                                                 thresh=args["thresh"], vis=args["vis"] > 0)
                # draw on the image, indicating whether or not it is blurry
                image = np.dstack([image] * 3)
                color = (0, 0, 255) if blurry else (0, 255, 0)
                text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
                text = text.format(mean)
                cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)
                print("[INFO] Kernel: {}, Result: {}".format(radius, text))
            # show the image
            cv2.imshow("Test Image", image)
            cv2.waitKey(0)


def detect_blur_fft(image, size=60, thresh=10, vis=False):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more easy to analyze
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
    ax[0][2].set_title("Magnitude Reduced Spectrum")
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
    # for i in range(10):
    #	 print("magnitude:", magnitude[i])

    # To compare the original image vs inversed image
    color = (0, 0, 255) if mean <= thresh else (0, 255, 0)
    # (fig, ax) = plt.subplots(1, 2, )
    text = "Blurry ({:.4f})" if mean <= thresh else "Not Blurry ({:.4f})"
    text = text.format(mean)
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    print("[INFO] {}".format(text))

    # Display altered image
    ax[1][0].imshow(abs(recon), cmap="gray")
    ax[1][0].set_title("Reconstructed")
    ax[1][0].set_xticks([])
    ax[1][0].set_yticks([])
    # display the inversed image after zero-out image
    ax[1][2].imshow(magnitude, cmap="gray")
    ax[1][2].set_title("Inversed image after zero-out center")
    ax[1][2].set_xticks([])
    ax[1][2].set_yticks([])
    plt.show()

    print("Mean:", mean)

    # Show reconstructed image
    cv2.imshow("Output", abs(recon))
    cv2.waitKey(0)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh)


if __name__ == "__main__":
    # python detect_blur_image.py -i "test images" -t 10 -v -1 -d 1
    # uncomment below if want to debug using pycharm in alp's laptop
    # sys.argv = ['detect_blur_image.py', '-i', "C:\\Users\\alpha\\PycharmProjects\\pythonProject3\\test images", '-t', '10', '-v', '-1', '-d', '1', '-s', '-1', 'z', '60']
    # \\autumn leaf.jpeg
    args = ap.parse_args()
    args = main(args)
    # load the input image from disk, resize it, and convert it to
    # grayscale
    # print("args is:", args)

    print(args['save'])
    if args["image"].lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        # print("inside if")
        image = cv2.imread(os.path.join(args["image"]))
        detect_single_image(image, args["save"])
    else:
        # print("inside else")
        imagedir = args['image']
        # print(imagedir)
        images = load_images_from_folder(args["image"])
        for image in images:
            # print(os.path.join(imagedir, image))
            detect_single_image(os.path.join(imagedir, image), args["save"])
    cv2.destroyAllWindows()

    if args["save"] == '1':
        # Stores info list into pd and then convert to excel format.
        df = pd.DataFrame(info, columns=['filename', 'detected_threshold', 'blurry', 'Remarks'])
        print(df)
        df.to_excel('blur_detection.xlsx', sheet_name='sheet1', index=True)