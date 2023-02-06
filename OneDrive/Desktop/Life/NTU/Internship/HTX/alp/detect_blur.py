# import the necessary packages
# from blur_detector import pyimagesearch
from blur_detector import detect_blur_fft
import numpy as np
import argparse
import imutils
import cv2
import os, sys

# construct the argument parser and parse the arguments
# detect_blur.py -i "../test folder" -t 40 -v -1 -d 1 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path input image that we'll detect blur in")
ap.add_argument("-t", "--thresh", type=int, default=40,
	help="threshold for our blur detector to fire")
ap.add_argument("-v", "--vis", type=int, default=-1,
	help="whether or not we are visualizing intermediary steps")
ap.add_argument("-d", "--test", type=int, default=-1,
	help="whether or not we should progressively blur the image")

def main(args):
	args = vars(ap.parse_args())
	return args

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) and filename is not None:
            images.append(filename)
    return images

def detect_single_image(image):
	# orig = image
	orig = cv2.imread(image)
	# orig = imutils.resize(orig, width=500)
	gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
	# apply our blur detector using the FFT
	(mean, blurry) = detect_blur_fft(gray, size=60, thresh=args["thresh"], vis=args["vis"] > 0)

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
	cv2.imshow("Output", image)
	cv2.waitKey(0)

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
				(mean, blurry) = detect_blur_fft(image, size=60,
					thresh=args["thresh"], vis=args["vis"] > 0)
				# draw on the image, indicating whether or not it is
				# blurry
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


if __name__ == "__main__":
	# python detect_blur_image.py -i "test images" -t 10 -v -1 -d 1
	# uncomment below if want to debug using pycharm in alp's laptop
	# sys.argv = ['detect_blur_image.py', '-i', "C:\\Users\\alpha\\PycharmProjects\\pythonProject3\\test images", '-t', '10', '-v', '-1', '-d', '1']
	# \\autumn leaf.jpeg
	args = ap.parse_args()
	args = main(args)
	# load the input image from disk, resize it, and convert it to
	# grayscale
	# print("args is:", args)
	if args["image"].lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
		# print("inside if")
		image = cv2.imread(os.path.join(args["image"]))
		detect_single_image(image)
	else:
		# print("inside else")
		imagedir = args['image']
		# print(imagedir)
		images = load_images_from_folder(args["image"])
		for image in images:
			# print(os.path.join(imagedir, image))
			detect_single_image(os.path.join(imagedir, image))
	cv2.destroyAllWindows()