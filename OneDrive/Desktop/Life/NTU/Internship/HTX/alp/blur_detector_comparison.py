"""
This script is to compare the original image and the reconstructed image after applying FFT masking.

"""

# https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import cv2
# def pyimagesearch():

	# image: Our input image for blur detection
	# size: The size of the radius around the centerpoint of the image for which we will zero out the FFT shift
	# thresh: A value which the mean value of the magnitudes (more on that later) will be compared to for determining whether an image is considered blurry or not blurry
	# vis: A boolean indicating whether to visualize/plot the original input image and magnitude image using matplotlib

def detect_blur_fft(image, size=60, thresh=10, vis=False):
	# grab the dimensions of the image and use the dimensions to
	# derive the center (x, y)-coordinates
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))

	# compute the FFT to find the frequency transform, then shift
	# the zero frequency component (i.e., DC component located at
	# the top-left corner) to the center where it will be more
	# easy to analyze
	fft = np.fft.fft2(image) # compute the FFT
	fftShift = np.fft.fftshift(fft) # shift the zero frequency component (DC component) of the result to the center for easier analysis

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
	color = (0, 0, 255) if mean <= thresh else (0, 255, 0)
	# (fig, ax) = plt.subplots(1, 2, )
	text = "Blurry ({:.4f})" if mean <= thresh else "Not Blurry ({:.4f})"
	text = text.format(mean)
	cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
	print("[INFO] {}".format(text))

	# Display altered image
	ax[1][0].imshow(abs(recon), cmap="gray")
	ax[1][0].set_title("Recon")
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
	# the image will be considered "blurry" if the mean value of the
	# magnitudes is less than the threshold value
	return (mean, mean <= thresh)

# https://www.analyticsvidhya.com/blog/2020/09/how-to-perform-blur-detection-using-opencv-in-python/
def analyticsvidhya():
	import cv2
	import argparse
	import os
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--images', required=True,)
	ap.add_argument('-t', '--threshold', type=float)
	args = vars(ap.parse_args())

	cwd = "C:\\Users\\alpha\\OneDrive\\Desktop\\Life\\NTU\\Internship\\HTX\\FFT blurring images\\test images"

	images = []

	for image in os.listdir(cwd):
		img = cv2.imread(f'{cwd}\\{image}')
		images.append(img)

	for image in images:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		fm = cv2.Laplacian(gray, cv2.CV_64F).var()
		text = "Not Blurry"

		if fm < args["threshold"]:
			text = "Blurry"

		cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
		cv2.imshow("Image", image)
		cv2.waitKey(0)

if __name__ == "__main__":
	print("Wrong function called, use detect_blur_image.py")

