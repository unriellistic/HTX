"""
A script that utilises FFT to detect blur images.
"""

# https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np

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
	if vis:
		# compute the magnitude spectrum of the transform
		magnitude = 20 * np.log(np.abs(fftShift))
		# display the original input image
		(fig, ax) = plt.subplots(1, 2, )
		ax[0].imshow(image, cmap="gray")
		ax[0].set_title("Input")
		ax[0].set_xticks([])
		ax[0].set_yticks([])
		# display the magnitude image
		ax[1].imshow(magnitude, cmap="gray")
		ax[1].set_title("Magnitude Spectrum")
		ax[1].set_xticks([])
		ax[1].set_yticks([])
		# show our plots
		plt.show()

	# zero-out the center of the FFT shift (i.e., remove low frequencies),
	# apply the inverse shift such that the DC component once again becomes the top-left
	# and then apply the inverse FFT
	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)

	# compute the magnitude spectrum of the reconstructed image,
	# then compute the mean of the magnitude values
	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)
	# the image will be considered "blurry" if the mean value of the
	# magnitudes is less than the threshold value
	return (mean, mean <= thresh)

if __name__ == "__main__":
	print("Wrong function called, use detect_blur_image.py")

