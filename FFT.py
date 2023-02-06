import scipy, matplotlib
from scipy.fft import rfft, fft, rfftfreq, fftfreq
import cv2, matplotlib.pyplot as plt, numpy as np, keyboard

# image = "test images\\flowers.jpeg"
image = "test images\\against_the_light.png"
# image = "test images\\autumn leaf.jpeg"
im = cv2.imread(image, 0)

# cv2.imshow('image', im)
# cv2.waitKey()
#
# plt.figure()
# plt.imshow(im)
# plt.title('Original image')
# plt.show()
im_fft = fft(im)
im_rfft = rfft(im)

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    # plt.imshow(np.abs(im_fft))
    plt.colorbar()

plt.figure()
# plot_spectrum(im_fft)
plot_spectrum(im_rfft)
plt.title('Fourier transform')
plt.show()


# Easy to close window
keyboard.wait("q")
plt.close('all')