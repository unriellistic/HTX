"""
A script used to do exploratory of the different filters when applied to images
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image
from scipy import fftpack
from skimage.io import imread

def plot(data, title):
    plot.i += 1
    plt.subplot(2,2,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
plot.i = 0

def high_pass():

    # Load the data...
    im = Image.open('PA8506K Higer 49 seats-clean-1-1 Monochrome.tiff')
    data = np.array(im, dtype=float)
    print("data:", np.shape(data))
    # data = data[:,:,None] # Add singleton dimension?
    print("data:", np.shape(data))
    plot(data, 'Original')

    # A very simple and very narrow highpass filter
    kernel = np.array([[-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1]])
    highpass_3x3 = ndimage.convolve(data, kernel)
    plot(highpass_3x3, 'Simple 3x3 Highpass')

    # A slightly "wider", but sill very simple highpass filter 
    kernel = np.array([[-1, -1, -1, -1, -1],
                    [-1,  1,  2,  1, -1],
                    [-1,  2,  4,  2, -1],
                    [-1,  1,  2,  1, -1],
                    [-1, -1, -1, -1, -1]])
    highpass_5x5 = ndimage.convolve(data, kernel)
    plot(highpass_5x5, 'Simple 5x5 Highpass')

    # Another way of making a highpass filter is to simply subtract a lowpass
    # filtered image from the original. Here, we'll use a simple gaussian filter
    # to "blur" (i.e. a lowpass filter) the original.
    lowpass = ndimage.gaussian_filter(data, 3)
    gauss_highpass = data - lowpass
    plot(gauss_highpass, r'Gaussian Highpass, $\sigma = 3 pixels$')

    plt.show()

def get_FTT_spectrum():
    im = Image.open('PA8506K Higer 49 seats-clean-1-1 Monochrome.tiff') # assuming an RGB image
    # valid_imshow_data(im)
    plt.figure(figsize=(10,10))
    plt.imshow(im, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()
    # F1 = fftpack.fft2((im).astype(float))
    F1 = fftpack.fft2(im)
    F2 = fftpack.fftshift(F1)
    plt.figure(figsize=(10,10))
    plt.imshow( (20*np.log10( 0.1 + F2)).astype(int), cmap=plt.cm.gray)
    plt.show()

# A checking function to see whether the data has how many dimensions and how many dimensions should it have
def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        print('Data has 2 dimensions')
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            print("data[2] is inbetween 3 and 4 dimension")
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False

if __name__ == "__main__":
    
    get_FTT_spectrum()