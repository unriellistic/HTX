"""
Simple exploration of how does masking on an FFT transformed image work
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

import keyboard
from PIL import Image

dark_image = Image.open('PA8506K Higer 49 seats-clean-1-1 DualEnergy.tiff')

# Converts image into grey scale (TURN THIS ON FOR IMAGES WITH COLOUR)
dark_image_grey = rgb2gray(dark_image)

# plt.figure(num=None, figsize=(8, 6), dpi=80)
# plt.imshow(dark_image_grey, cmap='gray')


# Use FFT function in skimage
dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(dark_image_grey))
# plt.figure(num=None, figsize=(8, 6), dpi=80)
# plt.imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')


def fourier_masker_vertical(image, i):
    f_size = 15
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(rgb2gray(image)))
    dark_image_grey_fourier[:225, 235:240] = i
    dark_image_grey_fourier[-225:, 235:240] = i
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    ax[0].set_title('Masked Fourier', fontsize=f_size)
    ax[1].imshow(rgb2gray(image), cmap='gray')
    ax[1].set_title('Greyscale Image', fontsize=f_size);
    ax[2].imshow(abs(np.fft.ifft2(dark_image_grey_fourier)),
                 cmap='gray')
    ax[2].set_title('Transformed Greyscale Image',
                    fontsize=f_size);

def fourier_masker_horizontal(image, i):
    f_size = 15
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(rgb2gray(image)))

    # [235:240, :xxx] represents the width, the y-coordinates that you want to mask.
    # [xxx:xxx, :230] represents the length, the x-coordinates that you want to mask.
    dark_image_grey_fourier[235:240] = i
    # dark_image_grey_fourier[100:105,-230:] = i
    fig, ax = plt.subplots(1,3,figsize=(15,15))
    ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    ax[0].set_title('Masked Fourier', fontsize = f_size)
    ax[1].imshow(rgb2gray(image), cmap = 'gray')
    ax[1].set_title('Greyscale Image', fontsize = f_size)
    ax[2].imshow(abs(np.fft.ifft2(dark_image_grey_fourier)),
                     cmap='gray')
    ax[2].set_title('Transformed Greyscale Image',
                     fontsize = f_size)

def fourier_iterator(image, value_list):
    for i in value_list:
        fourier_masker_horizontal(image, i)


def fourier_transform_rgb(image):
    f_size = 25
    transformed_channels = []
    for i in range(3):
        rgb_fft = np.fft.fftshift(np.fft.fft2((image[:, :, i])))
        rgb_fft[:225, 235:237] = 1
        rgb_fft[-225:, 235:237] = 1
        transformed_channels.append(abs(np.fft.ifft2(rgb_fft)))

    final_image = np.dstack([transformed_channels[0].astype(int),
                             transformed_channels[1].astype(int),
                             transformed_channels[2].astype(int)])

    fig, ax = plt.subplots(1, 2, figsize=(15, 11))
    ax[0].imshow(image)
    ax[0].set_title('Original Image', fontsize=f_size)
    ax[0].set_axis_off()

    ax[1].imshow(final_image)
    ax[1].set_title('Transformed Image', fontsize=f_size)
    ax[1].set_axis_off()

    fig.tight_layout()


fourier_iterator(dark_image, [0.001, 1, 100])
# fourier_transform_rgb(dark_image)
plt.show()

keyboard.wait("q")
plt.close('all')