"""
A script to see how to break down an image via FFT into it's sine wave components.
"""

from scipy.fft import rfft, fft
import matplotlib.pyplot as plt, numpy as np, keyboard
from PIL import Image

im = Image.open('PA8506K Higer 49 seats-clean-1-1 Monochrome.tiff')
data = np.array(im, dtype=float)
print("data:", np.shape(data))

im_fft = fft(im)
im_rfft = rfft(im)

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

plt.figure()
plot_spectrum(im_fft)
plt.title('Fourier transform')
plt.show()

# Easy to close window
keyboard.wait("q")
plt.close('all')