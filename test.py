"""
Just a mini testing file for me to try out python logic for debugging.
"""
import os
from pathlib import Path
import glob

path = r"test images"
import general_scripts as gs

images = gs.load_images(path)