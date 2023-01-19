import os
from PIL import Image
from pprint import pprint

import cv2
import glob
import numpy as np
import os


class ImageProcessor:

    def __init__(self, file=None):
        
        self.filenm= os.path.basename(file)
        self.img=cv2.imread(file)

    

def create_folder(dest_path):
    try:
        os.mkdir(dest_path)
    except FileExistsError:
        pass

def download_images(images_directory):
    all_images_paths = []
    images_directories = os.listdir(images_directory)
    for directory in images_directories:
        directory_relaitve_path = f"{images_directory}/{directory}"
        files = os.listdir(directory_relaitve_path)
        for file in files:
            if file.endswith(".png"):
                all_images_paths.append(f"{images_directory}/{directory}/{file}")
            else:
                 pass
    return all_images_paths