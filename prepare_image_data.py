import os
import cv2
import glob


class ImageDeveloper:

    def __init__(self, file=None):
        
        self.filename= os.path.basename(file)
        self.img=cv2.imread(file)
    
    def get_img_size(self):
        self.height, self.width, self.channels = self.img.shape

    def resize_image(self, selected_height=None):
        self.get_img_size()
        new_width = int(selected_height * (self.height / self.width))
        new_size = (selected_height,new_width)
        output = cv2.resize(self.img, new_size)
        cv2.imwrite(f"processed_images/{self.filename}", output)


def prep_images(path):
    height_list = []
    for file in path:
        developed_image = ImageDeveloper(file)
        developed_image.get_img_size()
        height_list.append(developed_image.height)
        min_height=min(height_list)
        developed_image.resize_image(min_height)
        print(min_height)
      
if __name__ == "__main__":
    print("leggo")
    path = []
    for image_path in glob.glob("image/*/*png"):
        path.append(image_path)
    prep_images(path)