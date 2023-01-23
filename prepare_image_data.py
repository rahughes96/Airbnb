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
        try:
            os.mkdir(f"/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb_Git_Repo/AirbnbDataSci/processed_images/{self.filename[:-6]}")
        except FileExistsError:
            pass
        cv2.imwrite(f"/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb_Git_Repo/AirbnbDataSci/processed_images/{self.filename[:-6]}/{self.filename}", output)


def prep_images(path):
    height_list = []
    for file in path:
        developed_image = ImageDeveloper(file)
        developed_image.get_img_size()
        height_list.append(developed_image.height)
        min_height=min(height_list)
        developed_image.resize_image(min_height)

def create_folder(dest_path):
    try:
        os.mkdir(dest_path)
    except FileExistsError:
        pass


if __name__ == "__main__":
    print("lets rumble")
    create_folder("/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb_Git_Repo/AirbnbDataSci/processed_images")
    path = []
    for image_path in glob.glob("/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb_Git_Repo/AirbnbDataSci/images/*/*"):
        path.append(image_path)
    print("resizing images")
    prep_images(path)
    print("finished :)")