import os
import cv2
import glob


class ImageDeveloper:

    def __init__(self, file=None):
        
        self.filename= os.path.basename(file)
        self.img=cv2.imread(file)
    
    def get_img_size(self):

        """
        Get the dimensions of the image.

        Retrieves and sets the height, width, and channels attributes of the image.

        """
        self.height, self.width, self.channels = self.img.shape

    def resize_image(self, selected_height=None):

        """

        Resize the image to a specified height while maintaining the aspect ratio.

        Parameters:
            selected_height (int): The desired height for the resized image.

        """
        self.get_img_size()
        new_width = int(selected_height * (self.height / self.width))
        new_size = (selected_height,new_width)
        output = cv2.resize(self.img, new_size)
        try:
            os.mkdir(f"/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Processed_Data/processed_images/{self.filename[:-6]}")
        except FileExistsError:
            pass
        cv2.imwrite(f"/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Processed_Data/processed_images/{self.filename[:-6]}/{self.filename}", output)


def prep_images(path):

    """

    Prepare images by resizing them to the minimum height in the provided path.

    Parameters:
        path (list): List of file paths for images to be prepared.

    """

    height_list = []
    for file in path:
        developed_image = ImageDeveloper(file)
        developed_image.get_img_size()
        height_list.append(developed_image.height)
        min_height=min(height_list)
        developed_image.resize_image(min_height)

def create_folder(dest_path):

    """

    Create a folder at the specified destination path if it doesn't already exist.

    Parameters:
        dest_path (str): The path where the folder should be created.


    """

    try:
        os.mkdir(dest_path)
    except FileExistsError:
        pass


if __name__ == "__main__":
    print("starting...")
    path = []
    for image_path in glob.glob("/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Raw_Data/images/*/*png"):
        path.append(image_path)
    print("resizing images")
    prep_images(path)
    print("finished :)")