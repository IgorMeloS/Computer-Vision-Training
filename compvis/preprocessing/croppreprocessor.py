# =============================================================================
# Crop preprocessor to boost the accuracy
# =============================================================================

# Importing libraries

import numpy as np
import cv2

# Definng the class for the cropping process
class CropPreprocessor:
    """Crop preprocessor crops the image in four corners and in the center, five new images.
    If horizontal flip is called, the total of new images become 10.
    Args:
        width: int number, new width value
        height: int number, new height value
        horiz: Boolean variable, by default True. If false, there's no horizontal flip
        inter: any OpenCV interpolation method.
    """
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
        # storing width, height, defining the inteporlation method and
        # creating a variable to decide if we consider horizontal flip
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    # Defining the preprocessor function
    def preprocess(self, image):
        """Preprocess function.
        Args:
            image: image to be preprocessed
        return array with the preprocessed images
        """
        # Initializing the crop list
        crops = []
        # grabing the width and height, we use these dimension for the cropping
        (h, w) = image.shape[:2]
        # Defining a list of coordinates to crop the images
        coords = [[0, 0, self.width, self.height],
                  [w - self.width, 0, w, self.height],
                  [w - self.width, h - self.height, w, h],
                  [0, h - self.height, self.width, h]]
        # computing the center of the crop
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])
        # Extracting the crop according the coordinates
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height),
                              interpolation = self.inter)
            crops.append(crop)
        # Checking the horizontal flip codition
        if self.horiz:
            # computing the flips for each image
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)
        # returning the crops
        return np.array(crops)
