# =============================================================================
# RGB mean preprocessing
# =============================================================================

# Importing libraries

import cv2

# Defining the class for the mean process
class MeanPreprocessor:
    """Class to realize the mean normalization, to put the pixels intensity around 0, for each channel.
    Args:
        rMean: red channel mean
        gMean: green channel mean
        bMean: blue channel mean
    """
    def __init__(self, rMean, gMean, bMean):
        # Storing the chanels
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean
    # Defining the preprocess function
    def preprocess(self, image):
        """Preprocess function
        Arg:
            image: image to be preprocessed
        return preprocessed image
        """
        # Splitting the image into the channels
        (B, G, R) = cv2.split(image.astype("float32"))

        # Subtract the channnels means
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean
        # merging the channels bact to the image
        return cv2.merge([B, G, R])
