# =============================================================================
# Patch preprocessor
# =============================================================================

# Importing libraries

from sklearn.feature_extraction.image import extract_patches_2d

# Defining the class for the Patch process

class PatchPreprocessor:
    """Patch preprocessor crops images into a desired dimensions.
    Args:
        width: int number, for the new width
        height: int number, for the new height
    """
    def __init__(self, width, height):
        # store the target width and height of the picture
        self.width = width
        self.height = height
    # Defining the preprocess function
    def preprocess(self, image, n_image = 1):
        """Preprocess function.
        Args:
            image: image to be cropped
            n_image: desired number of cropped image, by default 1
        return cropped image
        """
        # randomly crop with the width and height
        return extract_patches_2d(image, (self.height, self.width), max_patches=n_image)[0]
    