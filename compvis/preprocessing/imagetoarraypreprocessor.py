# =============================================================================
# Image preprocessing using TensorFlow and Keras
# Image to array
# https://www.tensorflow.org/versions/r2.1/api_docs/python/tf/keras/preprocessing/image/img_to_array
# =============================================================================

# Importing Libraries
import tensorflow as tf
from tf.keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    """Image to array preprpocessor
    Args:
        dataFormat: optional parameter. By default None (indicate the keras.json must be used).  Other values are channel_first and channel_last.
    
    """
    def __init__(self, dataFormat = None):
        # Image data Format (row, column, channel) or (channel, row, column)
        self.dataFormat = dataFormat
    
    def preprocess(self, image):
        """preprocess function
        Args:
            image: image to be placed into array
        """
        # Applying Keras function to convert image into array with the specific
        # format
        return img_to_array(image, data_format=self.dataFormat)
