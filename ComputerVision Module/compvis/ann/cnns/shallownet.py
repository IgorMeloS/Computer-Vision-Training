# =============================================================================
# Shallow Convolotution Neural Network with TensorFlow and Keras
# =============================================================================

# Importing Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


class ShallowNet:
    """ShallowNet architecture is a simple convolutional neural network, composed by one convolutional layer that contains 32
    filters with 3x3 kernel.
    """
    @staticmethod
    def build(width, height, depth, classes):
        """build function. This function build the shallownet architecture.
        Args:
            width: image width
            height: image height
            depth: number of channel for the input image
            classes: number of classes to be classified.
        This function returns the model.
        """
        # Initializing the model, considering that channels is the last input
        model = Sequential()
        inputShape = (height, width, depth)
        
        # If you're working with a different image input, channel first
        if K.image_data_format() == "channels_first":
            inputShape(depth, height, width)
        # Defining the Convolution Layer
        model.add(Conv2D(32, (3,3), padding = "same", input_shape = inputShape))
        # Adding the activation function
        model.add(Activation("relu"))
        # Preparing the ANN configuration, Flatten, Dense and output (softmax)
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        # returning the structured CNN
        return model