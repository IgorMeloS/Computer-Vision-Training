# =============================================================================
# Mini VGG Net with TensorFlow. Code inspired on the follow paper
# http://arxiv.org/abs/1409.1556 (cited on pages 113, 192, 195, 227, 229, 278).
# =============================================================================

# Importing Libraries

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

# Constructing the class model

class MiniVGG:
    """
    Mini VGG Net is a smaller version of the VGG family of networks.
    """
    @staticmethod
    def build(width, height, depth, classes):
        """build function to define the mini VGG
        Args:
            width: input of width requires a int number.
            height: input of height requires a int number.
            depth: number of channels, requires a int number.
            classes: number of class to classify, requires a int number.
        """
        # Initializing the model with channel last
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        # For the case where we consider channel first
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # Implementing the first layer conv => act 2x
        model.add(Conv2D(32, (3, 3), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(32, (3, 3), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))
        # Implementing the second layer
        model.add(Conv2D(64, (3, 3), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(64, (3, 3), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))
        # Implementing the Fully Connected layer
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # The output layer
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        #returning the model
        return model
