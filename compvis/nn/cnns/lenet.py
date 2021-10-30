# =============================================================================
# LeNet network 
# =============================================================================

# Importing Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class LeNet:
    """
    LeNet architecture
    Args:
        width: input width (int value)
        height: input height (int value)
        depth: number of channels (int value)
        classes: number of classes (int value)
    """
    @staticmethod
    def build(width, height, depth, classes):
        # Initializing the model
        model = Sequential()
        # architecture size
        inputShape = (height, width, depth)
        
        # Verifying the channel ordering
        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)
        
        # Setting the first layers of the CNN
        model.add(Conv2D(20, (5, 5), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        # Setting the second set of layers
        model.add(Conv2D(50, (5, 5), padding = "same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
        # Setting the Full Connected layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        # Setting the output layers
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model
    
