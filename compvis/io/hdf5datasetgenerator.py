# =============================================================================
# Dataset generator from an HDF5 file
# =============================================================================

# Importing Libraries

import tensorflow as tf
import numpy as np
import h5py

# Defining the class to generate the dataset

class HDF5DatasetGenerator:
    """HDF5 dataset generator reads HDF5 during the training and test process.
    This class is utility for larger datasets avoiding to run out of memory.
    Args:
        dbPath: string or list of strings containing the path for the hdf5 file
        batchSize: int number with the desired batch size
        preprocessor: by default None, list of preprocessor objects
        aug: by default None. It's possible to call any data augmentation function, as ImageDataGenerator, for example.
        binarize: by default True, it converts the labels into one hot encode format. If False the labels will be not encoded
        classes: int number with the total of classes.
    """
    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None,
                 binarize=True, classes=2):
        # Storing the class variables
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        # Opening the HDF5 database and determing the number of entries
        self.db = h5py.File(dbPath, "r")
        self.numImages = self.db["labels"].shape[0]
    # Defining the generator function
    def generator(self, passes=np.inf):
        """Generator function.
        Arg:
            passes: total number of epochs
        """
        # Initialize the epoch count
        epochs = 0
        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # Loop over the HDF5 file
            for i in np.arange(0, self.numImages, self.batchSize):
                # Extracting the images
                images = self.db["images"][i : i + self.batchSize]
                labels = self.db["labels"][i : i + self.batchSize]
                # Checking the binarizer condition
                if self.binarize:
                    labels = tf.keras.utils.to_categorical(labels, self.classes)
                # Checking if any preprocess was clamed
                if self.preprocessors is not None:
                    procImages = []

                    # Looping over the images
                    for image in images:
                        # applying the prepocessor to each image
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        procImages.append(image)
                    # updating the list of images
                    images = np.array(procImages)
                # If we consider data augumentation
                

                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels,
                                                          batch_size=self.batchSize, sample_weight=None))
                yield (images, labels)
            # increment the total number of epochs
            epochs += 1
    # Defining the close function
    def close(self):
        self.db.close()
