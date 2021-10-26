# =============================================================================
# Configuration file for Dogs vs Cats dataset
# =============================================================================

# Defining the path for the dataset
IMAGES_PATH = "/home/igor/Documents/Artificial_Inteligence/Datasets/Cats and dogs /train"

# Defining the size of training data
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# Defining the paths to the hdf5 files (training, test, validation...)

TRAIN_HDF5 = "/home/igor/Documents/Artificial_Inteligence/Datasets/Cats and dogs /hdf5/train.hdf5"
VAL_HDF5 = "/home/igor/Documents/Artificial_Inteligence/Datasets/Cats and dogs /hdf5/val.hdf5"
TEST_HDF5 = "/home/igor/Documents/Artificial_Inteligence/Datasets/Cats and dogs /hdf5/test.hdf5"
FEATURES = "/home/igor/Documents/Artificial_Inteligence/Datasets/Cats and dogs /hdf5/features.hdf5"
# Path to the outpu model file
MODEL_PATH = "output/alex_net_dogs_vs_cats.model"

# define the Path to the dataset mean

DATASET_MEAN = "output/dogs_vs_cats_mean.json"

# Defining the Path to the generals outputs

OUTPUT_PATH = "output"
