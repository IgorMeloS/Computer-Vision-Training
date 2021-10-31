# =============================================================================
# Configuration file for the Resnet trained on Tinny Image
# =============================================================================
# Importing Libraries
from os import path

# define the paths to the training and validation directories
TRAIN_IMAGES = "/home/igor/Documents/Artificial_Inteligence/Datasets/tiny-imagenet-200/train/images"
VAL_IMAGES = "/home/igor/Documents/Artificial_Inteligence/Datasets/tiny-imagenet-200/val/images"

# grabing the validations labels
VAL_MAPPINGS = "/home/igor/Documents/Artificial_Inteligence/Datasets/tiny-imagenet-200/val/val_annotations.txt"

# define the paths to the WordNet hierarchy files which are used to generate our class labels

WORDNET_IDS = "/home/igor/Documents/Artificial_Inteligence/Datasets/tiny-imagenet-200/wnids.txt"
WORD_LABELS = "/home/igor/Documents/Artificial_Inteligence/Datasets/tiny-imagenet-200/words.txt"

# since we do not have access to the testing data we need to take a number of images from the training data and use it instead
NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# define the path to the output training, validation, and testing HDF5 files
TRAIN_HDF5 = "/home/igor/Documents/Artificial_Inteligence/Datasets/tiny-imagenet-200/hdf5/train.hdf5"
VAL_HDF5 = "/home/igor/Documents/Artificial_Inteligence/Datasets/tiny-imagenet-200/hdf5/val.hdf5"
TEST_HDF5 = "/home/igor/Documents/Artificial_Inteligence/Datasets/tiny-imagenet-200/hdf5/test.hdf5"

# define the path to the dataset mean
DATASET_MEAN = "/home/igor/Documents/Artificial_Inteligence/Formation/Computer Vision Training/12 - ResNet/output/tiny-image-net-200-mean.json"

# define the path to the output directory used for storing plots, classification reports, etc.
# Setting the Paths to store the dataset in HD5F and the output results
OUTPUT_PATH = "output"
MODEL_PATH = path.sep.join([OUTPUT_PATH, "resnet_tinyimagenet.hdf5"])
FIG_PATH = path.sep.join([OUTPUT_PATH, "resnet56_tinyimagenet.png"])
JSON_PATH = path.sep.join([OUTPUT_PATH, "resnet56_tinyimagenet.json"])

stage = (3, 4, 6)
filters = (64, 128, 256, 512)