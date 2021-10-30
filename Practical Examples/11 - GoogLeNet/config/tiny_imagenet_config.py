# =============================================================================
# Configuration File for the Tiny Imagenet Challenge
# =============================================================================

# Importing Libraries
from os import path

# define the paths to the training and validation directories
TRAIN_IMAGES = "/path/to/tiny-imagenet-200/train"
VAL_IMAGES = "/path/to/tiny-imagenet-200/val"

# grabing the validations labels
VAL_MAPPINGS = "/path/to/tiny-imagenet-200/val/val_annotations.txt"

# define the paths to the WordNet hierarchy files which are used to generate our class labels

WORDNET_IDS = "/path/to/tiny-imagenet-200/wnids.txt"
WORD_LABELS = "/path/to/tiny-imagenet-200/words.txt"

# since we do not have access to the testing data we need to take a number of images from the training data and use it instead
NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# define the path to the output training, validation, and testing HDF5 files
TRAIN_HDF5 = "/path/to/tiny-imagenet-200/hdf5/train.hdf5"
VAL_HDF5 = "/path/to/tiny-imagenet-200/hdf5/val.hdf5"
TEST_HDF5 = "/path/to/tiny-imagenet-200/hdf5/test.hdf5"

# define the path to the dataset mean
DATASET_MEAN = "/path/to/output/tiny-image-net-200-mean.json"

# define the path to the output directory used for storing plots, classification reports, etc.
OUTPUT_PATH = "/path/to/output"
MODEL_PATH = path.sep.join([OUTPUT_PATH,
"checkpoints/epoch_70.hdf5"])
FIG_PATH = path.sep.join([OUTPUT_PATH,
"deepergooglenet_tinyimagenet.png"])
JSON_PATH = path.sep.join([OUTPUT_PATH,
"deepergooglenet_tinyimagenet.json"])