# =============================================================================
# Training Dogs vs Cats dataset with AlexNet
# =============================================================================

# Importing Libraries

import matplotlib
matplotlib.use("Agg")

from config import dogs_vs_cats_config as config
from compvis.preprocessing import ImageToArrayPreprocessor
from compvis.preprocessing import SimplePreprocessor
from compvis.preprocessing import PatchPreprocessor
from compvis.preprocessing import MeanPreprocessor
from compvis.callbacks import TrainingMonitor
from compvis.io import HDF5DatasetGenerator
from compvis.nn.cnns import AlexNet
#from compvis.memory import SetMem
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import json
import os
import h5py

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config_gpu = tf.compat.v1.ConfigProto()
config_gpu.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config_gpu.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config_gpu)
set_session(sess)

# def limitgpu(maxmem):
#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         # Restrict TensorFlow to only allocate a fraction of GPU memory
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_virtual_device_configuration(gpu,
#                         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=maxmem)])
#         except RuntimeError as e:
#             # Virtual devices must be set before GPUs have been initialized
#             print(e)
#
# limitgpu(1024*3 + 512)

# Constructing the image generator

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
                         height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

## Initializing the data pre-processor
# Loading the RGB file with the respectives means
means = json.loads(open("output/dogs_vs_cats_mean.json").read())

# Initializing the image preprocessor
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()
# Defining the data generator for the training and validation set
train_data = h5py.File(config.TRAIN_HDF5, "r")
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug = aug,
                                                     preprocessors = [pp, mp, iap],
                                                     classes = 2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64, preprocessors = [sp, mp, iap],
                                                   classes = 2)
# Initializing the model

print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

# train the network

model.fit(trainGen.generator(), steps_per_epoch=trainGen.numImages // 64,
                    validation_data=valGen.generator(), validation_steps=valGen.numImages // 64,
                    epochs=75, max_queue_size=64 * 2, callbacks=callbacks, verbose=1, workers = 1,
                    class_weight=None)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)
# close the HDF5 datasets
print("[INFO] Closing the files...")
trainGen.close()
valGen.close()
print("[INFO] End of the program.")
