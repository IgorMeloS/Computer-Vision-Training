# =============================================================================
# Fine-Tuning, from start to finish
# =============================================================================

# Importing Libraries

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from compvis.preprocessing import ImageToArrayPreprocessor
from compvis.preprocessing import ResizeAR
from compvis.datasets import SimpleDatasetLoader
from compvis.nn.cnns import FCHeadNet # class to create the top of the network
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from imutils import paths
import numpy as np
import os

# Setting the path for the dataset

dataset = "/home/igor/Documents/Artificial_Inteligence/Datasets/17flowers" # the path for the dataset

print("[INFO] loading images ...")
imagePaths = list(paths.list_images(dataset))

classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# Constructing the image genarator for data augumentation

aug = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.1,
                         height_shift_range = 0.1, shear_range = 0.2,
                         horizontal_flip = True, fill_mode = "nearest")

# Intializing the data processing

aap = ResizeAR(224, 224)
iap = ImageToArrayPreprocessor()

# Loading the dataser from disk
sdl = SimpleDatasetLoader(preprocessors = [aap, iap])
(data, labels) = sdl.load(imagePaths, verbose = 500)
data = data.astype("float") / 255.0

# Splitting the dataset into training and validation set

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size = 0.25,
                                                      random_state = 42)

# Converting the labels

y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

## Building the model
# Loading the VGG16 and ensuring the the FC is setted off

baseModel = VGG16(weights="imagenet", include_top = False, input_tensor=Input((224, 224, 3)))

# Initializing the new FC
headModel = FCHeadNet.build(baseModel, len(classNames), 256)

# placing the head FC model on top of the base model

model = Model(baseModel.input, headModel)



# Freezing the parameters from the pre-trained network

for layer in baseModel.layers:
    layer.trainable = False

# Compiling the model for the fine-tuning model
 
print("[INFO] compiling the model...")

opt = RMSprop(lr = 0.001)
model.compile(loss= "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

print(model.summary())

# Training the actual FC layers in a few epochs to learn with the weights from VGG16
bs = 32
H = model.fit(aug.flow(X_train, y_train, batch_size=bs), 
              validation_data = (X_test, y_test),
              epochs = 25, 
              steps_per_epoch = len(X_train) // bs,
              verbose = 1)

print("[INFO] evaluating the model...")
predictions = model.predict(X_test, batch_size = bs)
cr = classification_report(y_test.argmax(axis = 1),
                           predictions.argmax(axis = 1),
                           target_names=classNames)
print(cr)
print("[INFO] serializing model without fine tuning...")
#model.save("hdf5/no_fine_tuning.hdf5")

import matplotlib.pyplot as plt

acc = H.history['accuracy']
val_acc = H.history['val_accuracy']

loss = H.history['loss']
val_loss = H.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig("output/no_fine_tuning.jpg")

# =============================================================================
# Fine Tuning
# =============================================================================

# Let's unfreezing some network's layers 
# We train the model from the last covolutional layer until the head
for layer in baseModel.layers[15:]:
    layer.trainable = True
 

print("[INFO] re-compiling model...")
opt = SGD(lr = 0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print(model.summary())
# re-training the model
print("[INFO] fine-tuning model...")
H_fine = model.fit(aug.flow(X_train, y_train, batch_size=bs), 
                             validation_data=(X_test, y_test), 
                             epochs=100,
                             steps_per_epoch=len(X_train) // bs, 
                             verbose=1)
# evaluate the network on the fine-tuned model
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(X_test, batch_size = bs)
cr = classification_report(y_test.argmax(axis = 1), predictions.argmax(axis = 1),
                           target_names=classNames)
print(cr)
acc += H_fine.history['accuracy']
val_acc += H_fine.history['val_accuracy']

loss += H_fine.history['loss']
val_loss += H_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([25-1,25-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([25-1,25-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig("output/fine_tuning.jpg")
# save the model to disk
print("[INFO] serializing model...")
model.save("hdf5/fine_tuning.hdf5")
