# =============================================================================
# Training Monitor - Usign Keras callbacks
# =============================================================================

# Importing libraries
from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Building the monitor class
class TrainingMonitor(BaseLogger):
    """Training Monitor class plots the curves of the loss function and accuracy after each epoch
    Args:
        figPath: the path to save the plot
        jsonPath: path to store the json file with the history of the train process, by default None.
        startAt: int number to initialize the plot. By default 0, starting from the first epoch
    """
    def __init__(self, figPath, jsonPath=None, startAt=0):
        # Setting the paths for the monitor outputs (figures, json file and others)
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
    
    # Defining the function to start the monitor (called once on the start point)
    def on_train_begin(self, logs = {}):
        # History dictionary
        self.H = {}
        
        # Taking care about the json file and your path
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                #self.H = json.loads(open(self.jsonPath).read())
                # Verifying if there is an epoch 
                if self.startAt > 0:
                    # Cheching all entries of the dictionary and earesing the
                    # excedent epochs
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]
    # Updating the History dictionary at the end of each epoch
    def on_epoch_end(self, epoch, logs = {}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for(k, v) in logs.items():
            l = self.H.get(k, [])
            if not isinstance(v, np.float):
                v = round(float(v), 4)
            l.append(v)
            self.H[k] = l
        # check to see if the training history should be serialized
        # to file

        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()
# =============================================================================
        # Constructing the plot monitor past two epochs
        if len(self.H["loss"]) > 1 :
            
            N = np.arange(0, len(self.H["loss"])) # range of epochs
            # Plot configurations
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="train_acc")
            plt.plot(N, self.H["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig(self.figPath)
            plt.close()
        