# =============================================================================
# H5DF writer with h5py
# =============================================================================

# Importing libraries

import h5py
import os

# Defining the HD5F class constructor

class HDF5DatasetWriter:
    """Dataset writer using HDF5.
    Args:
        dims: tuple with the desired dimensions (n_elements, features size)
        outputPath: path to store the hdf5 file
        datakey: Optinal. The name of the file, by default images
        bufSize: the size of the buffer, by default 1000
    """
    def __init__(self, dims, outputPath, dataKey = "images", bufSize = 1000):
        # Checking with outputPath exist, warning in other case
        if os.path.exists(outputPath):
           
            answer = input("The supplied 'outputPath' already exist \n"
                           "Do you want overwrite (be sure)? Enter yes or no: ")    
            if answer == "yes":
                # Opening the HD5F, writing and creating classes
                os.remove(outputPath)
            elif answer == "no":
                raise ValueError("Please, manually delete the file before continuing. Or chose another name.", outputPath)
            else:
                print("Please enter yes or no.")
       
            # Opening the HD5F, writing and creating classes
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims , dtype = "float")
        self.labels = self.db.create_dataset("labels", (dims[0], ), dtype = "int")
        self.bufSize = bufSize
        self.buffer = {"data" : [], "labels" : []} # dictionary
        self.idx = 0

        # Setting the buffer size and intializing it
        
    # Function to add data to our buff
    def add(self, row, labels):
        """Adding function to store the data in the respective buffer
        Args:
            row: the features to be stored
            labels: the list of labels
        
        """
        # adding rows and labels
        self.buffer["data"].extend(row)
        self.buffer["labels"].extend(labels)
        
        # Checking if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()
    # Flushing the buffer (writing into the file and reseting the buffer)
    def flush(self):
        """ Function to flush the data into the disk.
        """
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data" : [], "labels" : []}
    # Defining function to write the row of labels if desired
    def storeClassLabels(self, classLabels):
        """Function to store the list of classes
        Arg:
            classLabels: list of classes name
        
        """
        # Creating a dataset to store the actual class labels names, then store the class labels
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names", (len(classLabels), ),
                                          dtype = dt)
        labelSet[:] = classLabels # to fill the list
    #Creating function to close the HD5f file
    def close(self):
        """Function to close the hdf5 file.
        """
        # Checking if the buff must be flushed into the disk
        if len(self.buffer["data"]) > 0:
            self.flush()
        # closing the file
        self.db.close()
            