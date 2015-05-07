from loader import MNIST
import numpy as np

# This is intended to be called from the directory above

mndata = MNIST('./data')
# Load a list with training images and training labels
training_ims, training_labels = mndata.load_training()
testing_ims, testing_labels = mndata.load_testing()

# Transform everything into array
training_ims = np.array(training_ims)
training_labels = np.array(training_labels)

# Get 0 and 1 values
x = training_labels == 0
y = training_labels == 1
indexes = x + y

training_ims = training_ims[indexes]
training_labels = training_labels[indexes]
