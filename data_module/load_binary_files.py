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

# Make them binary
training_ims[training_ims > 0] = 1
