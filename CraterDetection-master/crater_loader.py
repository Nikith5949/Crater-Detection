"""
craterLoader.py
Flavio Andrade, Nikith AnupKumar, Garrett Alston, Euclides Barahona
4-16-18

This program reads a pickle file containing training and test data.
Each data set, training and test, is a tuple of the the respective images and their labels.
The training set consists of 70% of the crater and 70% of the non-crater images, and the test
data set consists of 30% of the crater and 30% of the non-crater images.

After the file is read, each image from the data set is turned into a column vector
and put into a tuple consisting of the image itself, and its label. This is done for
the test and the training data. The final values are then returned.

Phase II--
load_crater_data_phaseII_wrapper(filename, size):
The training set consists of 70% of the crater and 70% of the non-crater images, and the test

test and validation set consist of 15% of of the crater and 15% of the non-crater images

"""

import cPickle
import pickle
import numpy as np
import theano
import theano.tensor as T

# call this to get image data and label
# take in a string filename
def load_crater_data_wrapper(filename):
    #all_data = [(all_images, labels), (all_test_images, all_test_labels)]
    my_file = open(filename, 'rb')
    training_data, test_data = pickle.load(my_file)
    my_file.close()
    # access the images of the tuple
    training_data_inputs = [np.reshape(x, (40000, 1)) for x in training_data[0]]
    # training data
    trd = zip(training_data_inputs, training_data[1])
    test_data_inputs = [np.reshape(x, (40000, 1)) for x in test_data[0]]
    # test data
    ted = zip(test_data_inputs, test_data[1])
    return (trd, ted)

def load_crater_data_phaseII_wrapper(filename, size):
    my_file = open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(my_file)
    my_file.close()

    training_data = shuffle_data(training_data, size)
    validation_data = shuffle_data(validation_data, size)

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]


def shuffle_data(data, size):
    data_input = [np.reshape(x, (size * size)) for x in data[0]]
    tup = zip(data_input, data[1])
    np.random.shuffle(tup)
    sep_x = [element[0] for element in tup]
    sep_y = [element[1] for element in tup]
    return (np.asarray(sep_x), sep_y)

if __name__ == '__main__':
    training_data, validation_data, test_data = load_crater_data_phaseII_wrapper("101x101.pkl", 101)
    print training_data[0]
