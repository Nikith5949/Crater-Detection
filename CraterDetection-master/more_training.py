from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle
import theano
import os
import theano.tensor as T
from network3 import *
from plotdatafitting import plotDataFit
import crater_loader
# uncomment if you do not want graph to show up.
IMAGE_SIZE = 28
EPOCHS = 20
MB_SIZE = 1
ETA = .00005
RUNS = 1
LAMBDA_LENGTH = 1
GEN = 3

PICKLE = "Pickle_Stash/GEN2-LReLU28x28-ntwk-e7-val0.9719-tst0.9705.pkl"
#training_data, validation_data, test_data = network3.load_data_shared()

# PHASE II -- Crater Data
training_data, validation_data, test_data = \
crater_loader.load_crater_data_phaseII_wrapper("non_rotated_28x28.pkl", IMAGE_SIZE)


if __name__ == "__main__":
    net = cPickle.load(open(PICKLE, 'rb'))
    os.system('rm Pickles/*.pkl')
    net.SGD("GEN%s-LReLU28x28" % GEN , training_data, EPOCHS, MB_SIZE, ETA, validation_data, test_data, lmbda=0.0001)
