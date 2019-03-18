from plotdatafitting import plotDataFit, plotData
from collections import Counter
import matplotlib.pyplot as plt
import theano.tensor as T
from network3 import *
matplotlib.use('Agg')
import crater_loader
import numpy as np
import matplotlib
import cPickle
import theano

IMAGE_SIZE = 101
EPOCHS = 50
MB_SIZE = 1
ETA = .00075
RUNS = 1
LAMBDA_LENGTH = 1

PICKLE = "Pickles/elu-network%sx%s" % (IMAGE_SIZE, IMAGE_SIZE)
training_data, validation_data, test_data = \
crater_loader.load_crater_data_phaseII_wrapper("101x101.pkl", IMAGE_SIZE)
total_validation_accuracies = []
total_test_accuracies = []
total_cost_accuracies = []


def leakyrelu():
    net = None
    for j in range(RUNS):
        print "num %s, leaky relu, with regularization %s" % (j, 0.001)
        net = Network([
            ConvPoolLayer(image_shape=(MB_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE),
                          filter_shape=(5, 1, 15, 15),
                          poolsize=(3, 3),
                          activation_fn=LReLU),
            ConvPoolLayer(image_shape=(MB_SIZE, 5, 29, 29),
                          filter_shape=(10, 5, 2, 2),
                          poolsize=(2, 2),
                          activation_fn=LReLU),
            FullyConnectedLayer(n_in=10*14*14, n_out=200, activation_fn=LReLU),
            FullyConnectedLayer(n_in=200, n_out=200, activation_fn=LReLU),
            FullyConnectedLayer(n_in=200, n_out=100, activation_fn=LReLU),
            SoftmaxLayer(n_in=100, n_out=2)], MB_SIZE)
        net.SGD("leaky0075_35", training_data, EPOCHS, MB_SIZE, ETA, validation_data, test_data, lmbda=0.001)
        total_validation_accuracies.append(net.validation_accuracies)
        total_test_accuracies.append(net.test_accuracies)
        total_cost_accuracies.append(net.cost)
    return net


def elu():
    net = None
    for j in range(RUNS):
        print "num %s, leaky relu, with regularization %s" % (j, 0.0001)
        net = Network([
            ConvPoolLayer(image_shape=(MB_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE),
                          filter_shape=(5, 1, 3, 3),
                          poolsize=(2, 2),
                          activation_fn=LReLU),
            ConvPoolLayer(image_shape=(MB_SIZE, 5, 13, 13),
                          filter_shape=(3, 5, 2, 2),
                          poolsize=(2, 2),
                          activation_fn=LReLU),
            FullyConnectedLayer(n_in=3*6*6, n_out=36, activation_fn=LReLU),
            FullyConnectedLayer(n_in=36, n_out=12, activation_fn=LReLU),
            FullyConnectedLayer(n_in=12, n_out=6, activation_fn=LReLU),
            SoftmaxLayer(n_in=6, n_out=2)], MB_SIZE)
        net.SGD("LReLU_28x28", training_data, EPOCHS, MB_SIZE, ETA, validation_data, test_data, lmbda=0.001)
        total_validation_accuracies.append(net.validation_accuracies)
        total_test_accuracies.append(net.test_accuracies)
    return net


def leakyrelu28x28():
    td, vd, ted = \
    crater_loader.load_crater_data_phaseII_wrapper("28x28.pkl", 28)
    net = None
    for j in range(RUNS):
        print "num %s, leaky relu, with regularization %s" % (j, 0.001)
        net = Network([
            ConvPoolLayer(image_shape=(MB_SIZE, 1, 28, 28),
                          filter_shape=(5, 1, 3, 3),
                          poolsize=(2, 2),
                          activation_fn=LReLU),
            ConvPoolLayer(image_shape=(MB_SIZE, 5, 13, 13),
                          filter_shape=(10, 5, 2, 2),
                          poolsize=(2, 2),
                          activation_fn=LReLU),
            FullyConnectedLayer(n_in=10*6*6, n_out=200, activation_fn=LReLU),
            FullyConnectedLayer(n_in=200, n_out=200, activation_fn=LReLU),
            FullyConnectedLayer(n_in=200, n_out=100, activation_fn=LReLU),
            SoftmaxLayer(n_in=100, n_out=2)], MB_SIZE)
        net.SGD("leaky28x28", td, EPOCHS, MB_SIZE, ETA, vd, ted, lmbda=0.001)
        total_validation_accuracies.append(net.validation_accuracies)
        total_test_accuracies.append(net.test_accuracies)
    return net


def flattenArray(two_d):
    return [element for array in two_d for element in array]


def run_experiments():
    net = leakyrelu()
    tta = flattenArray(total_test_accuracies)
    tva = flattenArray(total_validation_accuracies)
    tca = flattenArray(total_cost_accuracies)
    plotDataFit(tta, tva, len(tta), 1, "LReLU_101x101_25_Epochs-0075");
    plotData(len(tca), tca, 1, "Cost", "LReLU_101x101_25_Epochs_Cost-0075")
