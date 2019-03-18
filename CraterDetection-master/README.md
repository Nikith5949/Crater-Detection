# CraterDetection



This repository implements a Convolutional Neural Network that classifies crater images on Mars. Once the network had been trained, we are able to use the sliding window method to crop a piece of a large image, send it to the classifier, and have it tell us wether a certain part of the image has a crater in it or not. We circle green for true, red for false.

This was a group project for the Big Data Analytics Course at UMASS Boston. I implemented this along with Garret Alston, Nikith AnupKumar, and Euclides Barahona.


To run the program you can do and see the classification of craters, open a terminal and do the following:

python sliding_window_detector.py tile3_24.pgm gt_tile3_24.csv Pickles/leaky_e24_0075_val0.9903-txt0.98.05.pkl

The first argument is the big image the sliding window will move over, the second is a list of crater locations and radius for the .pgm file and the third argument is a pickle of the network at a state when it had the best validation and test accuracy found.

It will take a moment for the program to load, and it will start with a smaller
sized image and keep resizing, similar to an updside down pyramid. This is so
the classifier can classify the smaller craters.
