"""
Nikith AnupKumar, Flavio Andrade

This program reads the crater and non-crater images from their directories and
does the following:

1.) Shuffle each set of image files randomly.
2.) Read 70% of crater files, put them in an array, and put their label, 1, in the label array.
4.) Read 70% of the non crater files, put them in an array, and put their label, 0, in the label array.
5.) Steps 3, 4 are repeated for rest of the crater and non crater files.
6.) The images are dumped into a pickle file as a tuple.
7.) The rest of the images are split into 15% validation and test data.
"""

import sys
import os
import pickle
import cv2 as cv
import numpy as np
# comment this line out or remove it if you do not have paths module.
from paths import *
from matplotlib import pyplot as plt


np.set_printoptions(threshold=np.nan)

def readImagesFromPath(src, label, filename, size):
    """
    This function divides the dataset into 70% training, 15% validation, and 15%
    test data, amongst the crater and non-crater images.
    """
    # crater images
    crater_files = os.listdir(src[0])
    np.random.shuffle(crater_files)

    # non crater images
    non_crater_files = os.listdir(src[1])
    np.random.shuffle(non_crater_files)
    #print("kkkkkkkkkkkggg")
    #print(crater_files)
    leng = len(crater_files)
    nleng = len(non_crater_files)
    # training set length
    # 70 % for crater and 70 % non crater
    tsl = int(.7 * leng)
    # 70 % length of non crater files
    ntsl = int(.7 * nleng)
    all_images = []
    labels = []
    # 70% crater for training set
    print("training data...")
    for image_file in range(tsl):
        #print (1)
        image = cv.imread(src[0] + crater_files[image_file])
        if image is None:
            print (src[0] + non_crater_files[image_file])
            print ("Image Not Valid: 70% Crater Training Set.")
            continue
        # convert to grayscale
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.normalize(image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
        image.astype(np.float32)
        #print (image.shape)
        if image_file == 0:
            all_images = np.reshape(image, (1, size * size))
        else:
            image = np.reshape(image, (1, size * size))
            all_images = np.append(all_images, image, axis=0)
        labels.append(label[0])
    # 70 % non-crater for training set
    for image_file in range(ntsl):
        #print(2)
        image = cv.imread(src[1] + non_crater_files[image_file])

        if image is None:
            print (src[1] + non_crater_files[image_file])
            print ("Image Not Valid: 70% Non-Crater Training Set.")
            continue
        # convert to grayscale
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.normalize(image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
        image.astype(np.float32)
        image = np.reshape(image, (1, size * size))
        all_images = np.append(all_images, image, axis=0)
        labels.append(label[1])

    training_data = (all_images, labels)

    all_test_images = []
    all_test_labels = []
    # range from tsl to tsl + .15(length)
    newtsl = tsl + int(.15 * leng)
    newntsl = ntsl + int(.15 * nleng)
    # 15% crater for testing
    print("test data...")
    for image_file in range(tsl, newtsl + 1):
        image = cv.imread(src[0] + crater_files[image_file])
        if image is None:
            print (src[0] + crater_files[image_file])
            print ("Image Not Valid: 15% Crater Testing Set.")
            continue
        # convert to grayscale
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.normalize(image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
        image.astype(np.float32)
        if image_file == tsl:
            all_test_images = np.reshape(image, (1, size * size))
        else:
            image = np.reshape(image, (1, size * size))
            all_test_images = np.append(all_test_images, image, axis=0)
        all_test_labels.append(label[0])

    # 15% non-crater for testing
    for image_file in range(ntsl, newntsl + 1):
        image = cv.imread(src[1] + non_crater_files[image_file])

        if image is None:
            print (src[1] + non_crater_files[image_file])
            print ("Image Not Valid: 15% Non-Crater Testing Set.")
            continue;
        # convert to grayscale
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.normalize(image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
        image.astype(np.float32)
        image = np.reshape(image, (1, size * size))

        all_test_images = np.append(all_test_images, image, axis=0)
        all_test_labels.append(label[1])

    test_data = (all_test_images, all_test_labels)

    all_validation_images = []
    all_validation_labels = []
    # 15% of the crater set for validation
    print("validation...")
    for image_file in range(newtsl, leng):

        image = cv.imread(src[0] + crater_files[image_file])

        if image is None:
            print (src[0] + crater_files[image_file])
            print ("Image Not Valid: 15% Validation Set.")
            continue
        # convert to grayscale
        image.astype(np.uint8)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.normalize(image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
        image.astype(np.float32)
        if image_file == newtsl:
            all_validation_images = np.reshape(image, (1, size * size))
        else:
            image = np.reshape(image, (1, size * size))
            all_validation_images = np.append(all_validation_images, image, axis=0)
        all_validation_labels.append(label[0])

    # 15% of the non-crater set for validation
    for image_file in range(newntsl, nleng):
        image = cv.imread(src[1] + non_crater_files[image_file])

        # convert to grayscale
        if image is None:
            print (src[1] + non_crater_files[image_file])
            print ("Image Not Valid: 15% Non-Crater Validation Set.")
            continue;
        # convert to grayscale

        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.normalize(image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
        image.astype(np.float32)
        image = np.reshape(image, (1, size * size))
        all_validation_images = np.append(all_validation_images, image, axis=0)
        all_validation_labels.append(label[1])

    validation_data = (all_validation_images, all_validation_labels)

    all_data = (training_data, validation_data, test_data)

    #phase2_data = open('phase2.txt', 'w')
    #print >> phase2_data, all_data
    #phase2_data.close()

    my_file = open(filename, 'wb')
    pickle.dump(all_data, my_file)
    my_file.close()


if __name__ == '__main__':
    paths = [CRATER_PATH, NON_CRATER_PATH]
    labels = [1, 0]
    readImagesFromPath(paths, labels, "28x28.pkl", 28)
