"""
5-12-18 Garrett Alston,
"""

import cv2 as cv
import cPickle
import numpy as np

training_x, training_y = cPickle.load(open("non_rotated_28x28.pkl", 'rb'))[0]

def save_Image(name, array):
    """Save images to path"""
    # Resize into a 2D array
    shaped_arr = np.reshape((array * 255).astype('uint8'),
                    (28, 28))
    cv.imwrite(name,shaped_arr)

for (x,y) in zip(training_x, training_y):
    count = 0
    filename = ''
    if y == 1:
        filname = "28x28_Imgs/crater%s.png" % count
    else:
        filename = "28x28_Imgs/non-crater%s.png" % count
    save_Image(filename, x)
