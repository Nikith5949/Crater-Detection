"""
5-12-18, Nikith AnupKumar
"""


import os
import cv2 as cv
import random
#place this program in images folder to make it work

def randomscaling(srcd,destd, size, name):
    s = srcd + "/non-crater/"
    t = os.listdir(s)
    for t1 in t:
        print 1
        ran = 200 #random.randint(10,101)
        img = cv.imread(s + t1)
        img = cv.resize(img, (size, size))
        cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
        cv.imwrite(destd + '/non-crater/' + 'norm_' + name + "_" + str(t1), img)

    s = srcd + "/crater/"
    t = os.listdir(s)
    for t1 in t:
        print 2
        ran = 200 #random.randint(10,101)
        img = cv.imread(s + t1)
        img = cv.resize(img, (size, size))
        cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
        cv.imwrite(destd+'/crater/'+ 'norm_' + name + "_" + str(t1), img)

#relative adrres of directory that contains crater and non craters(inputs)
#it rotates and scales each image randomly
#it should be in image directory if u want to use it without changes
srcd = "/Users/flavioandrade/Desktop/Homework/480_Big_Data/Project/cs697_term_project_phaseI/Phase2/tile_3_24/"
destd = "28x28"
randomscaling(srcd,destd, 28, 'second_copy')
