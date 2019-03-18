"""
5-12-18 Garret Alston, Flavio Andrade, Nikith AnupKumar
"""

import argparse
import time
import cv2
import imutils
import sys
from skimage.transform import pyramid_gaussian
import argparse
import cv2
import math
from craters import *
from crater_classifier import *
import numpy as np
import cPickle

START = -4
END   =  1
SCALE = .657

DRAWING_DATA_OUT = "HitData/hit_data.pkl"
GREEN = (0,255,0)
YELLOW = (0,255,255)
RED = (0,0,255)

#count = 0

class Pyramid:
    def __init__(self, image_file_path, step_size, swz, gt, classifier):
        self.pyramid_image = cv2.imread(image_file_path)
        self.original_image = cv2.imread(image_file_path)
        self.image_shape = self.pyramid_image.shape
        # number of rows
        self.original_width = self.image_shape[0]
        self.height = self.image_shape[0]
        # number of columns
        self.width = self.image_shape[1]
        self.swz = swz
        self.step_size = step_size;
        self.GTs = CraterList()
        self.original_image_resized = self.resize(self.original_image, 500)
        with open(gt, 'rb') as file:
            for row in file:
                entry = map(int, row.strip('\n').split(','))
                self.GTs.add(entry[:2], entry[2])
        self.classifier = classifier
        self.hitlist = []


    def slidingWindow(self):
        """
        Create an n x n window that moves across the big image creating smaller
        images to be passed to the convolutional neural network. Sliding windows
        start off small and grow looking for craters that may fit inside the current
        window.
        """
        for y in xrange(0, self.height, self.step_size):
            for x in xrange(0, self.width, self.step_size):
                yield (x, y, self.pyramid_image[y:y + self.swz, x:x + self.swz])

    def pyramid(self, minSize=(30, 30),start=(-6),end=6):
        for s in range(start, end, 1):
            w = int(self.original_width * math.exp(SCALE * s))
            self.pyramid_image = imutils.resize(self.original_image, width=w)
            self.height = self.pyramid_image.shape[1]
            self.width = self.pyramid_image.shape[0]
            #if self.height < minSize[1] or self.width < minSize[0]:
            #    break
            yield self.pyramid_image

    def runPyramid(self):
        for resized in self.pyramid(start=START, end=END):
            print "\n\n---------------------------------------------"
            hits = CraterHitList(self.original_width, self.swz, self.GTs)
            self.hitlist.append(hits)
            images, centerpoints, scales = [], [], []
            scaled_size = resized.shape[0]
            count = 0
            for (x, y, window) in self.slidingWindow():
                clone = resized.copy()
                clone1 = resized.copy()
                if x + self.swz <= clone.shape[0] and y + self.swz <= clone.shape[1]:
                    count += 1
                    cv2.rectangle(clone, (x, y), (x + self.swz, y + self.swz), YELLOW, 3)
                    # calculate the center point of the sliding window
                    center_point_x = (self.swz / 2) + x
                    center_point_y = (self.swz / 2) + y
                    true_scale = float(scaled_size)/self.original_width
                    image = self.crop(x, y, self.swz, clone1)
                    if len(images) == 0:
                        images = np.reshape(image, (1, self.swz * self.swz))
                    else:
                        image = np.reshape(image, (1, self.swz * self.swz))
                        images = np.append(images, image, axis=0)
                    centerpoints += [(center_point_x,center_point_y)]
                    scales += [true_scale]
                    # classify 500 images at a time
                    if count % 500 == 0:
                        tp, fp = self.classify_imgs(images, centerpoints, scales)
                        print "Found: TP's: %s" % tp
                        print "       FP's: %s" % fp
                        print
                        # reset everything
                        images, centerpoints, scales = [], [], []
                    self.display(clone)
            self.classify_imgs(images, centerpoints, scales)
            filename = 'Detection_Imgs/scaled-%sx%s.jpg' % (scaled_size, scaled_size)
            print 'Writing out image -- %s' % filename
            cv2.imwrite(filename, self.plot_hits(resized))


    def classify_imgs(self, images, centerpoints, scales):
        classifications = self.classifier.get_classifications(images)
        query = (images, centerpoints, scales, classifications)
        return self.hitlist[-1].add_hits(query)


    def plot_hits2(self, img, cw):
        # this tries to draw the circle on fixed size image, but it is not working properly
        # something might be off with the math
        tps = self.hitlist[-1].get_TP_hits()
        fps = self.hitlist[-1].get_FP_hits()
        # the clone is bigger so scale down
        if self.original_image_resized.shape[0] <= cw:
            for hit in tps:
                scale = hit.scale
                sv = scale * (cw / self.original_image_resized.shape[0])
                x, y, radius = int(hit.x / sv), int(hit.y / sv), int(self.swz/(2 / sv))
                cv2.circle(img, (x, y), radius,GREEN,1)
            for hit in fps:
                x, y, radius = int(hit.x / hit.scale), int(hit.y / hit.scale), int(self.swz/(2 / sv))
                cv2.circle(img, (x, y), radius,RED,1)
        else:
            for hit in tps:
                scale = hit.scale
                sv = scale * (cw / self.original_image_resized.shape[0])
                x, y, radius = int(hit.x * sv), int(hit.y * sv), int(self.swz/(2 * sv))
                cv2.circle(img, (x, y), radius, GREEN,1)
            for hit in fps:
                scale = hit.scale
                sv = scale * (cw / self.original_image_resized.shape[0])
                x, y, radius = int(hit.x * sv), int(hit.y * sv), int(self.swz/(2 * sv))
                cv2.circle(img, (x, y), radius, RED,1)
        return img

    def plot_hits(self, img):
        tps = self.hitlist[-1].get_TP_hits()
        fps = self.hitlist[-1].get_FP_hits()
        for hit in tps:
            cv2.circle(img, (hit.x, hit.y), self.swz/2,GREEN,1)
        for hit in fps:
            cv2.circle(img, (hit.x, hit.y), self.swz/2,RED,1)
        return img

    def crop(self,x,y,size,image):
        image = image[y:y+size, x:x+size]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        image.astype(np.float32)
        return image

    def shared(self, data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data, dtype=theano.config.floatX), borrow=True)
        return shared_x

    def display(self, img):
        self.plot_hits(img)
        cv2.imshow("Window", img)
        cv2.waitKey(1)
        # time.sleep(0.00001)

    def resize(self, img, w):
        return imutils.resize(img, width=w, height=w)


def get_network_size(net):
    return net.layers[0].image_shape[2]


def main():
    """
    currently images are getting classified multipple times. We should save
    locations and check them before classifications.
    """
    pickle = str(sys.argv[3])
    print "Getting network.... "
    net = cPickle.load(open(pickle, 'rb'))
    size = get_network_size(net)
    print "Network size = %s" % size
    image_path = str(sys.argv[1])
    gt = str(sys.argv[2])
    classifier = CraterClassifier(net)
    print "Running the pyramid..."
    pyramid = Pyramid(image_path, 10, size, gt, classifier)
    drawing_data = pyramid.runPyramid()
    cPickle.dump(pyramid.hitlist,open(DRAWING_DATA_OUT, 'wb'))

if __name__ == '__main__':
    main()
