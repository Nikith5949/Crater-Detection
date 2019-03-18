"""
Garrett Alston,
"""

import cPickle
import imutils
import cv2
import sys
from crater_classifier import *
from craters import *
import time

"""
TILE_24 = 185
TILE_25 = 214
"""

GREEN = (0,255,0)

pickle = str(sys.argv[1])
og_im = cv2.imread(str(sys.argv[2]))
fileout = str(sys.argv[3])
data = cPickle.load(open(pickle, 'rb'))

my_dict = {}
FPs = 0
TPs = 0
CRATERS_DETECTED = 0

for hl in data:
    FPs += len(hl.FPs)
    for tp in hl.TPs:
        if tp in my_dict:
            my_dict[tp] += hl.TPs[tp]
        else:
            my_dict[tp] = hl.TPs[tp]
CRATERS_DETECTED = len(my_dict)

avg_tp_hits = []
for k in my_dict:
    hits = my_dict[k]
    n = len(hits)
    TPs += n
    x = sum(float(hit.x )for hit in hits)/n
    y = sum(float(hit.y )for hit in hits)/n
    radius = sum(float(hit.radius)for hit in hits)/n
    scale = sum(float(hit.scale )for hit in hits)/n
    avg_tp_hits.append(([x/scale,y/scale], radius/scale))

for a in avg_tp_hits:
    x = int(a[0][0])
    y = int(a[0][1])
    r = int(a[1])
    cv2.circle(og_im, (x,y), r,GREEN,3)


cv2.imwrite(fileout, og_im)

print "\nFP = %s TP = %s Detetced = %s " % (FPs, TPs, CRATERS_DETECTED)
