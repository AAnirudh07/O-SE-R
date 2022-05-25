#!/usr/bin/python
import math
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image whose text is to be separated from the UML class diagram box")
args = vars(ap.parse_args())
frame = cv2.imread(args["image"])

#grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#Canny
canny = cv2.Canny(frame,80,240,3)

#contours
contours, hierarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
c = max(contours, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)
cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('frame',frame)
cv2.waitKey()
cv2.destroyAllWindows()