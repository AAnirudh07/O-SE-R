#!/usr/bin/python
import math
import numpy as np
import cv2

frame = cv2.imread("D:/Projects/O-SE-R/Dataset/images/1.jpg")    
frame = cv2.resize(frame, (1280,720))
#dictionary of all contours
contours = {}
#array of edges of polygon
approx = []
#scale of the text
scale = 2

#calculate angle
def angle(pt1,pt2,pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)


#grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#Canny
canny = cv2.Canny(frame,80,240,3)

#contours
contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for i in range(0,len(contours)):
    #approximate the contour with accuracy proportional to
    #the contour perimeter
    approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.02,True)

    #Skip small or non-convex objects
    if(abs(cv2.contourArea(contours[i]))<100 or not(cv2.isContourConvex(approx))):
        print("IN")
        continue

    #triangle
    if(len(approx) == 3):
        x,y,w,h = cv2.boundingRect(contours[i])
        cv2.putText(frame,'TRI',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,0,0),2,cv2.LINE_AA)
    elif(len(approx)>=4 and len(approx)<=6):
        #nb vertices of a polygonal curve
        vtc = len(approx)
        #get cos of all corners
        cos = []
        for j in range(2,vtc+1):
            cos.append(angle(approx[j%vtc],approx[j-2],approx[j-1]))
        #sort ascending cos
        cos.sort()
        #get lowest and highest
        mincos = cos[0]
        maxcos = cos[-1]

        x,y,w,h = cv2.boundingRect(contours[i])
        if(vtc==4):
            cv2.putText(frame,'RECT',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,0,0),2,cv2.LINE_AA)

cv2.imshow('frame',frame)
cv2.imshow('canny',canny)
cv2.waitKey()
cv2.destroyAllWindows()