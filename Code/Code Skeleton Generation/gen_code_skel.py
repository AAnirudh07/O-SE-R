#classes: x and y coordinates of centers of classes
#inh: x and y coordinates of centers of inheritance symbols
#rectangles: boudning boxes of line segment groups

import math
from pickle import FALSE
import cv2
import numpy as np
import yolo_img
import maskbb
import find_endpoints
import get_class_inh_asc

inh_point_class = []
endpoints_per_group = []
endpoints_assc = []


def find_nearest_class(inh, classes):
    for inh_point in inh:
        nearest_distance = float('inf')
        class_no = 0
        for class_point in range(len(classes)):
            if np.linalg.norm(np.array(classes[class_point])-np.array(inh_point)) < nearest_distance:
                class_no = class_point
                nearest_distance = np.linalg.norm(np.array(classes[class_point])-np.array(inh_point))
        inh_point_class.append(class_no)



def find_nearest(point, classes, inh):
    nearest_distance = float('inf')
    type = ""
    for class_point in range(len(classes)):
        print(np.linalg.norm(np.array(classes[class_point])-np.array(point)))
        if np.linalg.norm(np.array(classes[class_point])-np.array(point)) < nearest_distance:
            nearest_distance = np.linalg.norm(np.array(classes[class_point])-np.array(point))
            type = ("class",class_point)

    for inh_point in range(len(inh)):
        if np.linalg.norm(np.array(inh[inh_point])-np.array(point)) < nearest_distance:
            nearest_distance = np.linalg.norm(np.array(inh[inh_point])-np.array(point))
            type = ("inheritance",inh_point)

    return type


#main() function
with open("D:/Projects/O-SE-R/Dataset/images/classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

(img,boxes,indexes,class_ids) = yolo_img.yolo_ret()
(classes,inh,mask) = maskbb.mask_it(img,boxes,indexes,class_names,class_ids)
(contour_img,rectangles) = get_class_inh_asc.find_contours(mask)

find_nearest_class(inh,classes)
print(classes)

for rectangle in rectangles:
    endpoints_per_group.append(find_endpoints.find_endpoints(mask[rectangle[1]:rectangle[3],rectangle[0]:rectangle[2]]))
