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


def find_nearest(inh, classes):
    for inh_point in inh:
        nearest_distance = float('inf')
        class_no = 0
        for class_point in range(len(classes)):
            if np.linalg.norm(np.array(classes[class_point])-np.array(inh_point)) < nearest_distance:
                class_no = class_point
                nearest_distance = np.linalg.norm(np.array(classes[class_point])-np.array(inh_point))
        inh_point_class.append(class_no)



def find_nearest_class(point, found, classes):
    nearest_distance = float('inf')
    type = ""
    for class_point in range(len(classes)):
        if np.linalg.norm(np.array(classes[class_point])-np.array(point)) < nearest_distance and class_point not in found:
            nearest_distance = np.linalg.norm(np.array(classes[class_point])-np.array(point))
            type = ("class",class_point)
    return type

#main() function
with open("D:/Projects/O-SE-R/Dataset/images/classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

(img,boxes,indexes,class_ids) = yolo_img.yolo_ret()
(classes,inh,mask) = maskbb.mask_it(img,boxes,indexes,class_names,class_ids)
(contour_img,rectangles) = get_class_inh_asc.find_contours(mask)

find_nearest(inh,classes)
print(classes,inh)
cv2.imshow("mask",mask)
cv2.waitKey()

for rectangle in rectangles:
    mask_1 = np.zeros(img.shape, dtype=np.uint8)
    mask_1[rectangle[1]:rectangle[3],rectangle[0]:rectangle[2]] = mask[rectangle[1]:rectangle[3],rectangle[0]:rectangle[2]]
    endpoints_per_group.append(find_endpoints.find_endpoints(mask_1))

for endpoints in endpoints_per_group: 
    inh_flag = FALSE
    curr_group_assc = []
    found_points = []

    for point in endpoints:
        print(point)
        near = find_nearest_class(point, found_points, classes)
        found_points.append(near[1])
        curr_group_assc.append(near)
    endpoints_assc.append(curr_group_assc)

print(endpoints_assc)