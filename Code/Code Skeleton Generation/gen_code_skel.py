#classes: x and y coordinates of centers of classes
#inh: x and y coordinates of centers of inheritance symbols
#rectangles: boudning boxes of line segment groups

import cv2
import numpy as np
import yolo_img
import maskbb
import find_endpoints
import get_class_inh_asc

endpoints_per_group = []

#main() function
with open("D:/Projects/O-SE-R/Dataset/images/classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
(img,boxes,indexes,class_ids) = yolo_img.yolo_ret()
(classes,inh,mask) = maskbb.mask_it(img,boxes,indexes,class_names,class_ids)
cv2.waitKey()
(contour_img,rectangles) = get_class_inh_asc.find_contours(mask)
cv2.waitKey()
for rectangle in rectangles:
    endpoints_per_group.append(find_endpoints.find_endpoints(mask[rectangle[1]:rectangle[3],rectangle[0]:rectangle[2]]))

#for endpoints in endpoints_per_group: 
#    print(endpoints)
#    print("---------------------------------------")

