from multiprocessing.connection import wait
import cv2
import numpy as np
import yolo_img
import maskbb
import find_endpoints

def find_contours(img):
    kernel = np.ones((5,5),np.uint8)
    rect_ret = []
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000 and cv2.contourArea(cnt) < 40000: 
            x, y, w, h = cv2.boundingRect(cnt)
            rect_ret.append([x,y,x+w,y+h])
            print(x,y)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    img = cv2.resize(img,(1280,720))
    return (img,rect_ret)
if __name__ == "__main__":
    with open("D:/Projects/O-SE-R/Dataset/images/classes.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    (img,boxes,indexes,class_ids) = yolo_img.yolo_ret()
    (classes,inh,mask) = maskbb.mask_it(img,boxes,indexes,class_names,class_ids)
    cv2.imshow("Masked",mask)
    cv2.waitKey()
    (contour_img,rectangles) = find_contours(mask)
    cv2.imshow("Contours",contour_img)
    cv2.waitKey()
    for rectangle in rectangles:
        pass