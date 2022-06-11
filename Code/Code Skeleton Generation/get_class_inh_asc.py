from multiprocessing.connection import wait
import cv2
import numpy as np
import yolo_img
import maskbb
import find_endpoints

def find_contours(img):
    kernel = np.ones((5,5),np.uint8)

    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000 and cv2.contourArea(cnt) < 40000: 
            x, y, w, h = cv2.boundingRect(cnt)
            print(x,y)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    img = cv2.resize(img,(1280,720))
    cv2.imshow("Image with separated contours", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    (img,boxes,indexes) = yolo_img.yolo_ret()
    mask = maskbb.mask_it(img,boxes,indexes)
    cv2.imshow("Masked",mask)
    cv2.waitKey()
    find_contours(mask)