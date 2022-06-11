import cv2
import numpy as np
import yolo_img

def mask_it(img,boxes,indexes):
    mask = img.copy()
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            mask[y:y+h,x:x+w] = 0
    
    return mask



if __name__ == "__main__":
    (img,boxes,indexes) = yolo_img.yolo_ret()
    mask = mask_it(img,boxes,indexes)
    img = cv2.resize(img, (1280,720))
    cv2.imshow("Image", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()