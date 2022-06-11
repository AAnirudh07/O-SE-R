import cv2
import numpy as np
import yolo_img

#classes - index represents class name; [(centerx,centery)]
#inh - index represents inheritance symbol; [(centerx,centery)]

def mask_it(img,boxes,indexes,class_names,class_ids):
    classes = []
    inh = []
    mask = img.copy()
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(class_names[class_ids[i]])
            if label=="UML class":
                classes.append([((x+w)/2),((y+h)/2)])
            else:
                inh.append([((x+w)/2),((y+h)/2)])
            mask[y:y+h,x:x+w] = 0
    
    return (classes,inh,mask)



if __name__ == "__main__":
    with open("D:/Projects/O-SE-R/Dataset/images/classes.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    (img,boxes,indexes,class_ids) = yolo_img.yolo_ret()
    (classes,inh,mask) = mask_it(img,boxes,indexes,class_names,class_ids)
    print(classes)
    print(inh)
    img = cv2.resize(img, (1280,720))
    cv2.imshow("Image", mask)
    cv2.waitKey()
    cv2.destroyAllWindows()