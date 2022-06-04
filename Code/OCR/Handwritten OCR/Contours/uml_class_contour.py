import cv2
import numpy as np

kernel = np.ones((11,11),np.uint8)

img = cv2.imread('../ocr_test_img.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
ret, thresh_img = cv2.threshold(gray_img, 254, 255, cv2.THRESH_BINARY)
#dilated = cv2.dilate(thresh_img,kernel,iterations = 1)

contours, hierarchy = cv2.findContours(thresh_img.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    print(x,y)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

cv2.imshow("Image with rectangles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
