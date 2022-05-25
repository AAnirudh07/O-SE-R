import cv2
import argparse
import math
import numpy as np
from scipy.fft import dst

KERNEL = np.ones((3,3), np.uint8)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image whose text is to be separated from the UML class diagram box")
args = vars(ap.parse_args())

image = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
dst_img = cv2.Canny(image, 50, 200, None, 3)
#dst_img = cv2.erode(canny_img,KERNEL,iterations=1)

lines = cv2.HoughLines(dst_img, 1, np.pi / 180, 150, None, 0, 0)

for i in range(0, len(lines)):
            rho_l = lines[i][0][0]
            theta_l = lines[i][0][1]
            a_l = math.cos(theta_l)
            b_l = math.sin(theta_l)
            x0_l = a_l * rho_l
            y0_l = b_l * rho_l
            pt1_l = (int(x0_l + 1000*(-b_l)), int(y0_l + 1000*(a_l)))
            pt2_l = (int(x0_l - 1000*(-b_l)), int(y0_l - 1000*(a_l)))
            cv2.line(image, pt1_l, pt2_l, (0,0,255), 3, cv2.LINE_AA)

cv2.imshow("Image with lines", image)
cv2.waitKey()