import cv2
import pytesseract
import argparse
import img_resize
import numpy as np

kernel = np.ones((5,5), np.uint8)

def preprocess(image, target=30):
    # correct the text height
    ratio = max(0.5, min(1.5, target / image.shape[0]))
    image = cv2.resize(image,
                       dsize=None,
                       fx=ratio,
                       fy=ratio,
                       interpolation=cv2.INTER_LANCZOS4)

    # apply Otsu thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # invert image if more than 3/5 of pixels are black
    height, width = thresh.shape
    if (cv2.countNonZero(thresh) / (height * width)) < (2 / 5):
        thresh = cv2.bitwise_not(thresh)

    return thresh

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image to be OCR'd")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = preprocess(image)

print("Tesseract Loaded!")
data = pytesseract.image_to_data(image)
print(data)
